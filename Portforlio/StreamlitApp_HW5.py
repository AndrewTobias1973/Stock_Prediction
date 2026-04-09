import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath

import joblib
import tarfile
import tempfile

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer

from imblearn.pipeline import Pipeline
import shap

# Setup & Path Configuration
warnings.simplefilter("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Access the secrets
aws_id     = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token  = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# AWS Session Management
@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )

session = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# ── Model Configuration ───────────────────────────────────────────────────
MODEL_INFO = {
    "endpoint": aws_endpoint,
    "explainer": 'explainer_pca.shap',
    "pipeline":  'finalized_pca_model.tar.gz',
    "inputs": [{"name": "AMD", "type": "number", "min": 0.0, "default": 100.0, "step": 10.0}]
}

# ── Load SP500 dataset (cached so it only reads once per session) ─────────
@st.cache_data
def load_dataset():
    return pd.read_csv('./SP500Data.csv', index_col=0)

def build_features(amd_price):
    """
    Given an AMD stock price, find the closest historical date and build
    the feature vector (cumulative returns of all SP500 stocks except AMD).
    Returns the feature DataFrame and the matched date.
    """
    dataset = load_dataset()
    return_period = 5

    # Find the date in history where AMD's price is closest to the input
    closest_date = (dataset['AMD'] - float(amd_price)).abs().idxmin()

    # Build X exactly as in the notebook
    X = np.log(dataset.drop(['AMD'], axis=1)).diff(return_period)
    X = np.exp(X).cumsum()
    X.columns = [name + "_CR_Cum" for name in X.columns]

    return X.loc[[closest_date]], closest_date

# ── S3 Helpers ────────────────────────────────────────────────────────────
def load_pipeline(_session, bucket, key):
    s3_client = _session.client('s3')
    filename = MODEL_INFO["pipeline"]
    s3_client.download_file(
        Filename=filename,
        Bucket=bucket,
        Key=f"{key}/{os.path.basename(filename)}"
    )
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]
    return joblib.load(joblib_file)

def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client('s3')
    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)
    with open(local_path, "rb") as f:
        return shap.Explainer.load(f)

# ── Prediction Logic ──────────────────────────────────────────────────────
def call_model_api(amd_price):
    # Build the full feature vector from the AMD price
    feature_row, closest_date = build_features(amd_price)

    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer()
    )

    try:
        raw_pred = predictor.predict(feature_row.values)
        pred_val = float(pd.DataFrame(raw_pred).values[-1][0])
        return round(pred_val, 4), 200, closest_date
    except Exception as e:
        return f"Error: {str(e)}", 500, None

# ── Local Explainability ──────────────────────────────────────────────────
def display_explanation(amd_price, session, aws_bucket):
    explainer_name = MODEL_INFO["explainer"]
    explainer = load_shap_explainer(
        session,
        aws_bucket,
        posixpath.join('explainer', explainer_name),
        os.path.join(tempfile.gettempdir(), explainer_name)
    )

    # Rebuild features for the same price
    feature_row, _ = build_features(amd_price)

    # Load full pipeline and strip the final model step for preprocessing
    best_pipeline = load_pipeline(session, aws_bucket, 'sklearn-pipeline-deployment')

    # Pipeline 3 has 4 steps: imputer → scaler → kpca → model
    # Keep all steps except the last (model) for preprocessing
    preprocessing_pipeline = Pipeline(steps=best_pipeline.steps[:-1])
    input_transformed = preprocessing_pipeline.transform(feature_row)
    feature_names = best_pipeline[:-1].get_feature_names_out()
    input_transformed = pd.DataFrame(input_transformed, columns=feature_names)

    shap_values = explainer(input_transformed)

    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0], max_display=10)
    st.pyplot(fig)

    top_feature = feature_names[np.argmax(np.abs(shap_values[0].values))]
    st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")

# ── Streamlit UI ──────────────────────────────────────────────────────────
st.set_page_config(page_title="ML Deployment", layout="wide")
st.title("👨‍💻 ML Deployment")

with st.form("pred_form"):
    st.subheader("Inputs")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp['name']] = st.number_input(
                inp['name'].replace('_', ' ').upper(),
                min_value=inp['min'], value=inp['default'], step=inp['step']
            )

    submitted = st.form_submit_button("Run Prediction")

if submitted:
    amd_price = user_inputs['AMD']
    res, status, closest_date = call_model_api(amd_price)

    if status == 200:
        st.caption(f"Using closest matching historical date: {closest_date}")
        st.metric("Predicted AMD Cumulative Return", res)
        display_explanation(amd_price, session, aws_bucket)
    else:
        st.error(res)
