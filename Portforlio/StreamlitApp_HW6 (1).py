import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath
import joblib
import tarfile
import tempfile
from joblib import dump, load
from sklearn.pipeline import Pipeline

# Must be first Streamlit command
st.set_page_config(page_title="ML Deployment", layout="wide")

warnings.simplefilter("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# ── Optional imports (show error on screen instead of crashing) ───────────────
try:
    import boto3
    import sagemaker
    from sagemaker.predictor import Predictor
    from sagemaker.serializers import NumpySerializer
    from sagemaker.deserializers import NumpyDeserializer
    AWS_AVAILABLE = True
except Exception as e:
    st.error(f"AWS/SageMaker import failed: {e}")
    AWS_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception as e:
    st.warning(f"SHAP not available: {e}")
    SHAP_AVAILABLE = False

try:
    from src.feature_utils import extract_features
    FEATURES_AVAILABLE = True
except Exception as e:
    st.warning(f"Could not import feature_utils: {e}")
    FEATURES_AVAILABLE = False

# ── Secrets ───────────────────────────────────────────────────────────────────
try:
    aws_id       = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
    aws_secret   = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
    aws_token    = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
    aws_bucket   = st.secrets["aws_credentials"]["AWS_BUCKET"]
    aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]
    SECRETS_OK = True
except Exception as e:
    st.error(f"Secrets not configured: {e}")
    SECRETS_OK = False

# ── AWS sessions ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )

@st.cache_resource
def get_sm_session(_session):
    return sagemaker.Session(boto_session=_session)

if AWS_AVAILABLE and SECRETS_OK:
    try:
        session = get_session(aws_id, aws_secret, aws_token)
        sm_session = get_sm_session(session)
    except Exception as e:
        st.error(f"AWS session failed: {e}")
        AWS_AVAILABLE = False

# ── Features ──────────────────────────────────────────────────────────────────
@st.cache_data
def get_features():
    try:
        return extract_features()
    except Exception as e:
        return pd.DataFrame()

df_features = get_features() if FEATURES_AVAILABLE else pd.DataFrame()

# ── Model config ──────────────────────────────────────────────────────────────
MODEL_INFO = {
    "endpoint": aws_endpoint if SECRETS_OK else "",
    "explainer": 'explainer_sentiment.shap',
    "pipeline":  'finalized_sentiment_model.tar.gz',
    "keys":   ['AMZN', 'GOOG', 'NFLX', 'PredictedSentiment'],
    "inputs": [{"name": k, "type": "number", "min": -1.0, "max": 1.0,
                "default": 0.0, "step": 0.01}
               for k in ['AMZN', 'GOOG', 'NFLX', 'PredictedSentiment']]
}

# ── Helper functions ──────────────────────────────────────────────────────────
def load_pipeline(_session, bucket, key):
    s3_client = _session.client('s3')
    filename = MODEL_INFO["pipeline"]
    s3_client.download_file(Filename=filename, Bucket=bucket,
                            Key=f"{key}/{os.path.basename(filename)}")
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]
    return joblib.load(joblib_file)

def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client('s3')
    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)
    with open(local_path, "rb") as f:
        return load(f)

def call_model_api(input_df):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer()
    )
    try:
        raw_pred = predictor.predict(input_df)
        pred_val = pd.DataFrame(raw_pred).values[-1][0]
        mapping = {-1: "SELL", 0: "HOLD", 1: "BUY"}
        return mapping.get(pred_val), 200
    except Exception as e:
        return f"Error: {str(e)}", 500

def display_explanation(input_df, session, aws_bucket):
    if not SHAP_AVAILABLE:
        st.warning("SHAP not available.")
        return
    explainer_name = MODEL_INFO["explainer"]
    explainer = load_shap_explainer(
        session, aws_bucket,
        posixpath.join('explainer', explainer_name),
        os.path.join(tempfile.gettempdir(), explainer_name)
    )
    best_pipeline = load_pipeline(session, aws_bucket, 'sklearn-pipeline-deployment')
    preprocessing_pipeline = Pipeline(steps=best_pipeline.steps[:-2])
    input_df_transformed = preprocessing_pipeline.transform(input_df)
    feature_names = best_pipeline[:-2].get_feature_names_out()
    input_df_transformed = pd.DataFrame(input_df_transformed, columns=feature_names)
    shap_values = explainer(input_df_transformed)

    st.subheader("Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0, :, 0])
    st.pyplot(fig)
    top_feature = pd.Series(
        shap_values[0, :, 0].values,
        index=shap_values[0, :, 0].feature_names
    ).abs().idxmax()
    st.info(f"**Business Insight:** The most influential factor was **{top_feature}**.")

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("ML Deployment")

with st.form("pred_form"):
    st.subheader("Inputs")
    cols = st.columns(2)
    user_inputs = {}
    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp['name']] = st.number_input(
                inp['name'].replace('_', ' ').upper(),
                min_value=inp['min'], max_value=inp['max'],
                value=inp['default'], step=inp['step']
            )
    submitted = st.form_submit_button("Run Prediction")

if submitted:
    if not AWS_AVAILABLE or not SECRETS_OK:
        st.error("AWS is not configured correctly. Check the errors above.")
    else:
        data_row = [user_inputs[k] for k in MODEL_INFO["keys"]]
        input_df = pd.DataFrame([data_row], columns=MODEL_INFO["keys"])
        res, status = call_model_api(input_df)
        if status == 200:
            st.metric("Prediction Result", res)
            display_explanation(input_df, session, aws_bucket)
        else:
            st.error(res)
