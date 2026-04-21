import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath
import joblib
import tarfile
import tempfile
from joblib import load
from sklearn.pipeline import Pipeline

# Must be the very first Streamlit command
st.set_page_config(page_title="ML Deployment", layout="wide")

warnings.simplefilter("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# ── Model config (no AWS calls here) ─────────────────────────────────────────
MODEL_INFO = {
    "explainer": 'explainer_sentiment.shap',
    "pipeline":  'finalized_sentiment_model.tar.gz',
    "keys":   ['AMZN', 'GOOG', 'NFLX', 'PredictedSentiment'],
    "inputs": [{"name": k, "min": -1.0, "max": 1.0, "default": 0.0, "step": 0.01}
               for k in ['AMZN', 'GOOG', 'NFLX', 'PredictedSentiment']]
}

# ── Helper functions (only called on form submit) ─────────────────────────────
def get_aws_session():
    import boto3
    aws_id       = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
    aws_secret   = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
    aws_token    = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )

def get_endpoint():
    return st.secrets["aws_credentials"]["AWS_ENDPOINT"]

def get_bucket():
    return st.secrets["aws_credentials"]["AWS_BUCKET"]

def load_pipeline(s3_client, bucket, key):
    filename = MODEL_INFO["pipeline"]
    s3_client.download_file(Bucket=bucket, Key=f"{key}/{filename}", Filename=filename)
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]
    return joblib.load(joblib_file)

def load_shap_explainer(s3_client, bucket, local_path):
    explainer_name = MODEL_INFO["explainer"]
    s3_key = posixpath.join('explainer', explainer_name)
    if not os.path.exists(local_path):
        s3_client.download_file(Bucket=bucket, Key=s3_key, Filename=local_path)
    with open(local_path, "rb") as f:
        return load(f)

def call_model_api(input_df, session, endpoint):
    import sagemaker
    from sagemaker.predictor import Predictor
    from sagemaker.serializers import NumpySerializer
    from sagemaker.deserializers import NumpyDeserializer
    sm_session = sagemaker.Session(boto_session=session)
    predictor = Predictor(
        endpoint_name=endpoint,
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer()
    )
    raw_pred = predictor.predict(input_df)
    pred_val = pd.DataFrame(raw_pred).values[-1][0]
    mapping = {0: "SELL", 1: "HOLD", 2: "BUY"}
    return mapping.get(int(pred_val), str(pred_val))

def display_explanation(input_df, s3_client, bucket):
    import shap
    local_path = os.path.join(tempfile.gettempdir(), MODEL_INFO["explainer"])
    explainer = load_shap_explainer(s3_client, bucket, local_path)
    best_pipeline = load_pipeline(s3_client, bucket, 'sklearn-pipeline-deployment')
    preprocessing_pipeline = Pipeline(steps=best_pipeline.steps[:-2])
    input_df_transformed = preprocessing_pipeline.transform(input_df)
    feature_names = best_pipeline[:-2].get_feature_names_out()
    input_df_transformed = pd.DataFrame(input_df_transformed, columns=feature_names)
    shap_values = explainer(input_df_transformed)
    st.subheader("Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0, :, 0], show=False)
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
    try:
        session    = get_aws_session()
        endpoint   = get_endpoint()
        bucket     = get_bucket()
        s3_client  = session.client('s3')

        data_row = [user_inputs[k] for k in MODEL_INFO["keys"]]
        input_df = pd.DataFrame([data_row], columns=MODEL_INFO["keys"])

        with st.spinner("Running prediction..."):
            result = call_model_api(input_df, session, endpoint)

        st.metric("Prediction Result", result)

        with st.spinner("Loading explanation..."):
            display_explanation(input_df, s3_client, bucket)

    except Exception as e:
        st.error(f"Error: {e}")
