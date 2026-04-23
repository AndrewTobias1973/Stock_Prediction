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
import json as _json

from sklearn.pipeline import Pipeline


from joblib import dump, load

# ── Setup ─────────────────────────────────────────────────────────────────────
warnings.simplefilter("ignore")

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Load X_train baseline from Portfolio folder (uploaded to GitHub)
file_path = os.path.join(project_root, 'Portforlio/X_train.csv')
dataset = pd.read_csv(file_path, nrows=1)
dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]

# ── AWS credentials from Streamlit secrets ────────────────────────────────────
aws_id       = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret   = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token    = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket   = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# ── AWS Session ───────────────────────────────────────────────────────────────
@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )

session = get_session(aws_id, aws_secret, aws_token)

# ── Model configuration ───────────────────────────────────────────────────────
# keys  = top features shown in the UI
# inputs = widget spec for each key
MODEL_INFO = {
    "endpoint"  : aws_endpoint,
    "explainer" : "explainer_fraud.shap",
    "pipeline"  : "fine_tuned_pipeline.tar.gz",
    "keys"      : ['TransactionAmt', 'addr1', 'C1', 'C2', 'dist1'],
    "inputs"    : [
        {"name": "TransactionAmt", "type": "number", "min": 0.0,     "max": 30000.0, "default": 100.0,  "step": 1.0},
        {"name": "addr1",          "type": "number", "min": 0.0,     "max": 600.0,   "default": 315.0,  "step": 1.0},
        {"name": "C1",             "type": "number", "min": 0.0,     "max": 3000.0,  "default": 1.0,    "step": 1.0},
        {"name": "C2",             "type": "number", "min": 0.0,     "max": 3000.0,  "default": 1.0,    "step": 1.0},
        {"name": "dist1",          "type": "number", "min": 0.0,     "max": 10000.0, "default": 0.0,    "step": 1.0},
    ]
}

# ── Load pipeline and SHAP explainer from S3 ─────────────────────────────────
def load_pipeline(_session, bucket, key):
    s3_client = _session.client('s3')
    filename  = MODEL_INFO["pipeline"]
    s3_client.download_file(Filename=filename, Bucket=bucket,
                            Key=f"{key}/{os.path.basename(filename)}")
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        pkl_file = [f for f in tar.getnames() if f.endswith('.pkl')][0]
    return joblib.load(pkl_file)

def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client('s3')
    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)
    with open(local_path, "rb") as f:
        return load(f)

# ── Endpoint prediction ───────────────────────────────────────────────────────
def call_model_api(input_df):
    runtime = session.client('sagemaker-runtime', region_name='us-east-1')
    if isinstance(input_df, pd.DataFrame):
        payload = input_df.fillna(0).to_json(orient='records')
    elif isinstance(input_df, dict):
        payload = _json.dumps([{k: (list(v.values())[0] if isinstance(v, dict) else v)
                                for k, v in input_df.items()}])
    else:
        payload = _json.dumps(input_df)
    try:
        response = runtime.invoke_endpoint(
            EndpointName=MODEL_INFO["endpoint"],
            ContentType='application/json',
            Body=payload
        )
        result = _json.loads(response['Body'].read().decode())
        pred_val = result[-1] if isinstance(result, list) else result
        mapping  = {0: "Legitimate", 1: "Fraud"}
        return mapping.get(pred_val), 200
    except Exception as e:
        return f"Error: {str(e)}", 500

# ── SHAP explanation ──────────────────────────────────────────────────────────
def display_explanation(input_df, session, aws_bucket):
    import shap 
    explainer_name = MODEL_INFO["explainer"]
    explainer = load_shap_explainer(
        session, aws_bucket,
        posixpath.join('explainer', explainer_name),
        os.path.join(tempfile.gettempdir(), explainer_name)
    )

    best_pipeline = load_pipeline(session, aws_bucket, 'sklearn-pipeline-deployment')

    # Remove the classifier (last step) to get only the preprocessing steps
    preprocessing_pipeline = Pipeline(steps=best_pipeline.steps[:-1])
    input_df = pd.DataFrame(input_df)
    input_df_transformed = preprocessing_pipeline.transform(input_df)

    try:
        feature_names = best_pipeline[:-1].get_feature_names_out()
        input_df_transformed = pd.DataFrame(input_df_transformed, columns=feature_names)
    except Exception:
        pass  # feature names not critical for the plot

    shap_values = explainer(input_df_transformed)

    # Binary classifier: shap_values is 2D (no class axis)
    sv0 = shap_values[0]

    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(sv0)
    st.pyplot(fig)

    top_feature = (
        pd.Series(sv0.values, index=sv0.feature_names)
        .abs().idxmax()
    )
    st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")

# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fraud Detection – IEEE-CIS", layout="wide")
st.title("👨‍💻 IEEE-CIS Fraud Detection")
st.markdown("**INSC 30273-065 | Milestone 4 | Andrew Tobias**")

with st.form("pred_form"):
    st.subheader("Transaction Inputs")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp['name']] = st.number_input(
                inp['name'].replace('_', ' ').upper(),
                min_value=inp['min'],
                max_value=inp['max'],
                value=inp['default'],
                step=inp['step']
            )

    submitted = st.form_submit_button("Run Prediction")

# Use first row of X_train as the full-column baseline, override with user inputs
original = dataset.iloc[0:1].to_dict()
original.update(user_inputs)

if submitted:
    res, status = call_model_api(original)
    if status == 200:
        colour = "red" if res == "Fraud" else "green"
        st.markdown(f"### Prediction: :{colour}[{res}]")
        st.metric("Result", res)
        display_explanation(original, session, aws_bucket)
    else:
        st.error(res)
