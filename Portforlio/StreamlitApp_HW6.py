import streamlit as st

st.set_page_config(page_title="ML Deployment", layout="wide")

MODEL_KEYS = ['AMZN', 'GOOG', 'NFLX', 'PredictedSentiment']

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("ML Deployment")

with st.form("pred_form"):
    st.subheader("Inputs")
    cols = st.columns(2)
    user_inputs = {}
    for i, key in enumerate(MODEL_KEYS):
        with cols[i % 2]:
            user_inputs[key] = st.number_input(
                key.replace('_', ' ').upper(),
                min_value=-1.0, max_value=1.0, value=0.0, step=0.01
            )
    submitted = st.form_submit_button("Run Prediction")

if submitted:
    try:
        import os, sys, warnings, posixpath, tarfile, tempfile
        import numpy as np
        import pandas as pd
        import joblib
        from joblib import load
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.pipeline import Pipeline
        import boto3
        import sagemaker
        from sagemaker.predictor import Predictor
        from sagemaker.serializers import NumpySerializer
        from sagemaker.deserializers import NumpyDeserializer
        import shap

        warnings.simplefilter("ignore")

        # Secrets
        aws_id       = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
        aws_secret   = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
        aws_token    = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
        aws_bucket   = st.secrets["aws_credentials"]["AWS_BUCKET"]
        aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

        # AWS session
        session = boto3.Session(
            aws_access_key_id=aws_id,
            aws_secret_access_key=aws_secret,
            aws_session_token=aws_token,
            region_name='us-east-1'
        )
        s3_client = session.client('s3')
        sm_session = sagemaker.Session(boto_session=session)

        # Prepare input
        data_row = [user_inputs[k] for k in MODEL_KEYS]
        input_df = pd.DataFrame([data_row], columns=MODEL_KEYS)

        # Call endpoint
        with st.spinner("Running prediction..."):
            predictor = Predictor(
                endpoint_name=aws_endpoint,
                sagemaker_session=sm_session,
                serializer=NumpySerializer(),
                deserializer=NumpyDeserializer()
            )
            raw_pred = predictor.predict(input_df)
            pred_val = int(pd.DataFrame(raw_pred).values[-1][0])
            mapping = {0: "SELL", 1: "HOLD", 2: "BUY"}
            result = mapping.get(pred_val, str(pred_val))

        st.metric("Prediction Result", result)

        # SHAP explanation
        with st.spinner("Loading explanation..."):
            pipeline_file = 'finalized_sentiment_model.tar.gz'
            s3_client.download_file(
                Bucket=aws_bucket,
                Key=f"sklearn-pipeline-deployment/{pipeline_file}",
                Filename=pipeline_file
            )
            with tarfile.open(pipeline_file, "r:gz") as tar:
                tar.extractall(path=".")
                joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]
            best_pipeline = joblib.load(joblib_file)

            explainer_name = 'explainer_sentiment.shap'
            local_explainer = os.path.join(tempfile.gettempdir(), explainer_name)
            if not os.path.exists(local_explainer):
                s3_client.download_file(
                    Bucket=aws_bucket,
                    Key=posixpath.join('explainer', explainer_name),
                    Filename=local_explainer
                )
            with open(local_explainer, "rb") as f:
                explainer = load(f)

            preprocessing_pipeline = Pipeline(steps=best_pipeline.steps[:-2])
            X_transformed = preprocessing_pipeline.transform(input_df)
            feature_names = best_pipeline[:-2].get_feature_names_out()
            X_transformed = pd.DataFrame(X_transformed, columns=feature_names)
            shap_values = explainer(X_transformed)

            st.subheader("Decision Transparency (SHAP)")
            fig, ax = plt.subplots(figsize=(10, 4))
            shap.plots.waterfall(shap_values[0, :, 0], show=False)
            st.pyplot(fig)

            top_feature = pd.Series(
                shap_values[0, :, 0].values,
                index=shap_values[0, :, 0].feature_names
            ).abs().idxmax()
            st.info(f"**Business Insight:** The most influential factor was **{top_feature}**.")

    except Exception as e:
        st.error(f"Error: {e}")
