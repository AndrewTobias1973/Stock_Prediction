import os
import json
import boto3
import streamlit as st

st.title("Stock Prediction App")

# Get endpoint name from Streamlit secrets / environment
ENDPOINT_NAME = os.getenv("SM_ENDPOINT_NAME", "")
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

st.write("Using region:", REGION)

if ENDPOINT_NAME == "":
    st.error("Missing SM_ENDPOINT_NAME. Set it as an environment variable or Streamlit secret.")
    st.stop()

st.success(f"Endpoint set: {ENDPOINT_NAME}")

# Simple inputs (you can rename these later to match your model features)
x1 = st.number_input("Feature 1", value=0.0)
x2 = st.number_input("Feature 2", value=0.0)
x3 = st.number_input("Feature 3", value=0.0)
x4 = st.number_input("Feature 4", value=0.0)

payload = f"{x1},{x2},{x3},{x4}\n"  # CSV row

if st.button("Predict"):
    runtime = boto3.client("sagemaker-runtime", region_name=REGION)

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="text/csv",
        Body=payload.encode("utf-8"),
    )

    result = response["Body"].read().decode("utf-8")
    st.subheader("Model Output")
    st.write(result)


