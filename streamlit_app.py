import os
import requests
import streamlit as st

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)

st.set_page_config(page_title="Iris Classifier")

st.title("🌸 Iris Flower Classifier")

response = requests.post(
    f"{API_URL}/predict",
    json=payload
)

try:
    health = requests.get(f"{API_URL}/health", timeout=5)

    if health.status_code == 200:
        st.success("🟢 API Connected")
    else:
        st.error("🔴 API Error")

except Exception:
    st.error("🔴 API Offline")

if st.button("Predict"):

    response = requests.post(
        f"{API_URL}/predict",
        json=payload
    )

    if response.status_code == 200:

        result = response.json()

        st.success(
            f"Prediction: {result['species']}"
        )

        st.json(result)

    else:

        st.error("Prediction failed.")
