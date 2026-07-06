import os
import requests
import streamlit as st

# ==========================
# Configuration
# ==========================

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)

st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

# ==========================
# Title
# ==========================

st.title("🌸 Iris Flower Classifier")
st.write("Predict the Iris flower species using a deployed FastAPI model.")

# ==========================
# API Health Check
# ==========================

try:
    health = requests.get(f"{API_URL}/health", timeout=5)

    if health.status_code == 200:
        st.success("🟢 API Connected")
    else:
        st.warning("🟡 API Responded Unexpectedly")

except Exception:
    st.error("🔴 Cannot connect to API")
    st.stop()

# ==========================
# User Inputs
# ==========================

st.subheader("Enter Flower Measurements")

sepal_length = st.number_input(
    "Sepal Length",
    min_value=0.0,
    max_value=10.0,
    value=5.1,
    step=0.1
)

sepal_width = st.number_input(
    "Sepal Width",
    min_value=0.0,
    max_value=10.0,
    value=3.5,
    step=0.1
)

petal_length = st.number_input(
    "Petal Length",
    min_value=0.0,
    max_value=10.0,
    value=1.4,
    step=0.1
)

petal_width = st.number_input(
    "Petal Width",
    min_value=0.0,
    max_value=10.0,
    value=0.2,
    step=0.1
)

payload = {
    "sepal_length": sepal_length,
    "sepal_width": sepal_width,
    "petal_length": petal_length,
    "petal_width": petal_width
}

# ==========================
# Prediction
# ==========================

if st.button("Predict", use_container_width=True):

    with st.spinner("Making prediction..."):

        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=10
            )

            if response.status_code == 200:

                result = response.json()

                st.success("Prediction Successful!")

                st.metric(
                    label="Predicted Species",
                    value=result["species"]
                )

                st.write("### API Response")

                st.json(result)

            else:

                st.error(
                    f"Prediction failed (HTTP {response.status_code})"
                )

                st.write(response.text)

        except requests.exceptions.RequestException as e:

            st.error("Failed to communicate with the API.")

            st.exception(e)
