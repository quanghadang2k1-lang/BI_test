
import streamlit as st
import requests

st.set_page_config(page_title="Iris Classifier")

st.title("🌸 Iris Flower Classifier")

st.write("Enter flower measurements below.")

sepal_length = st.number_input(
    "Sepal Length",
    value=5.1
)

sepal_width = st.number_input(
    "Sepal Width",
    value=3.5
)

petal_length = st.number_input(
    "Petal Length",
    value=1.4
)

petal_width = st.number_input(
    "Petal Width",
    value=0.2
)

if st.button("Predict"):

    payload = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=payload
    )

    result = response.json()

    st.success(
        f"Prediction: {result['species']}"
    )

    st.json(result)
