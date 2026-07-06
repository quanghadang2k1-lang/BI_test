from fastapi import FastAPI
from model import load_model
from schemas import IrisFeatures

import pandas as pd
import os
from datetime import datetime

# ==========================
# Initialize FastAPI
# ==========================

app = FastAPI(
    title="Iris Classification API",
    description="Predict Iris flower species using a trained Logistic Regression model.",
    version="1.0.0"
)

# ==========================
# Load Model
# ==========================

model = load_model()

species_names = {
    0: "Setosa 🌸",
    1: "Versicolor 🌼",
    2: "Virginica 🌺"
}

# ==========================
# Prediction Log
# ==========================

LOG_FILE = "prediction_log.csv"


def log_prediction(features: IrisFeatures, prediction: str):
    """
    Save every prediction to a CSV file.
    This file will later be used by Evidently.
    """

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sepal_length": features.sepal_length,
        "sepal_width": features.sepal_width,
        "petal_length": features.petal_length,
        "petal_width": features.petal_width,
        "prediction": prediction
    }

    df = pd.DataFrame([row])

    if os.path.exists(LOG_FILE):
        df.to_csv(
            LOG_FILE,
            mode="a",
            header=False,
            index=False
        )
    else:
        df.to_csv(
            LOG_FILE,
            index=False
        )


# ==========================
# Root Endpoint
# ==========================

@app.get("/")
def home():
    return {
        "message": "Welcome"
    }


# ==========================
# Health Check
# ==========================

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ==========================
# Prediction Endpoint
# ==========================

@app.post("/predict")
def predict(data: IrisFeatures):

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(features)[0]

    species = species_names[prediction]

    # Log prediction for monitoring
    log_prediction(data, species)

    return {
        "prediction": int(prediction),
        "species": species,
        "model": "Logistic Regression",
        "version": "1.0.0"
    }
