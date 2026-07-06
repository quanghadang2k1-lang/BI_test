import logging
import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = joblib.load("iris_classifier.joblib")

species = {
    0: "Setosa 🌸",
    1: "Versicolor 🌼",
    2: "Virginica 🌺"
}

def predict_iris(data):

    logger.info(f"Prediction request: {data}")

    df = pd.DataFrame([data])

    pred = int(model.predict(df)[0])

    logger.info(f"Prediction result: {pred}")

    return {
        "prediction": pred,
        "species": species[pred],
        "model": "Iris Classifier",
        "version": "1.0.0"
    }