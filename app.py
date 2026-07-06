from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Iris Prediction API")

model = joblib.load("iris_classifier.joblib")

species = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def home():
    return {"message": "FastAPI is running!"}


@app.post("/predict")
def predict(data: IrisInput):

    df = pd.DataFrame([data.model_dump()])

    pred = int(model.predict(df)[0])

    return {
        "prediction": pred,
        "species": species[pred]
    }
