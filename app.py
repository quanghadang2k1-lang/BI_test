from fastapi import FastAPI, HTTPException
from datetime import datetime

from schemas import IrisInput
from model import predict_iris

app = FastAPI(
    title="Iris Prediction API",
    version="1.0"
)

@app.get("/")
def home():
    return {"message": "Welcome"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now()
    }

@app.post("/predict")
def predict(data: IrisInput):
    try:
        return predict_iris(data.model_dump())

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )