from fastapi import FastAPI     
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List
model = None

app = FastAPI(title="Heart Disease Prediction API",
              description="API for predicting heart disease using a trained XGBoost model",
              version="1.0.0"  )

  

@app.on_event("startup")
def load_model():
    global model
    model = joblib.load("ml/model.pkl")

@app.get("/")
def root():
    return {"message": "Welcome to the Heart Disease Prediction API!"}

class Features(BaseModel):
    data: List[float]

@app.post("/predict")
def predict(features: Features):
    x = np.array(features.data).reshape(1, -1)
    prediction = model.predict(x)
    probability = model.predict_proba(x)[:, 1]
    return {
        "prediction": int(prediction[0]),
        "probability": float(probability[0])
    }