from fastapi import FastAPI
import pandas as pd
import joblib
from src.api.pydantic_models import InputData

model = joblib.load("data/processed/best_model.pkl")

app = FastAPI(title="Credit Risk API")

@app.get("/")
def root():
    return {"message": "Credit Risk API is running."}

@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.model_dump()])
    prob = model.predict_proba(df)[0][1]
    return {"risk_probability": prob}
