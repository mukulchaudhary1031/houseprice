from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

app = FastAPI()

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Input Schema
class InputData(BaseModel):
    area: float

@app.get("/")
def home():
    return {"message": "House Price Prediction API is working"}

@app.post("/predict")
def predict_price(data: InputData):
    area_val = np.array([[data.area]])
    prediction = model.predict(area_val)[0]

    return {
        "Area": data.area,
        "Predicted Price (Lakhs)": round(float(prediction), 2)
    }