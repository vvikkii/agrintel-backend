from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Load model
model = joblib.load("fertilizer_model.pkl")

# Input schema
class InputData(BaseModel):
    Nitrogen: float
    Phosphorus: float
    Potassium: float
    pH: float
    Moisture: float
    Temperature: float
    Humidity: float

@app.get("/")
def home():
    return {"message": "AgriIntel API Running"}

@app.post("/predict")
def predict(data: InputData):

    input_dict = {
        "Nitrogen": [data.Nitrogen],
        "Phosphorus": [data.Phosphorus],
        "Potassium": [data.Potassium],
        "pH": [data.pH],
        "Moisture": [data.Moisture],
        "Temperature": [data.Temperature],
        "Humidity": [data.Humidity]
    }

    df = pd.DataFrame(input_dict)

    prediction = model.predict(df)

    return {
        "recommended_fertilizer": str(prediction[0])
    }