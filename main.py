from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Load trained model
model = joblib.load("fertilizer_model.pkl")

# Input schema
class InputData(BaseModel):
    Soil_Type: int
    Soil_pH: float
    Soil_Moisture: float
    Organic_Carbon: float
    Electrical_Conductivity: float
    Nitrogen_Level: float
    Phosphorus_Level: float
    Potassium_Level: float
    Temperature: float
    Humidity: float
    Rainfall: float
    Crop_Type: int
    Crop_Growth_Stage: int
    Season: int
    Irrigation_Type: int
    Previous_Crop: int
    Region: int
    Fertilizer_Used_Last_Season: int
    Yield_Last_Season: float

@app.get("/")
def home():
    return {"message": "AgriIntel API Running"}

@app.post("/predict")
def predict(data: InputData):

    input_data = pd.DataFrame([{
        "Soil_Type": data.Soil_Type,
        "Soil_pH": data.Soil_pH,
        "Soil_Moisture": data.Soil_Moisture,
        "Organic_Carbon": data.Organic_Carbon,
        "Electrical_Conductivity": data.Electrical_Conductivity,
        "Nitrogen_Level": data.Nitrogen_Level,
        "Phosphorus_Level": data.Phosphorus_Level,
        "Potassium_Level": data.Potassium_Level,
        "Temperature": data.Temperature,
        "Humidity": data.Humidity,
        "Rainfall": data.Rainfall,
        "Crop_Type": data.Crop_Type,
        "Crop_Growth_Stage": data.Crop_Growth_Stage,
        "Season": data.Season,
        "Irrigation_Type": data.Irrigation_Type,
        "Previous_Crop": data.Previous_Crop,
        "Region": data.Region,
        "Fertilizer_Used_Last_Season": data.Fertilizer_Used_Last_Season,
        "Yield_Last_Season": data.Yield_Last_Season
    }])

    prediction = model.predict(input_data)

    return {
        "recommended_fertilizer": str(prediction[0])
    }
