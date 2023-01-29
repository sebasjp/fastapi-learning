import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib

class HouseInput(BaseModel):
    houseage: int = Field(..., le=100)
    averooms: int = Field(..., le=10)
    avebedrms: int = Field(..., ge=1, le=2)

# Ouput for data validation
class Output(BaseModel):
    prediction: float

def get_model_response(input):

    data = input.dict()
    X = pd.DataFrame(data, index=[0])
    
    scaler = joblib.load("model_files/scaler.pkl")
    model = joblib.load("model_files/model.pkl")    
    X[X.columns] = scaler.transform(X)
    prediction = model.predict(X)[0]

    result = {"prediction": prediction}

    return result

app = FastAPI()
@app.post("/predict", response_model=Output)
async def model_predict(input: HouseInput):
    """Predict with input"""
    response = get_model_response(input)
    return response