from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from app.model.model import predict_pipeline
from app.model.model import __version__ as model_version
from typing import Literal


app = FastAPI()


class CustomerData(BaseModel):
    CreditScore: int
    Geography: Literal['France', 'Spain', 'Germany']  # Only allows these values
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    Surname: str

    @validator('CreditScore')
    def validate_credit_score(cls, v):
        if not 300 <= v <= 850:
            raise ValueError('Credit score must be between 300 and 850')
        return v
    
    @validator('Age')
    def validate_age(cls, v):
        if not 18 <= v <= 100:
            raise ValueError('Age must be between 18 and 100')
        return v

    @validator('HasCrCard', 'IsActiveMember')
    def validate_binary(cls, v):
        if v not in (0, 1):
            raise ValueError('Value must be 0 or 1')
        return v


class PredictionOut(BaseModel):
    churn_probability: float
    will_churn: bool


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}


@app.post("/predict", response_model=PredictionOut)
def predict(payload: CustomerData):
    prediction = predict_pipeline(payload.dict())
    return prediction
