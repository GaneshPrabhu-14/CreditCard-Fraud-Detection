from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from io import BytesIO
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pipeline import train_and_evaluate_model, predict_single_account

app = FastAPI(title="Credit Card Anomaly Detection API", version="1.0")

class AccountPayload(BaseModel):
    BALANCE: float
    PURCHASES: float
    INSTALLMENTS_PURCHASES: float
    CASH_ADVANCE: float
    CREDIT_LIMIT: float
    PAYMENTS: float
    MINIMUM_PAYMENTS: float
    TENURE: int

class TrainConfig(BaseModel):
    contamination: float = 0.03
    test_size: float = 0.20

@app.get("/")
def root():
    return {"message": "Credit Card Anomaly Detection Backend is Running!"}

@app.post("/train")
def train_model(config: TrainConfig):
    try:
        results = train_and_evaluate_model(config.contamination, config.test_size)
        results["dataset_name"] = "Bundled credit-card sample"
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train/upload")
async def train_uploaded_model(
    request: Request,
    contamination: float = 0.03,
    test_size: float = 0.20,
    filename: str = "Uploaded CSV",
):
    content = await request.body()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded CSV is empty.")
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="CSV uploads must be 25 MB or smaller.")

    try:
        dataset = pd.read_csv(BytesIO(content))
        results = train_and_evaluate_model(contamination, test_size, dataset)
        results["dataset_name"] = filename
        return results
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as e:
        raise HTTPException(status_code=400, detail=f"Could not read the uploaded CSV: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict_anomaly(payload: AccountPayload):
    try:
        result = predict_single_account(payload.dict())
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))