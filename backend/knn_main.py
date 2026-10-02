from io import BytesIO
import os
import sys

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from knn_pipeline import predict_knn_account, train_and_evaluate_knn

app = FastAPI(title="KNN Credit Card Anomaly Detection API", version="1.0")


class AccountPayload(BaseModel):
    BALANCE: float
    PURCHASES: float
    INSTALLMENTS_PURCHASES: float
    CASH_ADVANCE: float
    CREDIT_LIMIT: float
    PAYMENTS: float
    MINIMUM_PAYMENTS: float
    TENURE: int


class KNNTrainConfig(BaseModel):
    contamination: float = 0.03
    test_size: float = 0.20
    n_neighbors: int = 10


@app.get("/")
def root():
    return {"message": "KNN nearest-neighbor anomaly detection API is running."}


@app.post("/train")
def train_model(config: KNNTrainConfig):
    try:
        results = train_and_evaluate_knn(
            config.contamination,
            config.test_size,
            config.n_neighbors,
        )
        results["dataset_name"] = "Bundled credit-card sample"
        return results
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/train/upload")
async def train_uploaded_model(
    request: Request,
    contamination: float = 0.03,
    test_size: float = 0.20,
    n_neighbors: int = 10,
    filename: str = "Uploaded CSV",
):
    content = await request.body()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded CSV is empty.")
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="CSV uploads must be 25 MB or smaller.")

    try:
        dataset = pd.read_csv(BytesIO(content))
        results = train_and_evaluate_knn(
            contamination,
            test_size,
            n_neighbors,
            dataset,
        )
        results["dataset_name"] = filename
        return results
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise HTTPException(status_code=400, detail=f"Could not read the uploaded CSV: {error}")
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/predict")
def predict_anomaly(payload: AccountPayload):
    try:
        return predict_knn_account(payload.dict())
    except FileNotFoundError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
