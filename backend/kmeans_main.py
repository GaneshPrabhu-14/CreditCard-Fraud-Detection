from io import BytesIO
import os
import sys

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from kmeans_pipeline import predict_kmeans_account, train_and_evaluate_kmeans

app = FastAPI(title="K-Means Credit Card Anomaly Detection API", version="1.0")


class AccountPayload(BaseModel):
    BALANCE: float
    PURCHASES: float
    INSTALLMENTS_PURCHASES: float
    CASH_ADVANCE: float
    CREDIT_LIMIT: float
    PAYMENTS: float
    MINIMUM_PAYMENTS: float
    TENURE: int


class KMeansTrainConfig(BaseModel):
    test_size: float = 0.20
    n_clusters: int = 3


@app.get("/")
def root():
    return {"message": "K-means clustering anomaly detection API is running."}


@app.post("/train")
def train_model(config: KMeansTrainConfig):
    try:
        results = train_and_evaluate_kmeans(
            config.test_size,
            config.n_clusters,
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
    test_size: float = 0.20,
    n_clusters: int = 3,
    filename: str = "Uploaded CSV",
):
    content = await request.body()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded CSV is empty.")
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="CSV uploads must be 25 MB or smaller.")

    try:
        dataset = pd.read_csv(BytesIO(content))
        results = train_and_evaluate_kmeans(
            test_size,
            n_clusters,
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
        return predict_kmeans_account(payload.dict())
    except FileNotFoundError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
