# Credit Card Anomaly Detection

A Streamlit and FastAPI project for exploring unusual credit-card account activity. It includes two independent unsupervised anomaly detectors:

- **Isolation Forest** identifies accounts that are isolated from the broader feature distribution.
- **K-Nearest Neighbors (KNN)** scores accounts by their mean distance to nearby training examples. Larger distances indicate higher anomaly risk.

The KNN dashboard also provides a model switch and a same-dataset comparison workflow. KNN is used here as a distance-based anomaly detector, not as a clustering algorithm.

## Requirements

- Python 3.10 or newer
- pip

Install the dependencies from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents virtual-environment activation, use `\.venv\Scripts\python.exe` in place of `python` in the commands below.

## Run the Isolation Forest App

Start the API and dashboard in separate terminals from the project root:

```powershell
python -m uvicorn backend.main:app --reload --port 8000
```

```powershell
python -m streamlit run frontend/app.py --server.port 8501
```

Open [http://localhost:8501](http://localhost:8501). The API root is at [http://localhost:8000](http://localhost:8000), and its interactive API documentation is at [http://localhost:8000/docs](http://localhost:8000/docs).

## Run the KNN App

Start its separate API and dashboard in two more terminals:

```powershell
python -m uvicorn backend.knn_main:app --reload --port 8001
```

```powershell
python -m streamlit run frontend/knn_app.py --server.port 8502
```

Open [http://localhost:8502](http://localhost:8502). The model toggle switches between Isolation Forest and KNN. **Train and compare both** trains both models using the same selected dataset, contamination rate, and test split, then reports ROC-AUC and anomaly-class F1 side by side.

Keep each API and its dashboard running in separate terminals. The two apps use distinct model artifacts, so training one does not replace the other model's files.

## Training Data

Both dashboards can train from the bundled `data/credit_card.csv` file or an uploaded CSV. Uploads must be 25 MB or smaller, contain at least 10 rows, and include these numeric columns:

| Column | Description |
| --- | --- |
| `BALANCE` | Account balance |
| `PURCHASES` | Total purchases |
| `INSTALLMENTS_PURCHASES` | Installment purchases |
| `CASH_ADVANCE` | Cash advances |
| `CREDIT_LIMIT` | Credit limit |
| `PAYMENTS` | Total payments |
| `MINIMUM_PAYMENTS` | Minimum payments |
| `TENURE` | Account tenure in months |

`CUST_ID` and other extra columns are ignored. Missing values in required features are filled with each feature's training median. At least two rows must remain in the test split. For KNN, `k` must be smaller than the training-row count.

## Evaluation Notes

The dashboards show a confusion matrix, ROC curve and AUC, classification report, score distribution, and PCA projection. The classification report is calculated when training runs; changing the model selector or settings alone does not retrain it.

The dataset does not provide verified fraud labels. For evaluation only, the application creates reference labels using `CASH_ADVANCE > 4500` or `BALANCE > 8000`. These are heuristics, not confirmed fraud outcomes, so comparison results should not be interpreted as proof that one detector is universally better.

In the KNN comparison, ROC-AUC indicates how well anomaly scores rank the heuristic high-risk examples across thresholds. Anomaly-class F1 reflects precision and recall at the model's chosen cutoff. A small AUC difference may not indicate a meaningful overall winner.

## API Endpoints

Both APIs expose the same routes; use port `8000` for Isolation Forest and `8001` for KNN.

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/` | API health check |
| `POST` | `/train` | Train from the bundled CSV |
| `POST` | `/train/upload` | Train from a raw CSV request body |
| `POST` | `/predict` | Score one account |

Training options include `contamination` and `test_size`. The KNN API also accepts `n_neighbors`. The upload endpoint accepts these as query parameters and expects the CSV bytes in the request body.

## Model Artifacts

Training saves artifacts under `models/`:

- Isolation Forest: `anomaly_model.joblib`, `scaler.joblib`
- KNN: `knn_neighbor_model.joblib`, `knn_scaler.joblib`

Retraining a detector replaces that detector's artifacts.
