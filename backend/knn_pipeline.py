import os

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/credit_card.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
SCALER_PATH = os.path.join(MODEL_DIR, "knn_scaler.joblib")
MODEL_PATH = os.path.join(MODEL_DIR, "knn_neighbor_model.joblib")
FEATURE_COLUMNS = [
    "BALANCE",
    "PURCHASES",
    "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "MINIMUM_PAYMENTS",
    "TENURE",
]

os.makedirs(MODEL_DIR, exist_ok=True)


def _mean_neighbor_distance(model, values, n_neighbors):
    distances = model.kneighbors(values, n_neighbors=n_neighbors, return_distance=True)[0]
    return distances.mean(axis=1)


def train_and_evaluate_knn(
    contamination: float = 0.03,
    test_size: float = 0.20,
    n_neighbors: int = 10,
    dataset: pd.DataFrame | None = None,
):
    """Train a nearest-neighbor distance detector and return evaluation data."""
    if not 0 < contamination < 0.5:
        raise ValueError("Contamination must be greater than 0 and less than 0.5.")
    if not 0 < test_size < 1:
        raise ValueError("Test size must be greater than 0 and less than 1.")
    if n_neighbors < 1:
        raise ValueError("K must be at least 1.")

    df = pd.read_csv(DATA_PATH) if dataset is None else dataset.copy()
    missing_columns = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing_columns)}")
    if len(df) < 10:
        raise ValueError("CSV must contain at least 10 data rows to train and evaluate the model.")

    X = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    empty_columns = X.columns[X.isna().all()].tolist()
    if empty_columns:
        raise ValueError(f"Required columns contain no numeric values: {', '.join(empty_columns)}")
    X = X.fillna(X.median())

    X_train, X_test = train_test_split(X, test_size=test_size, random_state=42)
    if len(X_test) < 2:
        raise ValueError("The test split must contain at least two rows.")
    if n_neighbors >= len(X_train):
        raise ValueError(f"K must be less than the {len(X_train)} training rows.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    neighbor_model = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    neighbor_model.fit(X_train_scaled)
    training_scores = _mean_neighbor_distance(neighbor_model, None, n_neighbors)
    threshold = float(np.quantile(training_scores, 1 - contamination))
    anomaly_scores = _mean_neighbor_distance(neighbor_model, X_test_scaled, n_neighbors)
    predictions = (anomaly_scores > threshold).astype(int)

    y_true = (
        (X_test["CASH_ADVANCE"] > 4500) | (X_test["BALANCE"] > 8000)
    ).astype(int).to_numpy()

    cm = confusion_matrix(y_true, predictions, labels=[0, 1]).tolist()
    class_report = classification_report(
        y_true,
        predictions,
        labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        output_dict=True,
        zero_division=0,
    )
    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, anomaly_scores)
        roc_auc = float(auc(fpr, tpr))
    else:
        fpr, tpr, roc_auc = np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.5

    pca_coordinates = PCA(n_components=2).fit_transform(X_test_scaled)
    pca_scatter = [
        {"pca1": float(point[0]), "pca2": float(point[1]), "is_anomaly": int(label)}
        for point, label in zip(pca_coordinates, predictions)
    ]

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(
        {"model": neighbor_model, "threshold": threshold, "n_neighbors": n_neighbors},
        MODEL_PATH,
    )

    return {
        "status": "Success",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "n_neighbors": n_neighbors,
        "anomaly_threshold": threshold,
        "feature_columns": FEATURE_COLUMNS,
        "confusion_matrix": cm,
        "classification_report": class_report,
        "score_distribution": [
            {"anomaly_score": float(score), "is_anomaly": int(label)}
            for score, label in zip(anomaly_scores, predictions)
        ],
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": round(roc_auc, 4),
        },
        "pca_scatter": pca_scatter,
    }


def predict_knn_account(data: dict):
    """Score one account by its average distance to the saved training neighbors."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("KNN model files not found. Train the KNN model first.")

    scaler = joblib.load(SCALER_PATH)
    saved_model = joblib.load(MODEL_PATH)
    input_frame = pd.DataFrame([data], columns=FEATURE_COLUMNS)
    scaled_input = scaler.transform(input_frame)
    anomaly_score = float(
        _mean_neighbor_distance(
            saved_model["model"], scaled_input, saved_model["n_neighbors"]
        )[0]
    )
    is_anomaly = anomaly_score > saved_model["threshold"]

    return {
        "is_anomaly": bool(is_anomaly),
        "status": "HIGH RISK ANOMALY" if is_anomaly else "NORMAL",
        "decision_score": round(anomaly_score, 4),
        "score_direction": "Higher scores indicate higher anomaly risk",
        "threshold": round(float(saved_model["threshold"]), 4),
    }
