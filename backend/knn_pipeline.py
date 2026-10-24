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

SUSPICIOUS_QUANTILE = 0.90
HIGH_RISK_QUANTILE = 0.97
os.makedirs(MODEL_DIR, exist_ok=True)


def _neighbor_distance(model, values):
    distances, _ = model.kneighbors(values)
    return distances.mean(axis=1)


def train_and_evaluate_knn(
    test_size: float = 0.20,
    n_neighbors: int = 5,
    dataset: pd.DataFrame | None = None,
):
    """Train KNN and score rows by their average distance to nearby accounts."""
    if not 0 < test_size < 1:
        raise ValueError("Test size must be greater than 0 and less than 1.")
    if n_neighbors < 1:
        raise ValueError("The number of neighbors must be at least 1.")

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
    if n_neighbors > len(X_train):
        raise ValueError(f"The number of neighbors cannot exceed the {len(X_train)} training rows.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    neighbor_model = NearestNeighbors(n_neighbors=n_neighbors)
    neighbor_model.fit(X_train_scaled)

    training_scores = _neighbor_distance(neighbor_model, X_train_scaled)
    suspicious_threshold = float(np.quantile(training_scores, SUSPICIOUS_QUANTILE))
    high_risk_threshold = float(np.quantile(training_scores, HIGH_RISK_QUANTILE))
    anomaly_scores = _neighbor_distance(neighbor_model, X_test_scaled)
    predictions = (anomaly_scores >= high_risk_threshold).astype(int)
    risk_levels = np.select(
        [anomaly_scores >= high_risk_threshold, anomaly_scores >= suspicious_threshold],
        ["HIGH RISK", "SUSPICIOUS"],
        default="NORMAL",
    )

    ranked_training_indices = np.argsort(training_scores)
    suspicious_index = ranked_training_indices[
        min(int(0.94 * len(ranked_training_indices)), len(ranked_training_indices) - 1)
    ]
    high_risk_index = ranked_training_indices[-1]
    dataset_examples = {
        "Suspicious account (dataset)": {
            column: (
                int(X_train.iloc[suspicious_index][column])
                if column == "TENURE"
                else float(X_train.iloc[suspicious_index][column])
            )
            for column in FEATURE_COLUMNS
        },
        "High-risk account (dataset)": {
            column: (
                int(X_train.iloc[high_risk_index][column])
                if column == "TENURE"
                else float(X_train.iloc[high_risk_index][column])
            )
            for column in FEATURE_COLUMNS
        },
    }

    y_true = ((X_test["CASH_ADVANCE"] > 4500) | (X_test["BALANCE"] > 8000)).astype(int).to_numpy()
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
        {
            "pca1": float(point[0]),
            "pca2": float(point[1]),
            "is_anomaly": int(is_anomaly),
            "risk_level": str(risk_level),
            "anomaly_score": float(score),
        }
        for point, is_anomaly, risk_level, score in zip(
            pca_coordinates, predictions, risk_levels, anomaly_scores
        )
    ]

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(
        {
            "model": neighbor_model,
            "suspicious_threshold": suspicious_threshold,
            "high_risk_threshold": high_risk_threshold,
            "n_neighbors": n_neighbors,
        },
        MODEL_PATH,
    )

    return {
        "status": "Success",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "n_neighbors": n_neighbors,
        "suspicious_threshold": suspicious_threshold,
        "anomaly_threshold": high_risk_threshold,
        "dataset_examples": dataset_examples,
        "feature_columns": FEATURE_COLUMNS,
        "confusion_matrix": cm,
        "classification_report": class_report,
        "score_distribution": [
            {"anomaly_score": float(score), "is_anomaly": int(label)}
            for score, label in zip(anomaly_scores, predictions)
        ],
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": round(roc_auc, 4)},
        "pca_scatter": pca_scatter,
    }


def predict_knn_account(data: dict):
    """Score one account by its average distance to the saved nearest neighbors."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("KNN model files not found. Train the KNN model first.")

    scaler = joblib.load(SCALER_PATH)
    saved_model = joblib.load(MODEL_PATH)
    if "suspicious_threshold" not in saved_model or "high_risk_threshold" not in saved_model:
        raise FileNotFoundError("The saved KNN model is outdated. Retrain it before assessing accounts.")
    input_frame = pd.DataFrame([data], columns=FEATURE_COLUMNS)
    scaled_input = scaler.transform(input_frame)
    anomaly_score = float(_neighbor_distance(saved_model["model"], scaled_input)[0])
    is_anomaly = anomaly_score >= saved_model["high_risk_threshold"]
    is_suspicious = anomaly_score >= saved_model["suspicious_threshold"]
    risk_level = "HIGH RISK" if is_anomaly else "SUSPICIOUS" if is_suspicious else "NORMAL"

    return {
        "is_anomaly": bool(is_anomaly),
        "risk_level": risk_level,
        "status": "HIGH RISK ANOMALY" if is_anomaly else risk_level,
        "decision_score": round(anomaly_score, 4),
        "score_direction": "Greater average distance from nearest neighbors indicates higher anomaly risk",
        "threshold": round(float(saved_model["high_risk_threshold"]), 4),
        "suspicious_threshold": round(float(saved_model["suspicious_threshold"]), 4),
    }
