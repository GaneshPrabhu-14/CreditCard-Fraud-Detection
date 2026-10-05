import os

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/credit_card.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
SCALER_PATH = os.path.join(MODEL_DIR, "kmeans_scaler.joblib")
MODEL_PATH = os.path.join(MODEL_DIR, "kmeans_model.joblib")
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

SUSPICIOUS_QUANTILE = 0.90
HIGH_RISK_QUANTILE = 0.97


def _centroid_distance(model, values):
    return model.transform(values).min(axis=1)


def train_and_evaluate_kmeans(
    test_size: float = 0.20,
    n_clusters: int = 3,
    dataset: pd.DataFrame | None = None,
):
    """Train K-means and score rows by their distance from the nearest centroid."""
    if not 0 < test_size < 1:
        raise ValueError("Test size must be greater than 0 and less than 1.")
    if n_clusters < 1:
        raise ValueError("The number of clusters must be at least 1.")

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
    if n_clusters > len(X_train):
        raise ValueError(f"The number of clusters cannot exceed the {len(X_train)} training rows.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_model.fit(X_train_scaled)
    training_scores = _centroid_distance(cluster_model, X_train_scaled)
    suspicious_threshold = float(np.quantile(training_scores, SUSPICIOUS_QUANTILE))
    high_risk_threshold = float(np.quantile(training_scores, HIGH_RISK_QUANTILE))
    anomaly_scores = _centroid_distance(cluster_model, X_test_scaled)
    predictions = (anomaly_scores >= high_risk_threshold).astype(int)
    risk_levels = np.select(
        [anomaly_scores >= high_risk_threshold, anomaly_scores >= suspicious_threshold],
        ["HIGH RISK", "SUSPICIOUS"],
        default="NORMAL",
    )
    cluster_labels = cluster_model.predict(X_test_scaled)

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
        {
            "pca1": float(point[0]),
            "pca2": float(point[1]),
            "is_anomaly": int(is_anomaly),
            "risk_level": str(risk_level),
            "anomaly_score": float(score),
            "cluster_id": int(cluster_id),
        }
        for point, is_anomaly, risk_level, score, cluster_id in zip(
            pca_coordinates,
            predictions,
            risk_levels,
            anomaly_scores,
            cluster_labels,
        )
    ]

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(
        {
            "model": cluster_model,
            "suspicious_threshold": suspicious_threshold,
            "high_risk_threshold": high_risk_threshold,
            "n_clusters": n_clusters,
        },
        MODEL_PATH,
    )

    return {
        "status": "Success",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "n_clusters": n_clusters,
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
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": round(roc_auc, 4),
        },
        "pca_scatter": pca_scatter,
    }


def predict_kmeans_account(data: dict):
    """Score one account by its distance from the nearest saved cluster centroid."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("K-means model files not found. Train the K-means model first.")

    scaler = joblib.load(SCALER_PATH)
    saved_model = joblib.load(MODEL_PATH)
    if "suspicious_threshold" not in saved_model or "high_risk_threshold" not in saved_model:
        raise FileNotFoundError("The saved K-means model is outdated. Retrain it before assessing accounts.")
    input_frame = pd.DataFrame([data], columns=FEATURE_COLUMNS)
    scaled_input = scaler.transform(input_frame)
    anomaly_score = float(_centroid_distance(saved_model["model"], scaled_input)[0])
    is_anomaly = anomaly_score >= saved_model["high_risk_threshold"]
    is_suspicious = anomaly_score >= saved_model["suspicious_threshold"]
    risk_level = "HIGH RISK" if is_anomaly else "SUSPICIOUS" if is_suspicious else "NORMAL"

    return {
        "is_anomaly": bool(is_anomaly),
        "risk_level": risk_level,
        "status": "HIGH RISK ANOMALY" if is_anomaly else risk_level,
        "decision_score": round(anomaly_score, 4),
        "score_direction": "Greater distance from the nearest centroid indicates higher anomaly risk",
        "threshold": round(float(saved_model["high_risk_threshold"]), 4),
        "suspicious_threshold": round(float(saved_model["suspicious_threshold"]), 4),
    }
