import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/credit_card.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
MODEL_PATH = os.path.join(MODEL_DIR, "anomaly_model.joblib")
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

def train_and_evaluate_model(
    contamination: float = 0.03,
    test_size: float = 0.20,
    dataset: pd.DataFrame | None = None,
):
    """Trains Isolation Forest, computes evaluation metrics, and saves model artifacts."""
    df = pd.read_csv(DATA_PATH) if dataset is None else dataset.copy()
    missing_columns = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing_columns)}")

    X = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    empty_columns = X.columns[X.isna().all()].tolist()
    if empty_columns:
        raise ValueError(f"Required columns contain no numeric values: {', '.join(empty_columns)}")
    if len(X) < 10:
        raise ValueError("CSV must contain at least 10 data rows to train and evaluate the model.")
    X = X.fillna(X.median())
    
    X_train, X_test = train_test_split(X, test_size=test_size, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    iso_forest.fit(X_train_scaled)
    
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(iso_forest, MODEL_PATH)
    
    preds = iso_forest.predict(X_test_scaled)
    is_anomaly = [1 if p == -1 else 0 for p in preds]
    decision_scores = iso_forest.decision_function(X_test_scaled)
    
    y_true_test = ((X_test['CASH_ADVANCE'] > 4500) | (X_test['BALANCE'] > 8000)).astype(int).tolist()
    
    cm = confusion_matrix(y_true_test, is_anomaly).tolist()
    class_report = classification_report(
        y_true_test,
        is_anomaly,
        labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        output_dict=True,
        zero_division=0,
    )
    
    inverted_scores = -decision_scores
    fpr, tpr, _ = roc_curve(y_true_test, inverted_scores)
    roc_auc = float(auc(fpr, tpr))
    
    pca = PCA(n_components=2)
    X_test_pca = pca.fit_transform(X_test_scaled)
    
    pca_data = [
        {"pca1": float(coord[0]), "pca2": float(coord[1]), "is_anomaly": int(label)}
        for coord, label in zip(X_test_pca, is_anomaly)
    ]
    
    return {
        "status": "Success",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_columns": FEATURE_COLUMNS,
        "confusion_matrix": cm,
        "classification_report": class_report,
        "score_distribution": [
            {"decision_score": float(score), "is_anomaly": int(label)}
            for score, label in zip(decision_scores, is_anomaly)
        ],
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": round(roc_auc, 4)
        },
        "pca_scatter": pca_data
    }

def predict_single_account(data: dict):
    """Loads saved model artifacts and performs real-time inference on a customer payload."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Model files not found. Train the model first.")
        
    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)
    
    input_df = pd.DataFrame([data])
    scaled_df = scaler.transform(input_df)
    
    pred = model.predict(scaled_df)[0]
    score = model.decision_function(scaled_df)[0]
    
    is_anomaly = True if pred == -1 else False
    
    return {
        "is_anomaly": is_anomaly,
        "status": "HIGH RISK ANOMALY" if is_anomaly else "NORMAL",
        "decision_score": round(float(score), 4)
    }