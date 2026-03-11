import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

class IsolationForestModel:
    def __init__(self, contamination: float, n_estimators: int = 100, random_state: int = 42):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        self.mm_scaler = MinMaxScaler(feature_range=(0, 100))

    def fit_predict(self, X_scaled: np.ndarray):
        """
        Fits the Isolation Forest, computes anomaly scores, predicts binary labels,
        and generates 0-100 risk scores.
        """
        print("=" * 60)
        print("STEP 4: TRAIN ISOLATION FOREST")
        print("=" * 60)

        self.model.fit(X_scaled)
        print("Model trained.\n")

        # Raw scores and inverted scores (higher = more anomalous)
        raw_scores = self.model.decision_function(X_scaled)
        inverted_scores = -raw_scores

        # Binary Predictions
        raw_preds = self.model.predict(X_scaled)
        iso_predictions = np.where(raw_preds == -1, 1, 0)
        
        print(f"Anomalies detected: {iso_predictions.sum()}")
        print(f"Normal detected   : {(iso_predictions == 0).sum()}\n")

        print("=" * 60)
        print("STEP 5: RISK SCORE TIERING")
        print("=" * 60)

        # 0-100 risk score
        iso_risk_score = self.mm_scaler.fit_transform(inverted_scores.reshape(-1, 1)).flatten()

        return iso_predictions, iso_risk_score

    def save(self, filepath: str):
        """Saves the trained model to disk."""
        joblib.dump(self.model, filepath)
        print(f"Model saved: {filepath}")

def assign_tier(score: float) -> str:
    """Assigns risk tier based on continuous risk score."""
    if score <= 70:
        return "Approve"
    elif score <= 92:
        return "Flag"
    else:
        return "Block"

def generate_tier_summary(df_filtered: pd.DataFrame) -> pd.DataFrame:
    """Generates an aggregation of transactions by risk tier."""
    df_filtered["risk_tier"] = df_filtered["iso_risk_score"].apply(assign_tier)
    
    tier_summary = (
        df_filtered.groupby("risk_tier")
        .agg(
            transaction_count=("iso_prediction", "count"),
            fraud_count=("isFraud", "sum"),
        )
        .reindex(["Approve", "Flag", "Block"])
    )
    return tier_summary
