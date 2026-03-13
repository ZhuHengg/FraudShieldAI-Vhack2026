import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import joblib
import time

class IsolationForestModel:
    def __init__(self, contamination: float, n_estimators: int = 200,
                 max_samples: int = 512, max_features: float = 0.8,
                 random_state: int = 42):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,
        )
        self.mm_scaler = MinMaxScaler(feature_range=(0, 100))
        self.approve_threshold = 0
        self.flag_threshold = 0

    def fit_predict(self, X: np.ndarray):
        """
        Fits the Isolation Forest, computes anomaly scores, predicts binary labels,
        and generates 0-100 risk scores.
        """
        print("=" * 60)
        print("STEP 4: TRAIN ISOLATION FOREST")
        print("=" * 60)

        start = time.time()
        self.model.fit(X)
        print(f"Training complete in {time.time()-start:.1f}s\n")

        print("Generating predictions and risk scores...")
        # Raw predictions: -1 = anomaly (fraud), 1 = normal (legitimate)
        raw_preds = self.model.predict(X)
        iso_predictions = np.where(raw_preds == -1, 1, 0)

        # Anomaly score: lower = more anomalous
        # Invert so that higher score = more suspicious
        anomaly_scores = self.model.decision_function(X)
        inverted_scores = -anomaly_scores

        print("=" * 60)
        print("STEP 5: RISK SCORE TIERING")
        print("=" * 60)

        # 0-100 risk score
        iso_risk_score = self.mm_scaler.fit_transform(inverted_scores.reshape(-1, 1)).flatten()

        print(f"Risk score range: {iso_risk_score.min():.2f} to {iso_risk_score.max():.2f}")
        print(f"Risk score mean:  {iso_risk_score.mean():.2f}")

        # Percentile-based thresholds — adapts to actual score distribution
        self.approve_threshold = np.percentile(iso_risk_score, 85)
        self.flag_threshold    = np.percentile(iso_risk_score, 95)

        print(f"\nThresholds:")
        print(f"  Approve -> below {self.approve_threshold:.2f}")
        print(f"  Flag    -> {self.approve_threshold:.2f} to {self.flag_threshold:.2f}")
        print(f"  Block   -> above {self.flag_threshold:.2f}\n")

        return iso_predictions, iso_risk_score

    def assign_tier(self, score: float) -> str:
        """Assigns risk tier based on continuous risk score and established thresholds."""
        if score <= self.approve_threshold:
            return "Approve"
        elif score <= self.flag_threshold:
            return "Flag"
        else:
            return "Block"

    def save(self, model_path: str, mms_path: str):
        """Saves the trained model and min-max scaler to disk."""
        joblib.dump(self.model, model_path)
        joblib.dump(self.mm_scaler, mms_path)
        print(f"Model saved to: {model_path}")
        print(f"MinMax Scaler saved to: {mms_path}")

def generate_tier_summary(df_results: pd.DataFrame) -> pd.DataFrame:
    """Generates an aggregation of transactions by risk tier."""
    tier_summary = df_results.groupby('risk_tier').agg(
        transaction_count = ('is_fraud', 'count'),
        fraud_count       = ('is_fraud', 'sum'),
        fraud_rate        = ('is_fraud', 'mean')
    ).round(4)
    return tier_summary
