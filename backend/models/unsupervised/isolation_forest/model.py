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

    def fit(self, X_train: np.ndarray):
        """
        Fits the Isolation Forest and the MinMaxScaler based on training anomaly scores.
        """
        print("=" * 60)
        print("STEP 4: TRAIN ISOLATION FOREST")
        print("=" * 60)

        start = time.time()
        self.model.fit(X_train)
        print(f"Training complete in {time.time()-start:.1f}s")

        # Fit MinMaxScaler on TRAIN scores
        raw_scores_train = -self.model.decision_function(X_train)
        self.mm_scaler.fit(raw_scores_train.reshape(-1, 1))
        
        # Calculate thresholds on TRAIN scores
        risk_scores_train = self.mm_scaler.transform(raw_scores_train.reshape(-1, 1)).flatten()
        self.approve_threshold = np.percentile(risk_scores_train, 85)
        self.flag_threshold    = np.percentile(risk_scores_train, 95)
        
        print(f"MinMax Scaler fitted and thresholds established.\n")

    def predict(self, X: np.ndarray):
        """
        Predicts binary labels and generates 0-100 risk scores for given data.
        """
        # Raw predictions: -1 = anomaly (fraud), 1 = normal (legitimate)
        raw_preds = self.model.predict(X)
        iso_predictions = np.where(raw_preds == -1, 1, 0)

        # Anomaly score: higher score = more suspicious
        anomaly_scores = self.model.decision_function(X)
        inverted_scores = -anomaly_scores

        # 0-100 risk score using fitted scaler
        iso_risk_score = self.mm_scaler.transform(inverted_scores.reshape(-1, 1)).flatten()

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
