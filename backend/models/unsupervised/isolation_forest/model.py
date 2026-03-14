import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import joblib
import time

class IsolationForestModel:
    def __init__(self, contamination: float, n_estimators: int = 200,
                 max_samples: int = 512, max_features: float = 0.5,  # CHANGED: 0.8 → 0.5
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
        self.approve_threshold = None   # CHANGED: None until set_threshold() is called
        self.flag_threshold    = None   # CHANGED: None until set_threshold() is called

    def fit(self, X_train: np.ndarray):
        """
        Fits the Isolation Forest on training data only.
        Does NOT fit MinMaxScaler or set thresholds here —
        those are done separately after validation tuning.
        """
        print("=" * 60)
        print("STEP 5: TRAIN ISOLATION FOREST")
        print("=" * 60)

        start = time.time()
        self.model.fit(X_train)
        print(f"Training complete in {time.time()-start:.1f}s\n")

    def fit_score_scaler(self, X_train: np.ndarray):
        """
        Fits MinMaxScaler on TRAIN anomaly scores only.
        Must be called after fit() and before get_risk_scores().
        Kept separate from fit() so thresholds can be tuned
        on validation set before being set via set_threshold().
        """
        # CHANGED: separated from fit() — scaler must be fitted
        # before validation scores are computed for threshold tuning
        raw_scores_train = -self.model.decision_function(X_train)
        self.mm_scaler.fit(raw_scores_train.reshape(-1, 1))
        print("MinMaxScaler fitted on train scores.")

    def get_risk_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Returns normalized 0-100 risk scores for any split.
        fit_score_scaler() must be called before this.
        Used to get validation scores for threshold tuning
        before final predictions are made on test set.
        """
        # CHANGED: new method — needed to get val scores
        # for PR curve threshold tuning in evaluation.py
        raw_scores = -self.model.decision_function(X)
        return self.mm_scaler.transform(
            raw_scores.reshape(-1, 1)
        ).flatten()

    def set_threshold(self, optimal_threshold: float):
        """
        Sets approve and flag thresholds using the optimal threshold
        found from PR curve tuning on validation set.
        approve_threshold = optimal threshold from max F1
        flag_threshold    = midpoint between optimal and 100
        """
        # CHANGED: new method — replaces percentile-based thresholds
        # thresholds now come from validation PR curve not train percentiles
        self.approve_threshold = optimal_threshold
        self.flag_threshold    = optimal_threshold + (
            (100 - optimal_threshold) * 0.5
        )
        print(f"Thresholds set from validation tuning:")
        print(f"  Approve → below {self.approve_threshold:.2f}")
        print(f"  Flag    → {self.approve_threshold:.2f} "
              f"to {self.flag_threshold:.2f}")
        print(f"  Block   → above {self.flag_threshold:.2f}")

    def predict(self, X: np.ndarray):
        """
        Predicts binary labels and generates 0-100 risk scores.
        set_threshold() must be called before predict()
        so approve_threshold and flag_threshold are defined.
        """
        # CHANGED: guard against calling predict before set_threshold
        if self.approve_threshold is None or self.flag_threshold is None:
            raise RuntimeError(
                "Thresholds not set. Call set_threshold() "
                "after tune_threshold() before calling predict()."
            )

        risk_scores = self.get_risk_scores(X)

        # Apply tuned threshold for binary prediction
        # Anything above approve_threshold is flagged as fraud
        iso_predictions = (risk_scores >= self.approve_threshold).astype(int)

        return iso_predictions, risk_scores

    def assign_tier(self, score: float) -> str:
        """Assigns risk tier based on tuned thresholds."""
        if self.approve_threshold is None:
            raise RuntimeError(
                "Thresholds not set. Call set_threshold() first."
            )
        if score < self.approve_threshold:
            return "Approve"
        elif score < self.flag_threshold:
            return "Flag"
        else:
            return "Block"

    def save(self, model_path: str, mms_path: str):
        """Saves the trained model and MinMaxScaler to disk."""
        joblib.dump(self.model, model_path)
        joblib.dump(self.mm_scaler, mms_path)
        print(f"Model saved to:         {model_path}")
        print(f"MinMaxScaler saved to:  {mms_path}")


def generate_tier_summary(df_results: pd.DataFrame) -> pd.DataFrame:
    """Generates an aggregation of transactions by risk tier."""
    tier_summary = df_results.groupby('risk_tier').agg(
        transaction_count = ('is_fraud', 'count'),
        fraud_count       = ('is_fraud', 'sum'),
        fraud_rate        = ('is_fraud', 'mean')
    ).round(4)
    return tier_summary