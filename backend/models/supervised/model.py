import lightgbm as lgb
import numpy as np
import pandas as pd
import joblib

class LightGBMModel:
    def __init__(self):
        self.model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=63,
            scale_pos_weight=1.0,
            # scale_pos_weight=1.0 because SMOTE already handles
            # class imbalance — avoid double correction
            objective='binary',
            random_state=42,
            n_jobs=-1
        )
        # Thresholds start as None
        # Must be set via set_threshold() after tune_threshold()
        # NEVER hardcode these values
        self.approve_threshold = None
        self.flag_threshold    = None

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Train with early stopping on validation AUC-PR.
        X_train is SMOTE-resampled.
        X_val is original unaugmented validation data.
        Early stopping prevents overfitting on synthetic SMOTE samples.
        """
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='aucpr',
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=50,
                    verbose=True
                )
            ]
        )
        print(f"Training complete.")
        print(f"Best iteration: {self.model.best_iteration_}")

    def predict_proba(self, X):
        """Returns fraud probability 0.0-1.0"""
        return self.model.predict_proba(X)[:, 1]

    def predict_scaled(self, X):
        """
        Returns 0-100 scaled risk score.
        Matches Isolation Forest output format exactly.
        Used for threshold tuning and ensemble fusion.
        Multiply probability by 100 — no other scaling applied.
        """
        return self.predict_proba(X) * 100

    def set_threshold(self, optimal_threshold: float):
        """
        Sets approve and flag thresholds.
        optimal_threshold MUST come from tune_threshold()
        which derives it empirically from PR curve on validation set.

        This method must NEVER be called with a hardcoded value.
        Correct usage:
            best_threshold, _, _, _ = tune_threshold(y_val, scores)
            lgb_model.set_threshold(best_threshold)

        Wrong usage:
            lgb_model.set_threshold(0.35)  ← NEVER DO THIS
            lgb_model.set_threshold(0.70)  ← NEVER DO THIS
        """
        self.approve_threshold = optimal_threshold
        self.flag_threshold    = optimal_threshold + (
            (100 - optimal_threshold) * 0.5
        )
        print(f"Empirically tuned thresholds set:")
        print(f"  Source:  PR curve optimisation on validation set")
        print(f"  Approve → below {self.approve_threshold:.2f}")
        print(f"  Flag    → {self.approve_threshold:.2f} "
              f"to {self.flag_threshold:.2f}")
        print(f"  Block   → above {self.flag_threshold:.2f}")

    def predict_with_tier(self, X):
        """
        Returns binary predictions and 0-100 risk scores.
        Raises RuntimeError if set_threshold() not called first.
        """
        if self.approve_threshold is None:
            raise RuntimeError(
                "Thresholds not set. Call set_threshold() "
                "after tune_threshold() before predict_with_tier()"
            )

        risk_scores = self.predict_scaled(X)
        predictions = (
            risk_scores >= self.approve_threshold
        ).astype(int)
        
        return predictions, risk_scores

    def assign_tier(self, score: float) -> str:
        """Maps 0-100 risk score to APPROVE/FLAG/BLOCK tier."""
        if self.approve_threshold is None:
            raise RuntimeError(
                "Thresholds not set. Call set_threshold() first."
            )
        if score < self.approve_threshold:
            return 'Approve'
        elif score < self.flag_threshold:
            return 'Flag'
        else:
            return 'Block'

    def get_feature_importance(self, feature_names: list):
        """Returns feature importances as sorted pandas Series."""
        return pd.Series(
            self.model.feature_importances_,
            index=feature_names
        ).sort_values(ascending=False)

    def save(self, model_path: str, features_path: str):
        joblib.dump(self.model, model_path)
        print(f"Model saved:    {model_path}")

    def load(self, model_path: str):
        self.model = joblib.load(model_path)
        print(f"Model loaded:   {model_path}")
