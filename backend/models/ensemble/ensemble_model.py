"""
Loads all three trained models and runs inference.
Does not retrain anything — models are already trained.
Handles feature preparation for each model separately.
"""
import numpy as np
import pandas as pd
import joblib
import json
import os

class EnsembleModel:
    def __init__(self,
                 iso_model_dir: str,
                 lgb_model_dir: str):
        """
        Loads all trained artifacts from their directories.
        iso_model_dir: path to isolation_forest/outputs/model/
        lgb_model_dir: path to supervised/outputs/model/
        """
        # Load Isolation Forest artifacts
        self.iso_model  = joblib.load(
            os.path.join(iso_model_dir,
                         'isolation_forest_model.pkl')
        )
        self.iso_scaler = joblib.load(
            os.path.join(iso_model_dir, 'scaler.pkl')
        )
        self.iso_mms    = joblib.load(
            os.path.join(iso_model_dir, 'minmax_scaler.pkl')
        )
        with open(os.path.join(iso_model_dir,
                               'threshold_config.json')) as f:
            self.iso_config = json.load(f)

        # Load LightGBM artifacts
        self.lgb_model    = joblib.load(
            os.path.join(lgb_model_dir, 'lgb_model.pkl')
        )
        self.lgb_scaler   = joblib.load(
            os.path.join(lgb_model_dir, 'scaler.pkl')
        )
        self.lgb_features = joblib.load(
            os.path.join(lgb_model_dir, 'feature_columns.pkl')
        )
        with open(os.path.join(lgb_model_dir,
                               'threshold_config.json')) as f:
            self.lgb_config = json.load(f)

        print("All model artifacts loaded successfully")
        print(f"IsoForest threshold: "
              f"{self.iso_config['optimal_threshold']:.2f}")
        print(f"LightGBM threshold:  "
              f"{self.lgb_config['optimal_threshold']:.2f}")

    def get_iso_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepares features for Isolation Forest.
        Binary features not scaled.
        Continuous features scaled with iso_scaler.
        """
        binary_features = [
            'transfer_type_encoded',
            'is_weekend',
            'sender_account_fully_drained',
            'is_new_device',
            'is_proxy_ip',
            'country_mismatch_suspicious',
            'established_user_new_recipient'
        ]
        continuous_features = [
            'amount_vs_avg_ratio',
            'transaction_hour',
            'session_duration_seconds',
            'failed_login_attempts',
            'ip_risk_score',
            'account_age_days',
            'tx_count_24h',
            'recipient_risk_profile_score'
        ]
        all_iso_features = binary_features + continuous_features

        df_iso = df.copy()
        df_iso[continuous_features] = self.iso_scaler.transform(
            df_iso[continuous_features]
        )
        return df_iso[all_iso_features].values

    def get_lgb_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepares features for LightGBM.
        Uses lgb_features list from feature_columns.pkl.
        Continuous features scaled with lgb_scaler.
        """
        continuous_features = [
            'log_amount',
            'log_avg_30d',
            'amount_vs_avg_ratio',
            'transaction_hour',
            'session_duration_seconds',
            'failed_login_attempts',
            'ip_risk_score',
            'account_age_days',
            'tx_count_24h',
            'recipient_risk_profile_score'
        ]
        df_lgb = df.copy()
        df_lgb[continuous_features] = self.lgb_scaler.transform(
            df_lgb[continuous_features]
        )
        
        # Column-alignment defensive step
        for col in self.lgb_features:
            if col not in df_lgb.columns:
                df_lgb[col] = 0.0
                
        return df_lgb[self.lgb_features].values

    def score_iso(self, X_iso: np.ndarray) -> np.ndarray:
        """
        Returns Isolation Forest risk scores on 0-100 scale.
        Uses iso_mms for normalization.
        """
        raw_scores  = -self.iso_model.decision_function(X_iso)
        risk_scores = self.iso_mms.transform(
            raw_scores.reshape(-1, 1)
        ).flatten()
        return risk_scores

    def score_lgb(self, X_lgb: np.ndarray) -> np.ndarray:
        """
        Returns LightGBM risk scores on 0-100 scale.
        Multiply predict_proba by 100.
        """
        return self.lgb_model.predict_proba(X_lgb)[:, 1] * 100

    def score_beh(self, df: pd.DataFrame,
                  profiler) -> tuple:
        """
        Returns behavioral scores on 0-100 scale and reasons.
        Multiplies profiler output (0-1) by 100.
        """
        beh_scores_raw, reasons = profiler.predict(df)
        return np.array(beh_scores_raw) * 100, reasons
