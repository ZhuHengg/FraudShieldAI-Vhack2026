import numpy as np
import pandas as pd
import joblib
import json
import os
import time
import logging

from api.schemas import TransactionRequest, RiskResponse, PrivacyInfo
from api.behavioural import BehavioralProfiler
from api.privacy import PrivacyProtector

logger = logging.getLogger(__name__)

class EnsembleEngine:
    def __init__(self,
                 iso_model_dir: str = 'models/unsupervised/isolation_forest/outputs/model',
                 lgb_model_dir: str = 'models/supervised/outputs/model',
                 ensemble_dir: str = 'models/ensemble/outputs/model'):
        """
        Loads all trained artifacts, components, and ensemble weights.
        """
        self.profiler = BehavioralProfiler()
        self.privacy = PrivacyProtector()
        
        # Load Isolation Forest artifacts
        self.iso_model  = joblib.load(os.path.join(iso_model_dir, 'isolation_forest_model.pkl'))
        self.iso_scaler = joblib.load(os.path.join(iso_model_dir, 'scaler.pkl'))
        self.iso_mms    = joblib.load(os.path.join(iso_model_dir, 'minmax_scaler.pkl'))

        # Load LightGBM artifacts
        self.lgb_model    = joblib.load(os.path.join(lgb_model_dir, 'lgb_model.pkl'))
        self.lgb_scaler   = joblib.load(os.path.join(lgb_model_dir, 'scaler.pkl'))
        self.lgb_features = joblib.load(os.path.join(lgb_model_dir, 'feature_columns.pkl'))

        # Load Ensemble Fusion config
        with open(os.path.join(ensemble_dir, 'ensemble_config.json')) as f:
            self.ensemble_config = json.load(f)
            
        self.w_lgb = self.ensemble_config['weights']['lgb']
        self.w_iso = self.ensemble_config['weights']['iso']
        self.w_beh = self.ensemble_config['weights']['beh']
        self.approve_threshold = self.ensemble_config['thresholds']['optimal_threshold']
        self.flag_threshold = self.ensemble_config['thresholds']['flag_threshold']

        logger.info("EnsembleEngine initialized successfully")

    def get_iso_features(self, df: pd.DataFrame) -> np.ndarray:
        binary_features = [
            'transfer_type_encoded', 'is_weekend', 'sender_account_fully_drained',
            'is_new_device', 'is_proxy_ip', 'country_mismatch_suspicious',
            'established_user_new_recipient'
        ]
        continuous_features = [
            'amount_vs_avg_ratio', 'transaction_hour', 'session_duration_seconds',
            'failed_login_attempts', 'ip_risk_score', 'account_age_days',
            'tx_count_24h', 'recipient_risk_profile_score'
        ]
        all_iso_features = binary_features + continuous_features
        df_iso = df.copy()
        
        # Add missing columns defensively
        for col in all_iso_features:
            if col not in df_iso.columns:
                df_iso[col] = 0.0
                
        df_iso[continuous_features] = self.iso_scaler.transform(df_iso[continuous_features])
        return df_iso[all_iso_features].values

    def get_lgb_features(self, df: pd.DataFrame) -> np.ndarray:
        continuous_features = [
            'log_amount', 'log_avg_30d', 'amount_vs_avg_ratio', 'transaction_hour',
            'session_duration_seconds', 'failed_login_attempts', 'ip_risk_score',
            'account_age_days', 'tx_count_24h', 'recipient_risk_profile_score'
        ]
        df_lgb = df.copy()
        
        # Add missing columns defensively
        for col in continuous_features:
            if col not in df_lgb.columns:
                df_lgb[col] = 0.0
                
        df_lgb[continuous_features] = self.lgb_scaler.transform(df_lgb[continuous_features])
        
        # Column-alignment defensive step
        for col in self.lgb_features:
            if col not in df_lgb.columns:
                df_lgb[col] = 0.0
                
        return df_lgb[self.lgb_features].values

    def score_iso(self, X_iso: np.ndarray) -> float:
        raw_scores = -self.iso_model.decision_function(X_iso)
        risk_scores = self.iso_mms.transform(raw_scores.reshape(-1, 1)).flatten()
        return risk_scores[0]

    def score_lgb(self, X_lgb: np.ndarray) -> float:
        return self.lgb_model.predict_proba(X_lgb)[:, 1][0] * 100

    def score_beh(self, df: pd.DataFrame) -> tuple:
        beh_scores_raw, reasons = self.profiler.predict(df)
        return float(beh_scores_raw[0] * 100), reasons[0].split(' | ')

    def predict(self, txn: TransactionRequest) -> RiskResponse:
        # 1. Apply Privacy Masking
        safe_txn_dict = self.privacy.prepare_for_inference(txn.model_dump())
        
        # 2. Convert to DataFrame
        df = pd.DataFrame([safe_txn_dict])
        
        # Simple feature inference placeholders (For full inference, would run FeatureEngineer)
        df['log_amount'] = np.log1p(df['amount'])
        if 'transfer_type_encoded' not in df.columns:
            df['transfer_type_encoded'] = (df['transaction_type'] == 'cash_out').astype(int)
        
        # 3. Model Scoring
        X_iso = self.get_iso_features(df)
        iso_score = self.score_iso(X_iso)
        
        X_lgb = self.get_lgb_features(df)
        lgb_score = self.score_lgb(X_lgb)
        
        beh_score, beh_reasons = self.score_beh(df)
        reasons = [r.strip() for r in beh_reasons if "Normal behavior" not in r]
        
        if not reasons:
            reasons.append("Normal behavior pattern")
            
        # 4. Fuse Scores
        final_score = (lgb_score * self.w_lgb) + (iso_score * self.w_iso) + (beh_score * self.w_beh)
        final_score = np.clip(final_score, 0.0, 100.0)
        
        # 5. Threshold Logic
        if final_score < self.approve_threshold:
            decision = "LOW"
        elif final_score < self.flag_threshold:
            decision = "MEDIUM"
        else:
            decision = "HIGH"
            
        # 6. Return standard RiskResponse
        return RiskResponse(
            transaction_id=txn.transaction_id,
            risk_score=round(float(final_score), 2),
            risk_level=decision,
            supervised_score=round(float(lgb_score/100.0), 4),
            unsupervised_score=round(float(iso_score/100.0), 4),
            reasons=reasons,
            privacy=PrivacyInfo(pii_hashed=True, hash_algorithm="SHA-256", dp_applied=False)
        )
