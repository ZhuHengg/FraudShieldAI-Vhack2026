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
        self.user_avg_cache = {}  # In-memory cache for exponential moving average
        
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

        # Initialize SHAP explainer for LightGBM
        try:
            import shap
            self.explainer = shap.TreeExplainer(self.lgb_model)
        except Exception as e:
            logger.warning(f"SHAP explainer could not be initialized: {e}")
            self.explainer = None

        logger.info("EnsembleEngine initialized successfully")

    def preprocess(self, txn_dict: dict) -> pd.DataFrame:
        """
        Applies privacy masking, cleans data, and derives basic features.
        Shared between predict() and explain_transaction().
        """
        # 1. Privacy Masking
        safe_txn = self.privacy.prepare_for_inference(txn_dict)
        df = pd.DataFrame([safe_txn])

        # 2. Fill Missing Columns and Numeric Conversion
        required_feature_cols = [
            'amount_vs_avg_ratio', 'avg_transaction_amount_30d', 'session_duration_seconds',
            'failed_login_attempts', 'tx_count_24h', 'sender_account_fully_drained',
            'is_new_recipient', 'established_user_new_recipient', 'account_age_days',
            'recipient_risk_profile_score', 'is_new_device', 'is_proxy_ip', 'ip_risk_score',
            'country_mismatch', 'country_mismatch_suspicious', 'transfer_type_encoded'
        ]
        for col in required_feature_cols:
            if col not in df.columns or df[col].isna().any():
                df[col] = df[col].fillna(0.0) if col in df.columns else 0.0
            
            # Ensure numeric
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        # 3. Derived Features
        df['log_amount'] = np.log1p(df['amount'].astype(float))
        
        # Avoid boolean attribute errors by checking values directly (input is single row)
        avg_amt = df['avg_transaction_amount_30d'].iloc[0]
        if pd.notna(avg_amt) and avg_amt > 0:
            df['log_avg_30d'] = np.log1p(float(avg_amt))
        else:
            df['log_avg_30d'] = df['log_amount']

        if df['transfer_type_encoded'].iloc[0] == 0.0:
            tx_type = str(df['transaction_type'].iloc[0]).upper()
            df['transfer_type_encoded'] = 1 if tx_type == 'CASH_OUT' else 0

        ts = pd.to_datetime(df['timestamp'], utc=True)
        df['transaction_hour'] = ts.dt.hour.iloc[0]
        
        if 'is_weekend' not in df.columns or df['is_weekend'].iloc[0] == 0.0:
            df['is_weekend'] = 1 if ts.dt.dayofweek.iloc[0] in [5, 6] else 0

        # Fix: Ensure realistic cold-start baselines and stateful tracking
        TYPE_DEFAULTS = {
            'TRANSFER':  300.0,
            'CASH_OUT':  800.0,
            'PAYMENT':   150.0,
        }
        tx_type = str(df['transaction_type'].iloc[0]).upper()
        
        # Get hashed sender ID (already masked by PrivacyProtector if applicable)
        hashed_sender = df['sender_id'].iloc[0] if 'sender_id' in df.columns else "unknown_sender"
        
        # Determine baseline past average
        if hashed_sender in self.user_avg_cache:
            past_avg = self.user_avg_cache[hashed_sender]
        elif pd.notna(avg_amt) and avg_amt > 1.0:
            past_avg = float(avg_amt)
        else:
            past_avg = TYPE_DEFAULTS.get(tx_type, 300.0)
            
        raw_ratio = df['amount'].iloc[0] / max(past_avg, 1.0)
        
        # Fix: Cap the ratio at 15x to prevent model distortion from extreme scaling
        df['amount_vs_avg_ratio'] = min(raw_ratio, 15.0)
        
        # Fix: Update log_avg_30d to match the true cached average
        df['log_avg_30d'] = np.log1p(past_avg)
        
        # Fast convergence update
        ALPHA = 0.3
        self.user_avg_cache[hashed_sender] = ALPHA * df['amount'].iloc[0] + (1 - ALPHA) * past_avg

        if 'country_mismatch_suspicious' not in df.columns or df['country_mismatch_suspicious'].iloc[0] == 0.0:
            df['country_mismatch_suspicious'] = df.get('country_mismatch', pd.Series([0])).iloc[0]

        return df

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
                
        df_iso[continuous_features] = self.iso_scaler.transform(df_iso[continuous_features].values)
        return df_iso[all_iso_features]

    def get_lgb_features(self, df: pd.DataFrame) -> pd.DataFrame:
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
                
        # To silence sklearn feature name warnings, pass a numpy array to the scaler
        # The scaler might expect specific feature names, but we only have `continuous_features`.
        # To be safe and just silence the estimator warning later, we transform and assign back.
        df_lgb[continuous_features] = self.lgb_scaler.transform(df_lgb[continuous_features].values)
        
        # Column-alignment defensive step
        for col in self.lgb_features:
            if col not in df_lgb.columns:
                df_lgb[col] = 0.0
                
        # Ensure all columns are numeric
        return df_lgb[self.lgb_features].astype(float)

    def score_iso(self, X_iso: pd.DataFrame) -> float:
        raw_scores = -self.iso_model.decision_function(X_iso.values)
        risk_scores = self.iso_mms.transform(raw_scores.reshape(-1, 1)).flatten()
        return risk_scores[0]

    def score_lgb(self, X_lgb: pd.DataFrame) -> float:
        return self.lgb_model.predict_proba(X_lgb)[:, 1][0] * 100

    def score_beh(self, df: pd.DataFrame) -> tuple:
        beh_scores_raw, reasons = self.profiler.predict(df)
        return float(beh_scores_raw[0] * 100), reasons[0].split(' | ')

    def predict(self, txn: TransactionRequest) -> RiskResponse:
        from api.schemas import RuleBreakdown, FeatureSnapshot
        
        # 1. Apply Shared Preprocessing
        df = self.preprocess(txn.model_dump())

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

        # 6. Extract per-rule scores for breakdown
        drain_raw = self.profiler._score_drain_unknown(df).iloc[0]
        dev_raw = self.profiler._score_high_deviation(df).iloc[0]
        ctx_raw = self.profiler._score_risky_context(df).iloc[0]
        vel_raw = self.profiler._score_rapid_session(df).iloc[0]
        
        rule_breakdown = RuleBreakdown(
            drain_score=float(np.round(drain_raw * self.profiler.rules['drain_to_unknown'], 4)),
            deviation_score=float(np.round(dev_raw * self.profiler.rules['high_amount_deviation'], 4)),
            context_score=float(np.round(ctx_raw * self.profiler.rules['risky_context'], 4)),
            velocity_score=float(np.round(vel_raw * self.profiler.rules['rapid_session'], 4)),
        )

        # 7. Extract key feature values for dashboard
        feature_snapshot = FeatureSnapshot(
            amount_vs_avg_ratio=float(df['amount_vs_avg_ratio'].iloc[0]),
            ip_risk_score=float(df['ip_risk_score'].iloc[0]) if 'ip_risk_score' in df.columns else 0.0,
            tx_count_24h=int(df['tx_count_24h'].iloc[0]) if 'tx_count_24h' in df.columns else 0,
            session_duration_seconds=float(df['session_duration_seconds'].iloc[0]) if 'session_duration_seconds' in df.columns else 0.0,
            is_new_device=int(df['is_new_device'].iloc[0]) if 'is_new_device' in df.columns else 0,
            country_mismatch=int(df['country_mismatch'].iloc[0]) if 'country_mismatch' in df.columns else 0,
            sender_fully_drained=int(df['sender_account_fully_drained'].iloc[0]) if 'sender_account_fully_drained' in df.columns else 0,
            is_new_recipient=int(df['is_new_recipient'].iloc[0]) if 'is_new_recipient' in df.columns else 0,
            account_age_days=float(df['account_age_days'].iloc[0]) if 'account_age_days' in df.columns else 0.0,
            is_proxy_ip=int(df['is_proxy_ip'].iloc[0]) if 'is_proxy_ip' in df.columns else 0,
        )
            
        # 8. Return enriched RiskResponse
        return RiskResponse(
            transaction_id=txn.transaction_id,
            risk_score=float(np.round(final_score, 2)),
            risk_level=decision,
            supervised_score=float(np.round(lgb_score / 100.0, 4)),
            unsupervised_score=float(np.round(iso_score / 100.0, 4)),
            behavioral_score=float(np.round(beh_score / 100.0, 4)),
            reasons=reasons,
            privacy=PrivacyInfo(pii_hashed=True, hash_algorithm="SHA-256", dp_applied=False),
            rule_breakdown=rule_breakdown,
            feature_snapshot=feature_snapshot,
        )
