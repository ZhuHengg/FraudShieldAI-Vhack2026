import numpy as np
import pandas as pd
import joblib
import json
import os
import time
import logging
from datetime import datetime, timezone

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

        # Pre-build feature name lists (avoids re-creating lists every call)
        self._iso_binary_features = [
            'transfer_type_encoded', 'is_weekend', 'sender_account_fully_drained',
            'is_new_device', 'is_proxy_ip', 'country_mismatch_suspicious',
            'established_user_new_recipient'
        ]
        self._iso_continuous_features = [
            'amount_vs_avg_ratio', 'transaction_hour', 'session_duration_seconds',
            'failed_login_attempts', 'ip_risk_score', 'account_age_days',
            'tx_count_24h', 'recipient_risk_profile_score'
        ]
        self._lgb_continuous_features = [
            'log_amount', 'log_avg_30d', 'amount_vs_avg_ratio', 'transaction_hour',
            'session_duration_seconds', 'failed_login_attempts', 'ip_risk_score',
            'account_age_days', 'tx_count_24h', 'recipient_risk_profile_score'
        ]
        # Pre-build set for fast lookup
        self._lgb_continuous_set = set(self._lgb_continuous_features)

        logger.info("EnsembleEngine initialized successfully")

    # ─── Helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _safe_float(val, default=0.0):
        """Convert a value to float safely."""
        try:
            v = float(val)
            return default if v != v else v   # NaN check without numpy
        except (ValueError, TypeError):
            return default

    # ─── Preprocessing (dict-based, NO Pandas) ─────────────────────────
    def preprocess(self, txn_dict: dict) -> dict:
        """
        Applies privacy masking, cleans data, and derives basic features.
        Returns a plain dict — no DataFrame created.
        """
        # 1. Privacy Masking (returns a shallow copy)
        d = self.privacy.prepare_for_inference(txn_dict)

        # 2. Ensure numeric for all required feature fields
        _sf = self._safe_float
        for col in (
            'amount_vs_avg_ratio', 'avg_transaction_amount_30d', 'session_duration_seconds',
            'failed_login_attempts', 'tx_count_24h', 'sender_account_fully_drained',
            'is_new_recipient', 'established_user_new_recipient', 'account_age_days',
            'recipient_risk_profile_score', 'is_new_device', 'is_proxy_ip', 'ip_risk_score',
            'country_mismatch', 'country_mismatch_suspicious', 'transfer_type_encoded',
        ):
            d[col] = _sf(d.get(col), 0.0)

        # 3. Derived features
        amount = _sf(d.get('amount'), 0.0)
        d['amount'] = amount
        d['log_amount'] = float(np.log1p(amount))

        avg_amt = d['avg_transaction_amount_30d']
        d['log_avg_30d'] = float(np.log1p(avg_amt)) if avg_amt > 0 else d['log_amount']

        # transfer_type_encoded: CASH_OUT=0, else=1
        tx_type = str(d.get('transaction_type', '')).upper()
        d['transfer_type_encoded'] = 0.0 if tx_type == 'CASH_OUT' else 1.0

        # Timestamp → transaction_hour, is_weekend
        ts_str = str(d.get('timestamp', ''))
        try:
            ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            d['transaction_hour'] = float(ts.hour)
            if d.get('is_weekend', 0.0) == 0.0:
                d['is_weekend'] = 1.0 if ts.weekday() in (5, 6) else 0.0
        except (ValueError, AttributeError):
            d.setdefault('transaction_hour', 0.0)
            d.setdefault('is_weekend', 0.0)

        # Amount-vs-avg ratio with stateful EMA cache
        TYPE_DEFAULTS = {'TRANSFER': 300.0, 'CASH_OUT': 800.0, 'PAYMENT': 150.0}
        hashed_sender = d.get('sender_id', d.get('name_sender', 'unknown_sender'))

        if hashed_sender in self.user_avg_cache:
            past_avg = self.user_avg_cache[hashed_sender]
        elif avg_amt > 1.0:
            past_avg = avg_amt
        else:
            past_avg = TYPE_DEFAULTS.get(tx_type, 300.0)

        raw_ratio = amount / max(past_avg, 1.0)
        d['amount_vs_avg_ratio'] = min(raw_ratio, 15.0)
        d['log_avg_30d'] = float(np.log1p(past_avg))

        ALPHA = 0.3
        self.user_avg_cache[hashed_sender] = ALPHA * amount + (1 - ALPHA) * past_avg

        if d.get('country_mismatch_suspicious', 0.0) == 0.0:
            d['country_mismatch_suspicious'] = d.get('country_mismatch', 0.0)

        return d

    # ─── Feature extraction (numpy-only) ───────────────────────────────
    def get_iso_features(self, d: dict) -> np.ndarray:
        """Build Isolation Forest feature array from dict — no DataFrame."""
        cont_vals = np.array([[d.get(f, 0.0) for f in self._iso_continuous_features]])
        cont_scaled = self.iso_scaler.transform(cont_vals)
        bin_vals = np.array([[d.get(f, 0.0) for f in self._iso_binary_features]])
        return np.hstack([bin_vals, cont_scaled])

    def get_lgb_features(self, d: dict) -> np.ndarray:
        """Build LightGBM feature array from dict — no DataFrame."""
        # Scale continuous features
        cont_vals = np.array([[d.get(f, 0.0) for f in self._lgb_continuous_features]])
        cont_scaled = self.lgb_scaler.transform(cont_vals)
        scaled_map = dict(zip(self._lgb_continuous_features, cont_scaled[0]))

        # Build full vector in self.lgb_features order
        row = []
        for feat in self.lgb_features:
            if feat in self._lgb_continuous_set:
                row.append(scaled_map[feat])
            else:
                row.append(self._safe_float(d.get(feat), 0.0))
        return np.array([row])

    # ─── Legacy DataFrame builders (used only by /explain SHAP) ────────
    def get_lgb_features_df(self, d: dict) -> pd.DataFrame:
        """
        Returns LightGBM features as a *DataFrame* (with column names)
        so that the SHAP explainer can map contributions back to feature names.
        Only used by the /explain endpoint — NOT on the hot scoring path.
        """
        arr = self.get_lgb_features(d)
        return pd.DataFrame(arr, columns=list(self.lgb_features))

    # ─── Model scoring ─────────────────────────────────────────────────
    def score_iso(self, X_iso: np.ndarray) -> float:
        raw_scores = -self.iso_model.decision_function(X_iso)
        risk_scores = self.iso_mms.transform(raw_scores.reshape(-1, 1)).flatten()
        return float(np.clip(risk_scores[0], 0.0, 100.0))

    def score_lgb(self, X_lgb: np.ndarray) -> float:
        return float(self.lgb_model.predict_proba(X_lgb)[:, 1][0] * 100)

    # ─── Behavioral scoring (inlined, no Pandas) ──────────────────────
    def _score_behavioral(self, d: dict):
        """
        Inline behavioral scoring using plain dict values.
        Returns (total_score_0_to_100, reasons_list, (drain_raw, dev_raw, ctx_raw, vel_raw))
        """
        # drain_to_unknown (weight 0.35)
        drain_raw = d.get('sender_account_fully_drained', 0.0) * d.get('is_new_recipient', 0.0)

        # high_amount_deviation (weight 0.25)
        ratio = d.get('amount_vs_avg_ratio', 0.0)
        dev_raw = max(0.0, min(1.0, (ratio - 1.5) / 3.5))

        # risky_context (weight 0.20)
        ctx_raw = (d.get('ip_risk_score', 0.0) * 0.5
                   + d.get('country_mismatch', 0.0) * 0.3
                   + d.get('is_new_device', 0.0) * 0.2)
        ctx_raw = max(0.0, min(1.0, ctx_raw))

        # rapid_session (weight 0.20)
        vel_raw = 0.0
        if d.get('tx_count_24h', 0) > 5:
            vel_raw += 0.5
        if d.get('session_duration_seconds', 999) < 60:
            vel_raw += 0.5
        vel_raw = min(vel_raw, 1.0)

        # Weighted totals
        r_drain = drain_raw * 0.35
        r_dev   = dev_raw   * 0.25
        r_ctx   = ctx_raw   * 0.20
        r_vel   = vel_raw   * 0.20
        total   = r_drain + r_dev + r_ctx + r_vel

        # Human-readable reasons
        reasons = []
        if r_drain > 0:
            reasons.append("Account fully drained to new recipient")
        if dev_raw > 0.1:
            reasons.append("Amount significantly exceeds user's average")
        if ctx_raw > 0.1:
            reasons.append("High context risk (Foreign/New Device/IP)")
        if r_vel > 0:
            reasons.append("High urgency/velocity signals detected")
        if not reasons:
            reasons.append("Normal behavior pattern")

        return total * 100, reasons, (drain_raw, dev_raw, ctx_raw, vel_raw)

    # ─── Legacy wrapper kept for /explain endpoint ─────────────────────
    def score_beh(self, d: dict):
        """Wrapper that returns (score, reasons) — same interface as before."""
        score, reasons, _ = self._score_behavioral(d)
        return score, reasons

    # ─── Main prediction (hot path — zero Pandas) ─────────────────────
    def predict(self, txn: TransactionRequest) -> RiskResponse:
        from api.schemas import RuleBreakdown, FeatureSnapshot
        
        # 1. Dict-based preprocessing (no DataFrame)
        d = self.preprocess(txn.model_dump())

        # 2. Model Scoring (numpy arrays only)
        X_iso = self.get_iso_features(d)
        iso_score = self.score_iso(X_iso)
        
        X_lgb = self.get_lgb_features(d)
        lgb_score = self.score_lgb(X_lgb)
        
        beh_score, beh_reasons, (drain_raw, dev_raw, ctx_raw, vel_raw) = self._score_behavioral(d)
        reasons = [r for r in beh_reasons if "Normal behavior" not in r]
        if not reasons:
            reasons.append("Normal behavior pattern")
            
        # 3. Fuse Scores
        final_score = (lgb_score * self.w_lgb) + (iso_score * self.w_iso) + (beh_score * self.w_beh)
        final_score = float(np.clip(final_score, 0.0, 100.0))
        
        # 4. Threshold Logic
        if final_score < self.approve_threshold:
            decision = "LOW"
        elif final_score < self.flag_threshold:
            decision = "MEDIUM"
        else:
            decision = "HIGH"

        # 5. Rule Breakdown (reuses raw scores — no second profiler call)
        rule_breakdown = RuleBreakdown(
            drain_score=float(np.round(drain_raw * self.profiler.rules['drain_to_unknown'], 4)),
            deviation_score=float(np.round(dev_raw * self.profiler.rules['high_amount_deviation'], 4)),
            context_score=float(np.round(ctx_raw * self.profiler.rules['risky_context'], 4)),
            velocity_score=float(np.round(vel_raw * self.profiler.rules['rapid_session'], 4)),
        )

        # 6. Feature Snapshot
        feature_snapshot = FeatureSnapshot(
            amount_vs_avg_ratio=float(d.get('amount_vs_avg_ratio', 0.0)),
            ip_risk_score=float(d.get('ip_risk_score', 0.0)),
            tx_count_24h=int(d.get('tx_count_24h', 0)),
            session_duration_seconds=float(d.get('session_duration_seconds', 0.0)),
            is_new_device=int(d.get('is_new_device', 0)),
            country_mismatch=int(d.get('country_mismatch', 0)),
            sender_fully_drained=int(d.get('sender_account_fully_drained', 0)),
            is_new_recipient=int(d.get('is_new_recipient', 0)),
            account_age_days=float(d.get('account_age_days', 0.0)),
            is_proxy_ip=int(d.get('is_proxy_ip', 0)),
        )
            
        # 7. Return enriched RiskResponse
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
