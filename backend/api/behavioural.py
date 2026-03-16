import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BehavioralProfiler:
    """
    Layer 3: Interpretable rule-based risk scoring.
    Assigns penalty points based on domain-specific high-risk patterns.
    """
    def __init__(self):
        # Define rule weights (max 1.0 total for easy normalization)
        self.rules = {
            'drain_to_unknown': 0.35,      # Draining account to new recipient
            'high_amount_deviation': 0.25, # Amount >> normal baseline
            'risky_context': 0.20,         # Risky IP / Device / Foreign
            'rapid_session': 0.20          # High urgency
        }
        
    def _score_drain_unknown(self, df):
        if all(c in df.columns for c in ['sender_account_fully_drained', 'is_new_recipient']):
            return df['sender_account_fully_drained'] * df['is_new_recipient']
        return pd.Series(0.0, index=df.index)
        
    def _score_high_deviation(self, df):
        if 'amount_vs_avg_ratio' in df.columns:
            # Score scales with ratio, capped at 1.0 (e.g., 5x or more is max risk)
            return np.clip((df['amount_vs_avg_ratio'] - 1.5) / 3.5, 0, 1)
        return pd.Series(0.0, index=df.index)
        
    def _score_risky_context(self, df):
        score = pd.Series(0.0, index=df.index)
        
        if 'ip_risk_score' in df.columns:
            score += df['ip_risk_score'] * 0.5
            
        if 'country_mismatch' in df.columns:
            score += df['country_mismatch'] * 0.3
            
        if 'is_new_device' in df.columns:
            score += df['is_new_device'] * 0.2
            
        return np.clip(score, 0, 1)
        
    def _score_rapid_session(self, df):
        score = pd.Series(0.0, index=df.index)
        if 'tx_count_24h' in df.columns:
            score += (df['tx_count_24h'] > 5).astype(int) * 0.5
        if 'session_duration_seconds' in df.columns:
            # E.g., very short sessions
            score += (df['session_duration_seconds'] < 60).astype(int) * 0.5
            
        return np.clip(score, 0, 1)

    def predict(self, df):
        """Returns risk score [0, 1] and a list of triggered risk reasons"""
        scores = pd.DataFrame(index=df.index)
        
        scores['r_drain'] = self._score_drain_unknown(df) * self.rules['drain_to_unknown']
        scores['r_dev'] = self._score_high_deviation(df) * self.rules['high_amount_deviation']
        scores['r_ctx'] = self._score_risky_context(df) * self.rules['risky_context']
        scores['r_vel'] = self._score_rapid_session(df) * self.rules['rapid_session']
        
        # Total score
        total_score = scores.sum(axis=1)
        
        # Generate human-readable reasons for top risks
        reasons = []
        for idx in df.index:
            row_reasons = []
            if scores.loc[idx, 'r_drain'] > 0:
                row_reasons.append("Account fully drained to new recipient")
            if scores.loc[idx, 'r_dev'] > 0.1:
                row_reasons.append("Amount significantly exceeds user's average")
            if scores.loc[idx, 'r_ctx'] > 0.1:
                row_reasons.append("High context risk (Foreign/New Device/IP)")
            if scores.loc[idx, 'r_vel'] > 0:
                row_reasons.append("High urgency/velocity signals detected")
                
            if not row_reasons:
                row_reasons.append("Normal behavior pattern")
                
            reasons.append(" | ".join(row_reasons))
            
        return total_score.values, reasons
