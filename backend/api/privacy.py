import hashlib
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

class PrivacyProtector:
    """
    Handles data masking and differential privacy for the FraudShield API.
    Ensures compliance with PDPA/GDPR for ASEAN user base.
    """
    def __init__(self, salt: str = None):
        self.salt = (salt or os.environ.get("FRAUDSHIELD_SALT", "vhack_2026_super_secret")).encode()
        
    def hash_pii(self, value: str) -> str:
        """One-way cryptographic hash for PII like Sender/Recipient IDs"""
        if not value: return ""
        hasher = hashlib.sha256()
        hasher.update(value.encode())
        hasher.update(self.salt)
        # Return full 64-char hex digest to preserve collision resistance
        return hasher.hexdigest()
        
    def add_dp_noise(self, score: float, epsilon: float = 5.0, sensitivity: float = 1.0) -> float:
        """
        Add Laplace noise for differential privacy on exported aggregate scores.
        Note: Epsilon=5.0 is relatively high (less privacy, higher utility).
        For strict privacy, lower epsilon (e.g., 1.0 or 0.1).
        """
        # Scale of Laplace noise is sensitivity / epsilon
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale)
        noisy_score = score + noise
        return np.clip(noisy_score, 0.0, 1.0)
        
    def prepare_for_inference(self, txn_dict: dict) -> dict:
        """Mask requested fields before sending to the model engine"""
        safe_txn = txn_dict.copy()
        
        # Hash identifiable information. Use sender_id and receiver_id mapping.
        if 'sender_id' in safe_txn:
            safe_txn['sender_id'] = self.hash_pii(safe_txn['sender_id'])
        elif 'name_sender' in safe_txn:
            safe_txn['name_sender'] = self.hash_pii(safe_txn['name_sender'])
            
        if 'receiver_id' in safe_txn:
            safe_txn['receiver_id'] = self.hash_pii(safe_txn['receiver_id'])
        elif 'name_recipient' in safe_txn:
            safe_txn['name_recipient'] = self.hash_pii(safe_txn['name_recipient'])
            
        return safe_txn
