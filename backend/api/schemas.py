from pydantic import BaseModel, Field
from typing import Optional

class TransactionRequest(BaseModel):
    """Schema for a transaction to be scored."""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    amount: float = Field(..., description="Transaction amount")
    sender_id: str = Field(..., description="Sender account/wallet ID")
    receiver_id: str = Field(..., description="Receiver account/wallet ID")
    transaction_type: str = Field(..., description="e.g. 'transfer', 'cash_out', 'payment'")
    timestamp: str = Field(..., description="Transaction timestamp (ISO 8601)")

    # Optional features — extend as needed
    sender_balance_before: Optional[float] = None
    sender_balance_after: Optional[float] = None
    receiver_balance_before: Optional[float] = None
    receiver_balance_after: Optional[float] = None

    # Behavioral / session context
    amount_vs_avg_ratio: Optional[float] = None
    avg_transaction_amount_30d: Optional[float] = None
    session_duration_seconds: Optional[float] = None
    failed_login_attempts: Optional[int] = None
    tx_count_24h: Optional[int] = None
    transaction_hour: Optional[int] = None       # derive from timestamp if None
    is_weekend: Optional[int] = None             # 0 or 1, derive from timestamp if None

    # Account / recipient context
    sender_account_fully_drained: Optional[int] = None   # 0 or 1
    is_new_recipient: Optional[int] = None               # 0 or 1
    established_user_new_recipient: Optional[int] = None # 0 or 1
    account_age_days: Optional[float] = None
    recipient_risk_profile_score: Optional[float] = None

    # Device / IP context
    is_new_device: Optional[int] = None     # 0 or 1
    is_proxy_ip: Optional[int] = None       # 0 or 1
    ip_risk_score: Optional[float] = None
    country_mismatch: Optional[int] = None            # 0 or 1
    country_mismatch_suspicious: Optional[int] = None # 0 or 1
    transfer_type_encoded: Optional[int] = None       # 0 or 1

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "TXN_001",
                "amount": 5000.00,
                "sender_id": "USER_A",
                "receiver_id": "USER_B",
                "transaction_type": "transfer",
                "timestamp": "2026-03-11T16:00:00Z",
                "sender_balance_before": 10000.00,
                "sender_balance_after": 5000.00,
                "receiver_balance_before": 2000.00,
                "receiver_balance_after": 7000.00,
                "amount_vs_avg_ratio": 1.5,
                "avg_transaction_amount_30d": 3333.33,
                "session_duration_seconds": 120.0,
                "failed_login_attempts": 0,
                "tx_count_24h": 2,
                "transaction_hour": 16,
                "is_weekend": 0,
                "sender_account_fully_drained": 0,
                "is_new_recipient": 1,
                "established_user_new_recipient": 0,
                "account_age_days": 180.5,
                "recipient_risk_profile_score": 0.1,
                "is_new_device": 0,
                "is_proxy_ip": 0,
                "ip_risk_score": 0.05,
                "country_mismatch": 0,
                "country_mismatch_suspicious": 0,
                "transfer_type_encoded": 0
            }
        }

class PrivacyInfo(BaseModel):
    pii_hashed: bool = Field(True, description="Whether PII was cryptographically hashed before inference")
    hash_algorithm: str = Field("SHA-256", description="Hash algorithm used")
    dp_applied: bool = Field(False, description="Whether differential privacy noise was applied")

class DashboardStats(BaseModel):
    total_transactions: int
    approved: int
    flagged: int
    blocked: int
    avg_latency_ms: float
    fraud_rate_estimate: float

class TopFeature(BaseModel):
    feature: str
    contribution: float

class EnsembleSHAPResponse(BaseModel):
    transaction_id: str
    base_value: float
    iso_score: float
    lgb_score: float
    beh_score: float
    top_features: list[TopFeature]

class RiskResponse(BaseModel):
    """Response schema for fraud prediction."""
    transaction_id: str
    risk_score: float = Field(..., ge=0, le=100, description="Final risk score (0-100)")
    risk_level: str = Field(..., description="LOW / MEDIUM / HIGH")
    supervised_score: float = Field(..., description="LightGBM fraud probability")
    unsupervised_score: float = Field(..., description="Isolation Forest anomaly score")
    behavioral_score: float = Field(..., description="Rule-based behavioral score")
    
    # Behavioral features
    reasons: list[str] = Field(default_factory=list, description="Reasons for the risk score")
    privacy: PrivacyInfo = Field(..., description="Privacy masking information")

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "TXN_001",
                "risk_score": 73.5,
                "risk_level": "HIGH",
                "supervised_score": 0.82,
                "unsupervised_score": 0.65,
                "reasons": ["High context risk (Foreign/New Device/IP)"],
                "privacy": {
                    "pii_hashed": True,
                    "hash_algorithm": "SHA-256",
                    "dp_applied": False
                }
            }
        }
