from pydantic import BaseModel, Field
from typing import Optional, List

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

class RuleBreakdown(BaseModel):
    """Per-rule weighted scores from the BehavioralProfiler."""
    drain_score: float = 0.0
    deviation_score: float = 0.0
    context_score: float = 0.0
    velocity_score: float = 0.0

class FeatureSnapshot(BaseModel):
    """Key feature values used during inference — for frontend dashboards."""
    amount_vs_avg_ratio: float = 1.0
    ip_risk_score: float = 0.0
    tx_count_24h: int = 0
    session_duration_seconds: float = 0.0
    is_new_device: int = 0
    country_mismatch: int = 0
    sender_fully_drained: int = 0
    is_new_recipient: int = 0
    account_age_days: float = 0.0
    is_proxy_ip: int = 0

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
    
    # Extended fields for dashboard analytics
    rule_breakdown: Optional[RuleBreakdown] = None
    feature_snapshot: Optional[FeatureSnapshot] = None

    # V4 Architecture Fields
    mule_flag: bool = Field(False, description="True if recipient mule network detected (Step 4)")
    engine_mode: str = Field("full", description="Engine mode: full/degraded_iso/degraded_lgb/behavioral_only/static_rules (Step 5)")
    active_models: list[str] = Field(default_factory=lambda: ["lgb", "iso", "beh"], description="Models that contributed to this score")
    calibration_source: str = Field("empirical_pr_curve", description="Source of threshold calibration (Step 6)")

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
                },
                "mule_flag": False,
                "engine_mode": "full",
                "active_models": ["lgb", "iso", "beh"],
                "calibration_source": "empirical_pr_curve"
            }
        }

class TransactionLogCreate(BaseModel):
    transaction_id: str
    user_hash: str
    recipient_hash: str
    transfer_type: str
    amount: float
    avg_transaction_amount_30d: float
    amount_vs_avg_ratio: float
    transaction_hour: int
    is_weekend: int
    sender_account_fully_drained: int
    is_new_device: int
    session_duration_seconds: int
    failed_login_attempts: int
    is_proxy_ip: int
    ip_risk_score: float
    country_mismatch: int
    account_age_days: int
    tx_count_24h: int
    is_new_recipient: int
    established_user_new_recipient: int
    recipient_risk_profile_score: float
    is_fraud: int
    action_taken: str
    ml_risk_score: float
    sender_balance_before: float
    sender_balance_after: float
    receiver_balance_before: float
    receiver_balance_after: float
    currency: str
    country: str
    device_type: str

class TransactionLogResponse(TransactionLogCreate):
    analyst_label: Optional[str] = None
    analyst_notes: Optional[str] = None
    labeled_at: Optional[str] = None

# ── Closed-Loop Retraining Schemas ────────────────────────────────────────────

class AnalystFeedback(BaseModel):
    """Submit human feedback for a transaction."""
    transaction_id: str
    analyst_label: str = Field(..., description="'FRAUD' or 'LEGIT'")
    analyst_notes: Optional[str] = None

class FeedbackStatsResponse(BaseModel):
    """Summary of labeling progress for retraining readiness."""
    total_transactions: int
    labeled_count: int
    unlabeled_count: int
    fraud_labels: int
    legit_labels: int
    ready_to_retrain: bool
    min_samples_needed: int = 50

class RetrainRequest(BaseModel):
    """Parameters for the retraining pipeline."""
    min_labeled_samples: int = Field(50, ge=10, description="Minimum labeled samples required.")

class RetrainResponse(BaseModel):
    """Results of a retraining run."""
    status: str
    samples_used: int
    old_weights: dict
    new_weights: dict
    old_thresholds: dict
    new_thresholds: dict
    old_f1: float
    new_f1: float
    improvement_pct: float
    message: str

class InvestigateRequest(BaseModel):
    """Free-form investigation query powered by LLM."""
    query: str = Field(..., description="Analyst's question about the transaction")
    transaction_id: Optional[str] = Field(None, description="Transaction ID to investigate (fetches full context)")
    context: Optional[dict] = Field(None, description="Optional pre-built context dict (overrides DB lookup)")

class InvestigateResponse(BaseModel):
    """Response from the LLM investigation assistant."""
    model_config = {"protected_namespaces": ()}

    response: str = Field(..., description="Natural-language analysis from the LLM")
    model_used: str = Field("gemini-2.0-flash", description="Which model generated the response")
    tokens_used: Optional[int] = None
    status: str = Field("success", description="success | error | unavailable")

# ── Quarantine Schemas (V4 Step 8) ────────────────────────────────────────────

class QuarantineStatsResponse(BaseModel):
    """Summary of quarantine status across all labeled transactions."""
    total_quarantined: int = 0
    validated: int = 0
    rejected: int = 0
    pending: int = 0
    direct_labels: int = 0  # Labels that bypassed quarantine

class QuarantineValidationResponse(BaseModel):
    """Result of running quarantine validation."""
    total_quarantined: int
    validated: int
    rejected: int
    results: list = Field(default_factory=list)

class CalibrationResponse(BaseModel):
    """Current threshold calibration metadata (V4 Step 6)."""
    approve_threshold: float
    flag_threshold: float
    calibration_source: str
    weights: dict
    validation_metrics: Optional[dict] = None
    retrain_metadata: Optional[dict] = None
