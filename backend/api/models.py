from sqlalchemy import Column, Integer, String, Float
from api.database import Base

class TransactionLog(Base):
    __tablename__ = "transaction_logs"

    # 1. Identifiers
    transaction_id = Column(String, primary_key=True, index=True)
    user_hash = Column(String, index=True)      # Mapped from name_sender
    recipient_hash = Column(String)             # Mapped from name_recipient
    
    # 2. Transaction Details
    transfer_type = Column(String)
    amount = Column(Float)
    avg_transaction_amount_30d = Column(Float)
    amount_vs_avg_ratio = Column(Float)
    transaction_hour = Column(Integer)
    
    # 3. Behavioral & Device Flags (0 or 1)
    is_weekend = Column(Integer)
    sender_account_fully_drained = Column(Integer)
    is_new_device = Column(Integer)
    session_duration_seconds = Column(Integer)
    failed_login_attempts = Column(Integer)
    is_proxy_ip = Column(Integer)
    ip_risk_score = Column(Float)
    country_mismatch = Column(Integer)
    
    # 4. History & Velocity
    account_age_days = Column(Integer)
    tx_count_24h = Column(Integer)
    is_new_recipient = Column(Integer)
    established_user_new_recipient = Column(Integer)
    recipient_risk_profile_score = Column(Float)
    
    # 5. Ground Truth & App Action
    is_fraud = Column(Integer)
    action_taken = Column(String)   # "APPROVE" or "BLOCK" for the app UI
    ml_risk_score = Column(Float)   # Risk percentage for the app UI

    # 6. Balance Fields
    sender_balance_before = Column(Float)
    sender_balance_after = Column(Float)
    receiver_balance_before = Column(Float)
    receiver_balance_after = Column(Float)

    # 7. UI Context Fields
    currency = Column(String)
    country = Column(String)
    device_type = Column(String)

    # 8. Analyst Feedback — for closed-loop retraining
    analyst_label = Column(String, nullable=True)      # "FRAUD" or "LEGIT" — human override
    analyst_notes = Column(String, nullable=True)       # Free-text analyst notes
    labeled_at = Column(String, nullable=True)          # ISO timestamp of when labeled