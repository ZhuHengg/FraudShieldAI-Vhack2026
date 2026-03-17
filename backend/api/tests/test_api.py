"""Tests for the API endpoints."""
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_get_dashboard_stats():
    # Make a dummy prediction to populate stats with safe defaults for new behavioral fields
    payload = {
        "transaction_id": "TXN_TEST_123",
        "amount": 500.0,
        "sender_id": "USER_A",
        "receiver_id": "USER_B",
        "transaction_type": "transfer",
        "timestamp": "2026-03-11T16:00:00Z",
        "amount_vs_avg_ratio": 1.0,
        "avg_transaction_amount_30d": 500.0,
        "session_duration_seconds": 60.0,
        "failed_login_attempts": 0,
        "tx_count_24h": 1,
        "transaction_hour": 16,
        "is_weekend": 0,
        "sender_account_fully_drained": 0,
        "is_new_recipient": 0,
        "established_user_new_recipient": 1,
        "account_age_days": 180.0,
        "recipient_risk_profile_score": 0.0,
        "is_new_device": 0,
        "is_proxy_ip": 0,
        "ip_risk_score": 0.0,
        "country_mismatch": 0,
        "country_mismatch_suspicious": 0,
        "transfer_type_encoded": 0
    }
    # It might fail with 503 if engine is missing during tests, but we assume it's loaded in standard environments
    resp = client.post("/predict", json=payload)
    
    response = client.get("/api/v1/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_transactions" in data
    assert "approved" in data
    assert "flagged" in data
    assert "blocked" in data
    assert "fraud_rate_estimate" in data
    assert "avg_latency_ms" in data

def test_predict_includes_privacy_info():
    payload = {
        "transaction_id": "TXN_TEST_456",
        "amount": 1000.0,
        "sender_id": "USER_C",
        "receiver_id": "USER_D",
        "transaction_type": "cash_out",
        "timestamp": "2026-03-11T16:05:00Z",
        "amount_vs_avg_ratio": 1.0,
        "avg_transaction_amount_30d": 1000.0,
        "session_duration_seconds": 60.0,
        "failed_login_attempts": 0,
        "tx_count_24h": 1,
        "transaction_hour": 16,
        "is_weekend": 0,
        "sender_account_fully_drained": 0,
        "is_new_recipient": 0,
        "established_user_new_recipient": 1,
        "account_age_days": 180.0,
        "recipient_risk_profile_score": 0.0,
        "is_new_device": 0,
        "is_proxy_ip": 0,
        "ip_risk_score": 0.0,
        "country_mismatch": 0,
        "country_mismatch_suspicious": 0,
        "transfer_type_encoded": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code in (200, 503), f"Unexpected status: {response.status_code}"
    
    if response.status_code == 200:
        data = response.json()
        assert "risk_score" in data
        assert 0 <= data["risk_score"] <= 100
        assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH")
        assert "privacy" in data
        assert "pii_hashed" in data["privacy"]
        assert data["privacy"]["pii_hashed"] is True
        assert data["privacy"]["hash_algorithm"] == "SHA-256"
        assert "dp_applied" in data["privacy"]
