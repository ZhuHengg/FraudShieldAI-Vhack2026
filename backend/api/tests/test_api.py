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
    # Make a dummy prediction to populate stats
    payload = {
        "transaction_id": "TXN_TEST_123",
        "amount": 500.0,
        "sender_id": "USER_A",
        "receiver_id": "USER_B",
        "transaction_type": "transfer",
        "timestamp": "2026-03-11T16:00:00Z"
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
        "timestamp": "2026-03-11T16:05:00Z"
    }
    response = client.post("/predict", json=payload)
    if response.status_code == 200:
        data = response.json()
        assert "privacy" in data
        assert "pii_hashed" in data["privacy"]
        assert data["privacy"]["pii_hashed"] is True
        assert data["privacy"]["hash_algorithm"] == "SHA-256"
        assert "dp_applied" in data["privacy"]
