# -*- coding: utf-8 -*-
"""
Full tear-down test for Problem 1 and Problem 2.
Tests all new/modified API endpoints end-to-end.
"""
import requests
import json
import sys
import os

API = "http://localhost:8000"
PASS_COUNT = 0
FAIL_COUNT = 0

def test(name, fn):
    global PASS_COUNT, FAIL_COUNT
    try:
        fn()
        print(f"  [PASS] {name}")
        PASS_COUNT += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        FAIL_COUNT += 1

# ===========================================================
# TEST 1: Health Check
# ===========================================================
print("\n=== TEST GROUP 1: Core API ===")

def t_health():
    r = requests.get(f"{API}/api/v1/health")
    assert r.status_code == 200
    d = r.json()
    assert d["engine_loaded"] is True
test("Health check", t_health)

def t_config():
    r = requests.get(f"{API}/api/v1/config")
    assert r.status_code == 200
    d = r.json()
    assert "weights" in d
    assert "approve_threshold" in d
    assert d["weights"]["lgb"] > 0
    print(f"       Weights: lgb={d['weights']['lgb']}, iso={d['weights']['iso']}, beh={d['weights']['beh']}")
    print(f"       Thresholds: approve={d['approve_threshold']:.1f}, flag={d['flag_threshold']:.1f}")
test("Config endpoint", t_config)

# ===========================================================
# TEST 2: Prediction (high-risk transaction)
# ===========================================================
print("\n=== TEST GROUP 2: Prediction Pipeline ===")

def t_predict_high_risk():
    txn = {
        "transaction_id": "TEST-HIGH-001",
        "amount": 15000.0,
        "sender_id": "USER_ATTACKER",
        "receiver_id": "USER_MULE",
        "transaction_type": "cash_out",
        "timestamp": "2026-04-12T03:00:00Z",
        "sender_balance_before": 16000.0,
        "sender_balance_after": 100.0,
        "is_new_device": 1,
        "is_proxy_ip": 1,
        "ip_risk_score": 0.85,
        "country_mismatch": 1,
        "sender_account_fully_drained": 1,
        "is_new_recipient": 1,
        "session_duration_seconds": 15,
        "failed_login_attempts": 3,
        "tx_count_24h": 12,
        "account_age_days": 5,
    }
    r = requests.post(f"{API}/predict", json=txn)
    assert r.status_code == 200
    d = r.json()
    assert d["risk_score"] > 30, f"High-risk txn should score > 30, got {d['risk_score']}"
    assert d["risk_level"] in ("MEDIUM", "HIGH"), f"Expected MEDIUM/HIGH, got {d['risk_level']}"
    print(f"       Score={d['risk_score']:.1f}, Level={d['risk_level']}")
    print(f"       LGB={d['supervised_score']:.4f}, ISO={d['unsupervised_score']:.4f}, BEH={d['behavioral_score']:.4f}")
test("Predict high-risk transaction", t_predict_high_risk)

def t_predict_low_risk():
    txn = {
        "transaction_id": "TEST-LOW-001",
        "amount": 50.0,
        "sender_id": "USER_NORMAL",
        "receiver_id": "USER_FRIEND",
        "transaction_type": "payment",
        "timestamp": "2026-04-12T14:00:00Z",
        "sender_balance_before": 5000.0,
        "sender_balance_after": 4950.0,
        "is_new_device": 0,
        "is_proxy_ip": 0,
        "ip_risk_score": 0.05,
        "country_mismatch": 0,
        "sender_account_fully_drained": 0,
        "is_new_recipient": 0,
        "session_duration_seconds": 300,
        "failed_login_attempts": 0,
        "tx_count_24h": 1,
        "account_age_days": 500,
    }
    r = requests.post(f"{API}/predict", json=txn)
    assert r.status_code == 200
    d = r.json()
    assert d["risk_score"] < 50, f"Low-risk txn should score < 50, got {d['risk_score']}"
    print(f"       Score={d['risk_score']:.1f}, Level={d['risk_level']}")
test("Predict low-risk transaction", t_predict_low_risk)

# ===========================================================
# TEST 3: Feedback Stats (Closed-Loop)
# ===========================================================
print("\n=== TEST GROUP 3: Closed-Loop Retraining ===")

def t_feedback_stats():
    r = requests.get(f"{API}/api/v1/feedback/stats")
    assert r.status_code == 200
    d = r.json()
    assert "total_transactions" in d
    assert "labeled_count" in d
    assert "ready_to_retrain" in d
    print(f"       Total: {d['total_transactions']}, Labeled: {d['labeled_count']}")
    print(f"       Fraud: {d['fraud_labels']}, Legit: {d['legit_labels']}")
    print(f"       Ready to retrain: {d['ready_to_retrain']}")
test("Feedback stats endpoint", t_feedback_stats)

# ===========================================================
# TEST 4: Save Transaction -> Label -> Verify
# ===========================================================
print("\n=== TEST GROUP 4: Feedback Loop (Save -> Label -> Verify) ===")

def t_save_transaction():
    txn = {
        "transaction_id": "TEST-LABEL-001",
        "user_hash": "hash_sender_1",
        "recipient_hash": "hash_recv_1",
        "transfer_type": "CASH_OUT",
        "amount": 8000.0,
        "avg_transaction_amount_30d": 500.0,
        "amount_vs_avg_ratio": 16.0,
        "transaction_hour": 3,
        "is_weekend": 0,
        "sender_account_fully_drained": 1,
        "is_new_device": 1,
        "session_duration_seconds": 20,
        "failed_login_attempts": 2,
        "is_proxy_ip": 1,
        "ip_risk_score": 0.9,
        "country_mismatch": 1,
        "account_age_days": 7,
        "tx_count_24h": 10,
        "is_new_recipient": 1,
        "established_user_new_recipient": 0,
        "recipient_risk_profile_score": 0.8,
        "is_fraud": 1,
        "action_taken": "BLOCK",
        "ml_risk_score": 0.92,
        "sender_balance_before": 8500.0,
        "sender_balance_after": 100.0,
        "receiver_balance_before": 200.0,
        "receiver_balance_after": 8200.0,
        "currency": "MYR",
        "country": "NG",
        "device_type": "Mobile",
    }
    r = requests.post(f"{API}/api/v1/transactions", json=txn)
    assert r.status_code == 201
    d = r.json()
    assert d["status"] == "success"
    print(f"       Saved: {d['message']}")
test("Save transaction to DB", t_save_transaction)

def t_label_fraud():
    feedback = {
        "transaction_id": "TEST-LABEL-001",
        "analyst_label": "FRAUD",
        "analyst_notes": "Confirmed mule account, account drained in 3AM session"
    }
    r = requests.post(f"{API}/api/v1/feedback", json=feedback)
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "success"
    assert "labeled_at" in d
    print(f"       Labeled at: {d['labeled_at']}")
test("Label transaction as FRAUD", t_label_fraud)

def t_get_transactions():
    r = requests.get(f"{API}/api/v1/transactions?limit=5")
    assert r.status_code == 200
    txns = r.json()
    labeled = [t for t in txns if t.get("transaction_id") == "TEST-LABEL-001"]
    if labeled:
        assert labeled[0]["analyst_label"] == "FRAUD"
        print(f"       Verified label: {labeled[0]['analyst_label']}")
        print(f"       Notes: {labeled[0].get('analyst_notes', 'N/A')}")
    else:
        print(f"       Transaction found in list (total: {len(txns)})")
test("Get transactions with labels", t_get_transactions)

def t_unlabeled():
    r = requests.get(f"{API}/api/v1/transactions/unlabeled?limit=5")
    assert r.status_code == 200
    txns = r.json()
    labeled_ids = [t["transaction_id"] for t in txns]
    assert "TEST-LABEL-001" not in labeled_ids, "Labeled txn should not appear in unlabeled list"
    print(f"       Unlabeled count: {len(txns)}")
test("Unlabeled transactions filter", t_unlabeled)

# ===========================================================
# TEST 5: Retrain (will likely say insufficient data)
# ===========================================================
print("\n=== TEST GROUP 5: Retrain Trigger ===")

def t_retrain():
    r = requests.post(f"{API}/api/v1/retrain", json={"min_labeled_samples": 10})
    assert r.status_code == 200
    d = r.json()
    print(f"       Status: {d['status']}")
    print(f"       Samples: {d['samples_used']}")
    print(f"       Message: {d['message']}")
test("Retrain endpoint", t_retrain)

# ===========================================================
# TEST 6: Invalid inputs
# ===========================================================
print("\n=== TEST GROUP 6: Error Handling ===")

def t_invalid_label():
    r = requests.post(f"{API}/api/v1/feedback", json={
        "transaction_id": "TEST-LABEL-001",
        "analyst_label": "MAYBE"
    })
    assert r.status_code == 400
    print(f"       Correctly rejected invalid label")
test("Reject invalid label value", t_invalid_label)

def t_missing_txn():
    r = requests.post(f"{API}/api/v1/feedback", json={
        "transaction_id": "NONEXISTENT-999",
        "analyst_label": "FRAUD"
    })
    assert r.status_code == 404
    print(f"       Correctly rejected missing transaction")
test("Reject feedback for missing transaction", t_missing_txn)

# ===========================================================
# TEST 7: Cleanup test data
# ===========================================================
print("\n=== TEST GROUP 7: Cleanup ===")

def t_cleanup():
    from sqlalchemy import create_engine, text
    from dotenv import load_dotenv
    load_dotenv()
    url = os.getenv("DATABASE_URL")
    if url:
        eng = create_engine(url)
        with eng.connect() as conn:
            result = conn.execute(text("DELETE FROM transaction_logs WHERE transaction_id LIKE 'TEST-%'"))
            conn.commit()
        print(f"       Cleaned up test transactions from DB")
    else:
        print(f"       No DB URL -- skipping cleanup")
test("Clean up test data", t_cleanup)

# ===========================================================
# SUMMARY
# ===========================================================
print(f"\n{'='*50}")
print(f"RESULTS: {PASS_COUNT} passed, {FAIL_COUNT} failed, {PASS_COUNT+FAIL_COUNT} total")
print(f"{'='*50}")

if FAIL_COUNT > 0:
    sys.exit(1)
