# -*- coding: utf-8 -*-
"""
V4 Architecture Full Test Suite
Tests Steps 4 (Anti-Mule), 5 (HA Fallback), 6 (Calibration), 8 (Quarantine)
Plus all existing endpoints.
"""
import requests
import json
import sys
import os
import time

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

# ===== TEST GROUP 1: Core API =====
print("\n=== TEST 1: Core API ===")

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
    print(f"       Weights: lgb={d['weights']['lgb']}, iso={d['weights']['iso']}, beh={d['weights']['beh']}")
test("Config endpoint", t_config)

# ===== TEST GROUP 2: Prediction + V4 fields =====
print("\n=== TEST 2: Prediction with V4 Fields ===")

def t_predict_v4_fields():
    txn = {
        "transaction_id": "V4-TEST-001",
        "amount": 15000.0,
        "sender_id": "V4_SENDER_1",
        "receiver_id": "V4_RECEIVER_1",
        "transaction_type": "cash_out",
        "timestamp": "2026-04-21T03:00:00Z",
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
    # Check V4 fields exist
    assert "mule_flag" in d, "Missing mule_flag"
    assert "engine_mode" in d, "Missing engine_mode"
    assert "active_models" in d, "Missing active_models"
    assert "calibration_source" in d, "Missing calibration_source"
    assert d["engine_mode"] == "full", f"Expected full mode, got {d['engine_mode']}"
    assert d["mule_flag"] == False, "Should not be mule flagged"
    assert "lgb" in d["active_models"]
    assert "iso" in d["active_models"]
    assert "beh" in d["active_models"]
    print(f"       Score={d['risk_score']:.1f}, Mode={d['engine_mode']}, Models={d['active_models']}")
    print(f"       Mule={d['mule_flag']}, Calibration={d['calibration_source']}")
test("Predict with V4 fields", t_predict_v4_fields)

# ===== TEST GROUP 3: Anti-Mule Layer (Step 4) =====
print("\n=== TEST 3: Anti-Mule Layer (Step 4) ===")

def t_velocity_stats():
    r = requests.get(f"{API}/api/v1/velocity/stats")
    assert r.status_code == 200
    d = r.json()
    assert "total_checks" in d
    assert "mule_detections" in d
    assert "sender_threshold" in d
    print(f"       Checks={d['total_checks']}, Detections={d['mule_detections']}, Threshold={d['sender_threshold']}")
test("Velocity stats endpoint", t_velocity_stats)

def t_mule_detection():
    """Send 11 transactions from different senders to same recipient -> should trigger mule"""
    target_receiver = "MULE_TARGET_WALLET"
    mule_triggered = False
    
    for i in range(12):
        txn = {
            "transaction_id": f"MULE-TEST-{i:03d}",
            "amount": 500.0,
            "sender_id": f"UNIQUE_SENDER_{i}",
            "receiver_id": target_receiver,
            "transaction_type": "transfer",
            "timestamp": "2026-04-21T12:00:00Z",
            "sender_balance_before": 5000.0,
            "sender_balance_after": 4500.0,
        }
        r = requests.post(f"{API}/predict", json=txn)
        assert r.status_code == 200
        d = r.json()
        
        if d.get("mule_flag"):
            mule_triggered = True
            print(f"       Mule triggered at sender #{i+1}: {d['reasons'][0][:60]}...")
            print(f"       Score={d['risk_score']}, Mode={d['engine_mode']}")
            break
    
    assert mule_triggered, f"Mule should have triggered after 10+ unique senders, but didn't after {i+1}"
test("Mule detection (10+ unique senders)", t_mule_detection)

# ===== TEST GROUP 4: Calibration (Step 6) =====
print("\n=== TEST 4: Calibration (Step 6) ===")

def t_calibration():
    r = requests.get(f"{API}/api/v1/calibration")
    assert r.status_code == 200
    d = r.json()
    assert "approve_threshold" in d
    assert "flag_threshold" in d
    assert "calibration_source" in d
    assert "weights" in d
    print(f"       Approve={d['approve_threshold']:.1f}, Flag={d['flag_threshold']:.1f}")
    print(f"       Source: {d['calibration_source']}")
    print(f"       Weights: {d['weights']}")
    if d.get("validation_metrics"):
        print(f"       Metrics: {d['validation_metrics']}")
test("Calibration endpoint", t_calibration)

# ===== TEST GROUP 5: Save + Label + Quarantine (Step 8) =====
print("\n=== TEST 5: Quarantine Pipeline (Step 8) ===")

def t_save_flagged_txn():
    """Save a transaction that was FLAGGED by the system"""
    txn = {
        "transaction_id": "QUARANTINE-TEST-001",
        "user_hash": "hash_qtest_sender",
        "recipient_hash": "hash_qtest_recv",
        "transfer_type": "TRANSFER",
        "amount": 8000.0,
        "avg_transaction_amount_30d": 500.0,
        "amount_vs_avg_ratio": 16.0,
        "transaction_hour": 3,
        "is_weekend": 0,
        "sender_account_fully_drained": 0,
        "is_new_device": 1,
        "session_duration_seconds": 45,
        "failed_login_attempts": 0,
        "is_proxy_ip": 0,
        "ip_risk_score": 0.3,
        "country_mismatch": 0,
        "account_age_days": 90,
        "tx_count_24h": 3,
        "is_new_recipient": 1,
        "established_user_new_recipient": 0,
        "recipient_risk_profile_score": 0.2,
        "is_fraud": 0,
        "action_taken": "FLAG",
        "ml_risk_score": 0.55,
        "sender_balance_before": 20000.0,
        "sender_balance_after": 12000.0,
        "receiver_balance_before": 500.0,
        "receiver_balance_after": 8500.0,
        "currency": "MYR",
        "country": "MY",
        "device_type": "Mobile",
    }
    r = requests.post(f"{API}/api/v1/transactions", json=txn)
    assert r.status_code == 201
    print(f"       Saved flagged transaction")
test("Save flagged transaction", t_save_flagged_txn)

def t_label_flagged_as_legit():
    """Label a FLAGGED transaction as LEGIT -> should be quarantined"""
    feedback = {
        "transaction_id": "QUARANTINE-TEST-001",
        "analyst_label": "LEGIT",
        "analyst_notes": "User confirmed this transfer was intentional"
    }
    r = requests.post(f"{API}/api/v1/feedback", json=feedback)
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "success"
    assert d["quarantined"] == True, f"FLAG+LEGIT should be quarantined, got quarantined={d.get('quarantined')}"
    print(f"       Labeled at: {d['labeled_at']}")
    print(f"       Quarantined: {d['quarantined']}")
test("FLAG+LEGIT -> auto-quarantine", t_label_flagged_as_legit)

def t_quarantine_stats():
    r = requests.get(f"{API}/api/v1/quarantine/stats")
    assert r.status_code == 200
    d = r.json()
    assert d["pending"] >= 1, f"Should have at least 1 pending quarantine, got {d['pending']}"
    print(f"       Pending={d['pending']}, Validated={d['validated']}, Rejected={d['rejected']}")
    print(f"       Direct labels={d['direct_labels']}")
test("Quarantine stats", t_quarantine_stats)

def t_quarantine_validate():
    r = requests.post(f"{API}/api/v1/quarantine/validate")
    assert r.status_code == 200
    d = r.json()
    assert d["total_quarantined"] >= 1
    print(f"       Processed: {d['total_quarantined']}")
    print(f"       Validated: {d['validated']}, Rejected: {d['rejected']}")
    if d.get("results"):
        for result in d["results"][:2]:
            print(f"       - {result['transaction_id']}: {result['status']}")
            checks = result.get("checks", {})
            for check_name, check_detail in checks.get("details", {}).items():
                print(f"         {check_name}: {'PASS' if check_detail['passed'] else 'FAIL'} - {check_detail['reason'][:60]}")
test("Quarantine validation", t_quarantine_validate)

# ===== TEST GROUP 6: Direct label (no quarantine) =====
print("\n=== TEST 6: Direct Label (no quarantine) ===")

def t_save_blocked_txn():
    txn = {
        "transaction_id": "DIRECT-LABEL-001",
        "user_hash": "hash_direct_sender",
        "recipient_hash": "hash_direct_recv",
        "transfer_type": "CASH_OUT",
        "amount": 25000.0,
        "avg_transaction_amount_30d": 500.0,
        "amount_vs_avg_ratio": 50.0,
        "transaction_hour": 2,
        "is_weekend": 1,
        "sender_account_fully_drained": 1,
        "is_new_device": 1,
        "session_duration_seconds": 10,
        "failed_login_attempts": 5,
        "is_proxy_ip": 1,
        "ip_risk_score": 0.95,
        "country_mismatch": 1,
        "account_age_days": 3,
        "tx_count_24h": 15,
        "is_new_recipient": 1,
        "established_user_new_recipient": 0,
        "recipient_risk_profile_score": 0.9,
        "is_fraud": 1,
        "action_taken": "BLOCK",
        "ml_risk_score": 0.95,
        "sender_balance_before": 25500.0,
        "sender_balance_after": 100.0,
        "receiver_balance_before": 200.0,
        "receiver_balance_after": 25200.0,
        "currency": "MYR",
        "country": "NG",
        "device_type": "Mobile",
    }
    r = requests.post(f"{API}/api/v1/transactions", json=txn)
    assert r.status_code == 201
test("Save blocked transaction", t_save_blocked_txn)

def t_label_blocked_as_fraud():
    """Label a BLOCKED transaction as FRAUD -> should NOT be quarantined (direct label)"""
    feedback = {
        "transaction_id": "DIRECT-LABEL-001",
        "analyst_label": "FRAUD",
        "analyst_notes": "Confirmed fraud - account takeover"
    }
    r = requests.post(f"{API}/api/v1/feedback", json=feedback)
    assert r.status_code == 200
    d = r.json()
    assert d["quarantined"] == False, f"BLOCK+FRAUD should NOT be quarantined"
    print(f"       Quarantined: {d['quarantined']} (correct - direct label)")
test("BLOCK+FRAUD -> direct label (no quarantine)", t_label_blocked_as_fraud)

# ===== TEST GROUP 7: Cleanup =====
print("\n=== TEST 7: Cleanup ===")

def t_cleanup():
    from sqlalchemy import create_engine, text
    from dotenv import load_dotenv
    load_dotenv()
    url = os.getenv("DATABASE_URL")
    if url:
        eng = create_engine(url)
        with eng.connect() as conn:
            result = conn.execute(text(
                "DELETE FROM transaction_logs WHERE transaction_id LIKE 'V4-%' "
                "OR transaction_id LIKE 'MULE-%' "
                "OR transaction_id LIKE 'QUARANTINE-%' "
                "OR transaction_id LIKE 'DIRECT-%'"
            ))
            conn.commit()
        print(f"       Cleaned up test data")
    else:
        print(f"       No DB URL - skipping cleanup")
test("Clean up test data", t_cleanup)

# ===== SUMMARY =====
print(f"\n{'='*55}")
print(f"V4 RESULTS: {PASS_COUNT} passed, {FAIL_COUNT} failed, {PASS_COUNT+FAIL_COUNT} total")
print(f"{'='*55}")

if FAIL_COUNT > 0:
    sys.exit(1)
