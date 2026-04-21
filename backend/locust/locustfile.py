"""
locustfile.py — FraudShield AI Load Test

Simulates realistic concurrent traffic against the FastAPI backend.
Endpoints tested:
  - GET  /api/v1/health            (lightweight, high frequency)
  - POST /predict                  (ML inference — the hot path)
  - GET  /api/v1/stats             (dashboard polling)
  - POST /api/v1/transactions      (DB write)
  - GET  /api/v1/transactions      (DB read)
  - GET  /api/v1/transactions/search?q=...  (DB search)
  - POST /api/v1/explain/{txn_id}  (SHAP — expensive)
"""

import random
import string
import time
from locust import HttpUser, task, between, tag


# ─── Helpers ─────────────────────────────────────────────────────
def _uid(prefix="TXN"):
    return f"{prefix}-{''.join(random.choices(string.ascii_uppercase + string.digits, k=5))}"


def _random_transaction_payload():
    """Build a payload that matches TransactionRequest schema."""
    amount = round(random.uniform(10, 45000), 2)
    avg30 = round(random.uniform(100, 2000), 2)
    bal_before = round(amount + random.uniform(100, 5000), 2)
    return {
        "transaction_id": _uid(),
        "amount": amount,
        "sender_id": str(random.randint(1000000000, 9999999999)),
        "receiver_id": str(random.randint(1000000000, 9999999999)),
        "transaction_type": random.choice(["transfer", "cash_out", "payment"]),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sender_balance_before": bal_before,
        "sender_balance_after": round(bal_before - amount, 2),
        "receiver_balance_before": round(random.uniform(100, 10000), 2),
        "receiver_balance_after": round(random.uniform(100, 10000) + amount, 2),
        "amount_vs_avg_ratio": round(amount / max(avg30, 1), 3),
        "avg_transaction_amount_30d": avg30,
        "session_duration_seconds": random.randint(5, 2400),
        "failed_login_attempts": random.randint(0, 4),
        "tx_count_24h": random.randint(1, 20),
        "transaction_hour": random.randint(0, 23),
        "is_weekend": random.choice([0, 1]),
        "sender_account_fully_drained": random.choice([0, 0, 0, 1]),
        "is_new_recipient": random.choice([0, 1]),
        "established_user_new_recipient": random.choice([0, 1]),
        "account_age_days": random.randint(1, 2000),
        "recipient_risk_profile_score": round(random.uniform(0, 1), 3),
        "is_new_device": random.choice([0, 1]),
        "is_proxy_ip": random.choice([0, 0, 0, 1]),
        "ip_risk_score": round(random.uniform(0, 1), 3),
        "country_mismatch": random.choice([0, 0, 1]),
    }


def _save_transaction_payload(predict_payload, decision, risk_score):
    """Build payload matching TransactionLogCreate from a predict payload."""
    return {
        "transaction_id": predict_payload["transaction_id"],
        "user_hash": predict_payload["sender_id"],
        "recipient_hash": predict_payload["receiver_id"],
        "transfer_type": predict_payload["transaction_type"].upper(),
        "amount": predict_payload["amount"],
        "avg_transaction_amount_30d": predict_payload["avg_transaction_amount_30d"],
        "amount_vs_avg_ratio": predict_payload["amount_vs_avg_ratio"],
        "transaction_hour": predict_payload["transaction_hour"],
        "is_weekend": predict_payload["is_weekend"],
        "sender_account_fully_drained": predict_payload["sender_account_fully_drained"],
        "is_new_device": predict_payload["is_new_device"],
        "session_duration_seconds": predict_payload["session_duration_seconds"],
        "failed_login_attempts": predict_payload["failed_login_attempts"],
        "is_proxy_ip": predict_payload["is_proxy_ip"],
        "ip_risk_score": predict_payload["ip_risk_score"],
        "country_mismatch": predict_payload["country_mismatch"],
        "account_age_days": predict_payload["account_age_days"],
        "tx_count_24h": predict_payload["tx_count_24h"],
        "is_new_recipient": predict_payload["is_new_recipient"],
        "established_user_new_recipient": predict_payload["established_user_new_recipient"],
        "recipient_risk_profile_score": predict_payload["recipient_risk_profile_score"],
        "is_fraud": 0,
        "action_taken": decision,
        "ml_risk_score": risk_score,
        "sender_balance_before": predict_payload["sender_balance_before"],
        "sender_balance_after": predict_payload["sender_balance_after"],
        "receiver_balance_before": predict_payload["receiver_balance_before"],
        "receiver_balance_after": predict_payload["receiver_balance_after"],
        "currency": "MYR",
        "country": random.choice(["MY", "SG", "TH", "ID"]),
        "device_type": random.choice(["Mobile", "Desktop", "Tablet"]),
    }


# ─── Locust User ────────────────────────────────────────────────
class FraudShieldUser(HttpUser):
    """
    Simulates a frontend client or API consumer.
    wait_time: 0.5–2 seconds between tasks (realistic user pacing).
    """
    wait_time = between(0.5, 2)

    @tag("smoke", "core")
    @task(1)
    def health_check(self):
        """Lightweight health probe."""
        self.client.get("/api/v1/health", name="/health")

    @tag("smoke", "core", "predict")
    @task(5)
    def score_transaction(self):
        """Primary ML inference endpoint — highest weight."""
        payload = _random_transaction_payload()
        with self.client.post(
            "/predict",
            json=payload,
            name="/predict",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                # Occasionally save to DB (mimics frontend behavior)
                if random.random() < 0.3:
                    decision = (
                        "APPROVE" if data["risk_level"] == "LOW"
                        else "FLAG" if data["risk_level"] == "MEDIUM"
                        else "BLOCK"
                    )
                    save_payload = _save_transaction_payload(
                        payload, decision, data["risk_score"] / 100
                    )
                    self.client.post(
                        "/api/v1/transactions",
                        json=save_payload,
                        name="/transactions [save]",
                    )
            elif response.status_code == 503:
                response.failure("Engine not ready (503)")
            else:
                response.failure(f"Unexpected {response.status_code}")

    @tag("smoke", "core")
    @task(2)
    def get_stats(self):
        """Dashboard stats polling."""
        self.client.get("/api/v1/stats", name="/stats")

    @tag("core")
    @task(1)
    def get_transactions(self):
        """Fetch recent transaction list."""
        self.client.get("/api/v1/transactions?limit=50", name="/transactions [list]")

    @tag("core")
    @task(1)
    def search_transactions(self):
        """Search for a transaction by partial ID."""
        q = _uid("TXN")[:6]  # e.g. "TXN-AB"
        self.client.get(
            f"/api/v1/transactions/search?q={q}",
            name="/transactions/search",
        )

    @tag("heavy")
    @task(1)
    def explain_transaction(self):
        """SHAP explanation — computationally expensive."""
        payload = _random_transaction_payload()
        txn_id = payload["transaction_id"]
        self.client.post(
            f"/api/v1/explain/{txn_id}",
            json=payload,
            name="/explain",
        )
