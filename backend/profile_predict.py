"""
profile_predict.py — Internal Timing Breakdown for /predict

Measures exactly how many milliseconds each step takes inside a
single fraud scoring request, so you can see where time is spent.

Usage:
    python profile_predict.py
"""

import time
import numpy as np
import requests
import json

# ─── Configuration ──────────────────────────────────────────────
API_URL = "http://localhost:8000"
NUM_RUNS = 20  # Average over N runs to smooth out noise


def build_payload():
    """Build a realistic transaction payload."""
    import random, string
    amount = round(random.uniform(10, 45000), 2)
    avg30 = round(random.uniform(100, 2000), 2)
    bal_before = round(amount + random.uniform(100, 5000), 2)
    return {
        "transaction_id": f"PROF-{''.join(random.choices(string.ascii_uppercase + string.digits, k=5))}",
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


def profile_engine_directly():
    """Profile the engine internals by importing and timing each step."""
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    from api.inference import EnsembleEngine
    from api.schemas import TransactionRequest

    print("=" * 65)
    print("  INTERNAL ENGINE PROFILING (Direct Python, no HTTP)")
    print("=" * 65)
    print(f"  Averaging over {NUM_RUNS} runs...\n")

    # Load engine once
    t0 = time.perf_counter()
    engine = EnsembleEngine(
        iso_model_dir='models/unsupervised/isolation_forest/outputs/model',
        lgb_model_dir='models/supervised/outputs/model',
        ensemble_dir='models/ensemble/outputs/model'
    )
    load_time = (time.perf_counter() - t0) * 1000
    print(f"  Engine load time (one-time):  {load_time:.1f}ms\n")

    # Warmup run (discard)
    warmup_txn = TransactionRequest(**build_payload())
    engine.predict(warmup_txn)

    # Timed runs
    timings = {
        "1. Pydantic parse":    [],
        "2. Preprocess":        [],
        "3. ISO features":      [],
        "4. ISO score":         [],
        "5. LGB features":      [],
        "6. LGB score":         [],
        "7. Behavioral score":  [],
        "8. Fusion + threshold":[],
        "9. Response build":    [],
        "TOTAL (engine only)":  [],
    }

    for _ in range(NUM_RUNS):
        payload = build_payload()

        # Step 1: Pydantic validation
        t1 = time.perf_counter()
        txn = TransactionRequest(**payload)
        t2 = time.perf_counter()
        timings["1. Pydantic parse"].append((t2 - t1) * 1000)

        # Step 2: Preprocessing
        d = engine.preprocess(txn.model_dump())
        t3 = time.perf_counter()
        timings["2. Preprocess"].append((t3 - t2) * 1000)

        # Step 3: ISO feature extraction
        X_iso = engine.get_iso_features(d)
        t4 = time.perf_counter()
        timings["3. ISO features"].append((t4 - t3) * 1000)

        # Step 4: ISO scoring
        iso_score = engine.score_iso(X_iso)
        t5 = time.perf_counter()
        timings["4. ISO score"].append((t5 - t4) * 1000)

        # Step 5: LGB feature extraction
        X_lgb = engine.get_lgb_features(d)
        t6 = time.perf_counter()
        timings["5. LGB features"].append((t6 - t5) * 1000)

        # Step 6: LGB scoring
        lgb_score = engine.score_lgb(X_lgb)
        t7 = time.perf_counter()
        timings["6. LGB score"].append((t7 - t6) * 1000)

        # Step 7: Behavioral scoring
        beh_score, beh_reasons, _ = engine._score_behavioral(d)
        t8 = time.perf_counter()
        timings["7. Behavioral score"].append((t8 - t7) * 1000)

        # Step 8: Fusion
        final_score = (lgb_score * engine.w_lgb) + (iso_score * engine.w_iso) + (beh_score * engine.w_beh)
        final_score = float(np.clip(final_score, 0.0, 100.0))
        if final_score < engine.approve_threshold:
            decision = "LOW"
        elif final_score < engine.flag_threshold:
            decision = "MEDIUM"
        else:
            decision = "HIGH"
        t9 = time.perf_counter()
        timings["8. Fusion + threshold"].append((t9 - t8) * 1000)

        # Step 9: Response build
        from api.schemas import RiskResponse, PrivacyInfo, RuleBreakdown, FeatureSnapshot
        resp = RiskResponse(
            transaction_id=txn.transaction_id,
            risk_score=round(final_score, 2),
            risk_level=decision,
            supervised_score=round(lgb_score / 100.0, 4),
            unsupervised_score=round(iso_score / 100.0, 4),
            behavioral_score=round(beh_score / 100.0, 4),
            reasons=beh_reasons,
            privacy=PrivacyInfo(pii_hashed=True, hash_algorithm="SHA-256", dp_applied=False),
            rule_breakdown=RuleBreakdown(drain_score=0, deviation_score=0, context_score=0, velocity_score=0),
            feature_snapshot=FeatureSnapshot(
                amount_vs_avg_ratio=0, ip_risk_score=0, tx_count_24h=0,
                session_duration_seconds=0, is_new_device=0, country_mismatch=0,
                sender_fully_drained=0, is_new_recipient=0, account_age_days=0, is_proxy_ip=0
            ),
        )
        t10 = time.perf_counter()
        timings["9. Response build"].append((t10 - t9) * 1000)
        timings["TOTAL (engine only)"].append((t10 - t1) * 1000)

    # --- Print Results ---
    print(f"  {'Step':<25} {'Avg (ms)':>10} {'P95 (ms)':>10}  {'% of Total':>10}  Bar")
    print("  " + "-" * 75)

    total_avg = np.mean(timings["TOTAL (engine only)"])

    for step, vals in timings.items():
        avg = np.mean(vals)
        p95 = np.percentile(vals, 95)
        pct = (avg / total_avg) * 100 if "TOTAL" not in step else 100
        bar_len = int(pct / 2)
        bar = "#" * bar_len

        if "TOTAL" in step:
            print("  " + "-" * 75)

        print(f"  {step:<25} {avg:>9.2f}ms {p95:>9.2f}ms  {pct:>9.1f}%  {bar}")

    # --- Also show HTTP round-trip overhead ---
    print(f"\n{'=' * 65}")
    print(f"  HTTP ROUND-TRIP PROFILING (via localhost)")
    print(f"{'=' * 65}")
    print(f"  Sending {NUM_RUNS} requests to {API_URL}/predict ...\n")

    http_times = []
    for _ in range(NUM_RUNS):
        payload = build_payload()
        t_start = time.perf_counter()
        r = requests.post(f"{API_URL}/predict", json=payload)
        t_end = time.perf_counter()
        if r.status_code == 200:
            http_times.append((t_end - t_start) * 1000)

    if http_times:
        engine_avg = total_avg
        http_avg = np.mean(http_times)
        http_p95 = np.percentile(http_times, 95)
        overhead = http_avg - engine_avg

        print(f"  {'Component':<30} {'Avg (ms)':>10} {'P95 (ms)':>10}")
        print("  " + "-" * 55)
        print(f"  {'Engine processing':<30} {engine_avg:>9.2f}ms")
        print(f"  {'HTTP + FastAPI + network':<30} {overhead:>9.2f}ms")
        print("  " + "-" * 55)
        print(f"  {'Total HTTP round-trip':<30} {http_avg:>9.2f}ms {http_p95:>9.2f}ms")
    else:
        print("  WARNING: Could not reach API. Is the server running?")

    print()


if __name__ == "__main__":
    profile_engine_directly()
