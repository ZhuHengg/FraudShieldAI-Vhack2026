"""
Quarantine Retraining Loop — Anti-Model-Poisoning (V4 Step 8)
==============================================================
When a user taps "YES" to approve an anomalous (flagged) transaction,
the data is quarantined — it does NOT immediately update the live AI.

Validation pipeline checks quarantined labels against:
  1. Biometric continuity  — same device as user's recent transactions
  2. Amount plausibility   — within 3 std devs of user's rolling average
  3. Regional trend check  — matches macro spending patterns (time/amount)

Only labels that pass >= 2/3 checks graduate from quarantine into the
training pool. This prevents scammers from "grooming" the AI with
small test transactions.
"""
import logging
import math
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import func

logger = logging.getLogger(__name__)


def should_quarantine(transaction_row, action_taken: str) -> bool:
    """
    Determine if a labeled transaction should be quarantined.
    
    A transaction enters quarantine if:
    - It was originally FLAGGED (anomalous) by the system
    - AND the analyst/user labeled it as LEGIT (confirming the anomaly was OK)
    
    Direct BLOCK+FRAUD labels are trusted (analyst agrees with system).
    Direct APPROVE+LEGIT labels are trusted (both agree it's safe).
    FLAG+LEGIT is suspicious — this is where grooming attacks happen.
    """
    if action_taken == 'FLAG' and hasattr(transaction_row, 'analyst_label'):
        if transaction_row.analyst_label == 'LEGIT':
            return True
    return False


def validate_quarantined_transaction(txn, db: Session) -> dict:
    """
    Run the 3-check validation pipeline on a quarantined transaction.
    
    Returns:
        {
            "passed": bool,
            "checks_passed": int,
            "checks_total": 3,
            "details": {
                "biometric_continuity": {"passed": bool, "reason": str},
                "amount_plausibility": {"passed": bool, "reason": str},
                "regional_trend": {"passed": bool, "reason": str},
            }
        }
    """
    from api.models import TransactionLog

    checks_passed = 0
    details = {}

    # ── Check 1: Biometric Continuity ─────────────────────────────────
    # Verify the device fingerprint matches the user's recent transactions
    recent_txns = db.query(TransactionLog).filter(
        TransactionLog.user_hash == txn.user_hash,
        TransactionLog.transaction_id != txn.transaction_id,
    ).order_by(TransactionLog.transaction_id.desc()).limit(5).all()

    if len(recent_txns) == 0:
        # New user with no history — can't verify biometrics, give benefit of doubt
        bio_passed = True
        bio_reason = "No prior history — new user bypass"
    else:
        # Check if device_type is consistent with recent transactions
        recent_devices = set(t.device_type for t in recent_txns if t.device_type)
        current_device = txn.device_type or "Unknown"
        
        if not recent_devices:
            bio_passed = True
            bio_reason = "No device data available — bypass"
        elif current_device in recent_devices:
            bio_passed = True
            bio_reason = f"Device '{current_device}' matches recent history"
        else:
            # Also check if the IP risk indicators are consistent
            recent_proxy_rate = sum(1 for t in recent_txns if t.is_proxy_ip) / len(recent_txns)
            if txn.is_proxy_ip and recent_proxy_rate < 0.2:
                bio_passed = False
                bio_reason = (f"Device '{current_device}' not in recent set {recent_devices}, "
                            f"and proxy usage inconsistent (current=proxy, history={recent_proxy_rate:.0%})")
            else:
                bio_passed = True
                bio_reason = f"Device changed but proxy pattern consistent"

    details["biometric_continuity"] = {"passed": bio_passed, "reason": bio_reason}
    if bio_passed:
        checks_passed += 1

    # ── Check 2: Amount Plausibility ──────────────────────────────────
    # Transaction amount must be within 3 standard deviations of user's average
    user_amounts = db.query(TransactionLog.amount).filter(
        TransactionLog.user_hash == txn.user_hash,
        TransactionLog.transaction_id != txn.transaction_id,
    ).all()

    if len(user_amounts) < 3:
        # Not enough history — use a generous global check
        amt_passed = (txn.amount or 0) < 50000  # Hard cap for new users
        amt_reason = f"Insufficient history ({len(user_amounts)} txns) — using hard cap (50K)"
    else:
        amounts = [a[0] for a in user_amounts if a[0] is not None]
        if not amounts:
            amt_passed = True
            amt_reason = "No valid amounts in history"
        else:
            mean_amt = sum(amounts) / len(amounts)
            variance = sum((a - mean_amt) ** 2 for a in amounts) / len(amounts)
            std_dev = max(math.sqrt(variance), 1.0)  # floor at 1.0 to avoid div-by-zero
            
            z_score = abs((txn.amount or 0) - mean_amt) / std_dev
            amt_passed = z_score <= 3.0
            amt_reason = (f"Amount={txn.amount:.2f}, Mean={mean_amt:.2f}, "
                        f"StdDev={std_dev:.2f}, Z-score={z_score:.2f} "
                        f"({'within' if amt_passed else 'EXCEEDS'} 3-sigma)")

    details["amount_plausibility"] = {"passed": amt_passed, "reason": amt_reason}
    if amt_passed:
        checks_passed += 1

    # ── Check 3: Regional Trend Check ─────────────────────────────────
    # Verify transaction matches macro spending patterns for the time/region
    # Compare against aggregate patterns of all users in the same country
    country = txn.country or "MY"

    # Get aggregate stats for the same hour and country
    hour = txn.transaction_hour or 12
    regional_txns = db.query(
        func.avg(TransactionLog.amount),
        func.count(TransactionLog.transaction_id),
    ).filter(
        TransactionLog.country == country,
        TransactionLog.transaction_hour == hour,
        TransactionLog.transaction_id != txn.transaction_id,
    ).first()

    regional_avg = regional_txns[0] if regional_txns[0] else 5000.0
    regional_count = regional_txns[1] if regional_txns[1] else 0

    if regional_count < 5:
        # Not enough regional data — bypass
        reg_passed = True
        reg_reason = f"Insufficient regional data ({regional_count} txns at hour {hour}) — bypass"
    else:
        # Transaction should be within 5x the regional average for that hour
        ratio = (txn.amount or 0) / max(regional_avg, 1.0)
        reg_passed = ratio <= 5.0
        reg_reason = (f"Amount={txn.amount:.2f}, Regional avg at hour {hour}={regional_avg:.2f}, "
                     f"Ratio={ratio:.2f}x ({'within' if reg_passed else 'EXCEEDS'} 5x limit)")

    details["regional_trend"] = {"passed": reg_passed, "reason": reg_reason}
    if reg_passed:
        checks_passed += 1

    # ── Final Decision ────────────────────────────────────────────────
    passed = checks_passed >= 2  # Must pass at least 2 of 3 checks

    return {
        "passed": passed,
        "checks_passed": checks_passed,
        "checks_total": 3,
        "details": details,
    }


def run_quarantine_validation(db: Session) -> dict:
    """
    Validate ALL quarantined transactions.
    
    Returns summary stats and per-transaction results.
    """
    from api.models import TransactionLog

    quarantined = db.query(TransactionLog).filter(
        TransactionLog.quarantine_status == 'QUARANTINED'
    ).all()

    if not quarantined:
        return {
            "total_quarantined": 0,
            "validated": 0,
            "rejected": 0,
            "results": [],
        }

    validated = 0
    rejected = 0
    results = []

    for txn in quarantined:
        result = validate_quarantined_transaction(txn, db)
        
        if result["passed"]:
            txn.quarantine_status = 'VALIDATED'
            txn.quarantine_reason = f"Passed {result['checks_passed']}/3 validation checks"
            validated += 1
        else:
            txn.quarantine_status = 'REJECTED'
            txn.quarantine_reason = f"Failed validation — only passed {result['checks_passed']}/3 checks"
            rejected += 1

        results.append({
            "transaction_id": txn.transaction_id,
            "status": txn.quarantine_status,
            "checks": result,
        })

    try:
        db.commit()
        logger.info(f"Quarantine validation complete: {validated} validated, {rejected} rejected")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to commit quarantine validation: {e}")
        raise

    return {
        "total_quarantined": len(quarantined),
        "validated": validated,
        "rejected": rejected,
        "results": results,
    }
