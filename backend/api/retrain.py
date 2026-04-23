"""
Closed-Loop Retraining Module
==============================
Re-optimizes ensemble weights and thresholds using analyst-labeled
transactions from the Neon database.

Does NOT retrain the base LightGBM or IsolationForest models.
Only re-tunes the fusion layer (weights + thresholds) based on
human feedback, which is the safest and highest-impact approach.

V4 Step 8: Quarantined labels are EXCLUDED from retraining.
Only labels with quarantine_status = NULL (direct) or VALIDATED (passed checks)
are used, preventing model poisoning from grooming attacks.
"""
import numpy as np
import pandas as pd
import json
import os
import logging
from itertools import product
from sklearn.metrics import precision_recall_curve, f1_score
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Path to the ensemble config that the inference engine reads
ENSEMBLE_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'models', 'ensemble', 'outputs', 'model', 'ensemble_config.json'
)


def fetch_labeled_data(db: Session):
    """
    Fetch all analyst-labeled transactions from Neon.
    Returns a DataFrame with features + analyst labels.
    """
    from api.models import TransactionLog

    # V4 Step 8: Exclude quarantined labels that haven't been validated
    # Only use: NULL (direct analyst label) or VALIDATED (passed quarantine checks)
    from sqlalchemy import or_
    rows = db.query(TransactionLog).filter(
        TransactionLog.analyst_label.isnot(None),
        or_(
            TransactionLog.quarantine_status.is_(None),
            TransactionLog.quarantine_status == 'VALIDATED',
        )
    ).all()

    if not rows:
        return pd.DataFrame()

    records = []
    for r in rows:
        records.append({
            'transaction_id': r.transaction_id,
            'amount': r.amount,
            'transfer_type': r.transfer_type,
            'avg_transaction_amount_30d': r.avg_transaction_amount_30d or 0,
            'amount_vs_avg_ratio': r.amount_vs_avg_ratio or 1.0,
            'transaction_hour': r.transaction_hour or 12,
            'is_weekend': r.is_weekend or 0,
            'sender_account_fully_drained': r.sender_account_fully_drained or 0,
            'is_new_device': r.is_new_device or 0,
            'session_duration_seconds': r.session_duration_seconds or 120,
            'failed_login_attempts': r.failed_login_attempts or 0,
            'is_proxy_ip': r.is_proxy_ip or 0,
            'ip_risk_score': r.ip_risk_score or 0.0,
            'country_mismatch': r.country_mismatch or 0,
            'account_age_days': r.account_age_days or 365,
            'tx_count_24h': r.tx_count_24h or 1,
            'is_new_recipient': r.is_new_recipient or 0,
            'established_user_new_recipient': r.established_user_new_recipient or 0,
            'recipient_risk_profile_score': r.recipient_risk_profile_score or 0.0,
            'ml_risk_score': r.ml_risk_score or 0.0,
            # Ground truth from analyst
            'analyst_label': r.analyst_label,
            'is_fraud': 1 if r.analyst_label == 'FRAUD' else 0,
        })

    return pd.DataFrame(records)


def score_with_engine(engine, df: pd.DataFrame):
    """
    Score a DataFrame of transactions through the real EnsembleEngine.
    Returns arrays of lgb, iso, beh scores (0-100 scale).
    """
    lgb_scores = []
    iso_scores = []
    beh_scores = []

    for _, row in df.iterrows():
        txn = row.to_dict()
        txn['sender_id'] = txn.get('user_hash', 'RETRAIN_USER')
        txn['receiver_id'] = txn.get('recipient_hash', 'RETRAIN_RECV')
        txn['transaction_type'] = txn.get('transfer_type', 'TRANSFER')
        txn['timestamp'] = '2026-01-15T12:00:00Z'

        try:
            df_proc = engine.preprocess(txn)
            X_iso = engine.get_iso_features(df_proc)
            X_lgb = engine.get_lgb_features(df_proc)

            lgb_s = engine.score_lgb(X_lgb)
            iso_s = engine.score_iso(X_iso)
            beh_s, _ = engine.score_beh(df_proc)

            lgb_scores.append(lgb_s)
            iso_scores.append(iso_s)
            beh_scores.append(beh_s)
        except Exception as e:
            logger.warning(f"Skipping transaction during retrain scoring: {e}")
            lgb_scores.append(0.0)
            iso_scores.append(0.0)
            beh_scores.append(0.0)

    return np.array(lgb_scores), np.array(iso_scores), np.array(beh_scores)


def grid_search_weights(y_true, lgb_scores, iso_scores, beh_scores, step=0.05):
    """
    Grid search for optimal ensemble weights on labeled data.
    Constraint: w_lgb + w_iso + w_beh = 1.0
    Finds combination that maximises F1 score.
    """
    best_f1 = 0
    best_weights = None
    best_threshold = None

    weight_options = np.arange(0.05, 0.95, step).round(2)

    for w_lgb, w_iso in product(weight_options, weight_options):
        w_beh = round(1.0 - w_lgb - w_iso, 2)
        if w_beh < 0.05 or w_beh > 0.90:
            continue

        # Fuse scores
        fused = lgb_scores * w_lgb + iso_scores * w_iso + beh_scores * w_beh

        # Find best threshold via PR curve
        precisions, recalls, thresholds = precision_recall_curve(y_true, fused)
        f1s = 2 * precisions * recalls / (precisions + recalls + 1e-10)
        best_f1_here = f1s.max()
        best_thresh_here = float(thresholds[np.argmax(f1s)]) if len(thresholds) > 0 else 50.0

        if best_f1_here > best_f1:
            best_f1 = best_f1_here
            best_weights = {'lgb': w_lgb, 'iso': w_iso, 'beh': w_beh}
            best_threshold = best_thresh_here

    return best_weights, best_threshold, best_f1


def run_retrain(engine, db: Session, min_samples: int = 50):
    """
    Main retraining pipeline:
    1. Fetch labeled data from Neon
    2. Score through real engine
    3. Grid-search for optimal weights/thresholds
    4. Save new config
    5. Hot-reload engine weights
    
    Returns dict with old vs new metrics for comparison.
    """
    # 1. Fetch labeled data
    df = fetch_labeled_data(db)

    if len(df) < min_samples:
        return {
            'status': 'insufficient_data',
            'samples_used': len(df),
            'old_weights': {'lgb': engine.w_lgb, 'iso': engine.w_iso, 'beh': engine.w_beh},
            'new_weights': {'lgb': engine.w_lgb, 'iso': engine.w_iso, 'beh': engine.w_beh},
            'old_thresholds': {
                'optimal_threshold': engine.approve_threshold,
                'flag_threshold': engine.flag_threshold,
            },
            'new_thresholds': {
                'optimal_threshold': engine.approve_threshold,
                'flag_threshold': engine.flag_threshold,
            },
            'old_f1': 0.0,
            'new_f1': 0.0,
            'improvement_pct': 0.0,
            'message': f'Need at least {min_samples} labeled samples, got {len(df)}.',
        }

    y_true = df['is_fraud'].values

    # 2. Save old config
    old_weights = {'lgb': engine.w_lgb, 'iso': engine.w_iso, 'beh': engine.w_beh}
    old_thresholds = {
        'optimal_threshold': engine.approve_threshold,
        'flag_threshold': engine.flag_threshold,
    }

    # 3. Score through engine
    logger.info(f"Scoring {len(df)} labeled transactions...")
    lgb_scores, iso_scores, beh_scores = score_with_engine(engine, df)

    # 4. Compute old F1
    old_fused = lgb_scores * engine.w_lgb + iso_scores * engine.w_iso + beh_scores * engine.w_beh
    old_pred = (old_fused >= engine.approve_threshold).astype(int)
    old_f1 = float(f1_score(y_true, old_pred, zero_division=0))

    # 5. Grid search for new weights
    logger.info("Running grid search for optimal weights...")
    new_weights, new_threshold, new_f1 = grid_search_weights(
        y_true, lgb_scores, iso_scores, beh_scores, step=0.05
    )

    if new_weights is None:
        new_weights = old_weights
        new_threshold = engine.approve_threshold
        new_f1 = old_f1

    # Derive flag threshold (halfway between approve and 100)
    new_flag_threshold = new_threshold + (100 - new_threshold) * 0.5

    new_thresholds = {
        'optimal_threshold': new_threshold,
        'flag_threshold': new_flag_threshold,
    }

    # 6. Compute improvement
    improvement_pct = ((new_f1 - old_f1) / max(old_f1, 1e-6)) * 100

    # 7. Compute calibration metrics at the new threshold (V4 Step 6)
    from sklearn.metrics import precision_score, recall_score
    new_fused = lgb_scores * new_weights['lgb'] + iso_scores * new_weights['iso'] + beh_scores * new_weights['beh']
    new_pred = (new_fused >= new_threshold).astype(int)
    cal_precision = float(precision_score(y_true, new_pred, zero_division=0))
    cal_recall = float(recall_score(y_true, new_pred, zero_division=0))
    cal_fpr = float(((new_pred == 1) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1))

    # 8. Save updated config
    config = {
        'weights': new_weights,
        'thresholds': new_thresholds,
        'validation_metrics': {
            'precision': cal_precision,
            'recall': cal_recall,
            'f1': float(new_f1),
            'fpr': cal_fpr,
        },
        'notes': {
            'weights_source': f'closed-loop grid search on {len(df)} analyst-labeled samples',
            'threshold_source': 'PR curve argmax F1 on labeled data',
            'scale': '0-100 for all model scores',
            'hardcoded_values': 'none — all empirically derived',
        },
        'retrain_metadata': {
            'samples_used': len(df),
            'fraud_count': int(y_true.sum()),
            'legit_count': int(len(y_true) - y_true.sum()),
            'old_f1': old_f1,
            'new_f1': float(new_f1),
            'improvement_pct': float(improvement_pct),
        }
    }

    config_path = os.path.normpath(ENSEMBLE_CONFIG_PATH)

    # Backup previous config before overwriting (V4 rollback support)
    if os.path.exists(config_path):
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
        backup_dir = os.path.join(os.path.dirname(config_path), 'history')
        os.makedirs(backup_dir, exist_ok=True)
        backup_path = os.path.join(backup_dir, f'ensemble_config_{ts}.json')
        import shutil
        shutil.copy2(config_path, backup_path)
        logger.info(f"Backed up previous config to: {backup_path}")

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"New ensemble config saved to: {config_path}")

    # 8. Hot-reload engine weights (no restart needed)
    engine.w_lgb = new_weights['lgb']
    engine.w_iso = new_weights['iso']
    engine.w_beh = new_weights['beh']
    engine.approve_threshold = new_threshold
    engine.flag_threshold = new_flag_threshold
    engine.ensemble_config = config
    logger.info("Engine weights hot-reloaded successfully")

    return {
        'status': 'success',
        'samples_used': len(df),
        'old_weights': old_weights,
        'new_weights': new_weights,
        'old_thresholds': old_thresholds,
        'new_thresholds': new_thresholds,
        'old_f1': old_f1,
        'new_f1': float(new_f1),
        'improvement_pct': float(improvement_pct),
        'message': (
            f'Retrained on {len(df)} samples. '
            f'F1: {old_f1:.4f} → {new_f1:.4f} ({improvement_pct:+.1f}%). '
            f'Weights and thresholds updated in-memory and saved to disk.'
        ),
    }
