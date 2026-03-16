"""
Ensemble Pipeline
Loads trained models, tunes weights and threshold on
validation set, evaluates on test set.
No model retraining — all models already trained.
All weights and thresholds empirically derived.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(
    BASE_DIR, '..', '..', 'data', 'ewallet_transaction.csv'
)
ISO_DIR     = os.path.join(
    BASE_DIR, '..', 'unsupervised',
    'isolation_forest', 'outputs', 'model'
)
LGB_DIR     = os.path.join(
    BASE_DIR, '..', 'supervised', 'outputs', 'model'
)
OUT_MODEL   = os.path.join(BASE_DIR, 'outputs', 'model')
OUT_PLOTS   = os.path.join(BASE_DIR, 'outputs', 'plots')
OUT_RESULTS = os.path.join(BASE_DIR, 'outputs', 'results')

for d in [OUT_MODEL, OUT_PLOTS, OUT_RESULTS]:
    os.makedirs(d, exist_ok=True)

from ensemble_model import EnsembleModel
from behavioral_profiler import BehavioralProfiler
from score_fusion import ScoreFusion
from evaluation import evaluate_ensemble, plot_weight_heatmap

def main():

    # STEP 1: LOAD DATA
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"Shape: {df.shape}")
    print(f"Fraud rate: {df['is_fraud'].mean():.4f}")

    # STEP 2: IDENTICAL SPLIT AS BOTH TRAINED MODELS
    # random_state=42 guarantees same rows as training splits
    df_train_full, df_test = train_test_split(
        df, test_size=0.20, random_state=42,
        stratify=df['is_fraud']
    )
    df_train, df_val = train_test_split(
        df_train_full, test_size=0.20, random_state=42,
        stratify=df_train_full['is_fraud']
    )
    y_val  = df_val['is_fraud'].values
    y_test = df_test['is_fraud'].values

    print(f"\nSplit verification:")
    print(f"Train: {len(df_train):,} | "
          f"Fraud rate: {df_train['is_fraud'].mean():.4f}")
    print(f"Val:   {len(df_val):,}  | "
          f"Fraud rate: {y_val.mean():.4f}")
    print(f"Test:  {len(df_test):,} | "
          f"Fraud rate: {y_test.mean():.4f}")

    # STEP 3: SHARED PREPROCESSING
    # Recipient risk from train only
    # All engineered features computed consistently
    sys.path.append(
        os.path.join(BASE_DIR, '..', 'supervised')
    )
    from preprocessing import calculate_recipient_risk

    risk_lookup = calculate_recipient_risk(df_train)
    for split in [df_train, df_val, df_test]:
        split['recipient_risk_profile_score'] = (
            split['name_recipient']
            .map(risk_lookup)
            .fillna(0.0)
        )

    # Engineered features needed by both models
    for split in [df_val, df_test]:
        split['log_amount'] = np.log1p(split['amount'])
        split['log_avg_30d'] = np.log1p(
            split['avg_transaction_amount_30d']
        )
        split['transfer_type_encoded'] = split[
            'transfer_type'
        ].map({'CASH_OUT': 0, 'TRANSFER': 1})
        split['country_mismatch_suspicious'] = (
            (split['country_mismatch'] == 1) &
            (
                (split['ip_risk_score'] > 0.50)          |
                (split['session_duration_seconds'] < 30)  |
                (split['account_age_days'] < 30)          |
                (split['is_new_device'] == 1)
            )
        ).astype(int)

    coverage = df_test['name_recipient'].isin(
        risk_lookup
    ).mean()
    print(f"\nTest recipient coverage: {coverage:.2%}")
    print(f"Unseen recipients → score 0.0")

    # STEP 4: LOAD ALL TRAINED MODELS
    print("\nLoading trained model artifacts...")
    ensemble = EnsembleModel(
        iso_model_dir=ISO_DIR,
        lgb_model_dir=LGB_DIR
    )
    profiler = BehavioralProfiler()

    # STEP 5: SCORE VALIDATION SET
    # Used for weight tuning and threshold tuning only
    # Test set not touched until step 8
    print("\nScoring validation set...")
    X_val_iso = ensemble.get_iso_features(df_val)
    X_val_lgb = ensemble.get_lgb_features(df_val)

    lgb_val        = ensemble.score_lgb(X_val_lgb)
    iso_val        = ensemble.score_iso(X_val_iso)
    beh_val, _     = ensemble.score_beh(df_val, profiler)

    print(f"\nValidation score distributions:")
    print(f"  LGB: mean={lgb_val.mean():.2f} "
          f"min={lgb_val.min():.2f} "
          f"max={lgb_val.max():.2f}")
    print(f"  ISO: mean={iso_val.mean():.2f} "
          f"min={iso_val.min():.2f} "
          f"max={iso_val.max():.2f}")
    print(f"  BEH: mean={beh_val.mean():.2f} "
          f"min={beh_val.min():.2f} "
          f"max={beh_val.max():.2f}")

    # STEP 6: TUNE WEIGHTS ON VALIDATION SET ONLY
    # Grid search finds optimal w_lgb, w_iso, w_beh
    # Constraint: weights sum to 1.0
    # NEVER hardcode weights
    print("\nTuning ensemble weights on validation set...")
    fusion = ScoreFusion()
    best_weights, best_weight_f1, weight_results = (
        fusion.tune_weights(
            y_val, lgb_val, iso_val, beh_val, step=0.10
        )
    )
    plot_weight_heatmap(weight_results, OUT_PLOTS)

    # STEP 7: TUNE THRESHOLD ON VALIDATION SET ONLY
    # PR curve on fused validation scores
    # Finds threshold maximising F1
    # NEVER hardcode threshold
    print("\nTuning ensemble threshold on validation set...")
    fused_val = fusion.fuse(lgb_val, iso_val, beh_val)
    best_threshold, val_precision, val_recall, val_f1 = (
        fusion.tune_threshold(y_val, fused_val, OUT_PLOTS)
    )

    # STEP 8: SCORE TEST SET ONLY
    # Test set has never influenced weights or threshold
    print("\nScoring test set...")
    X_test_iso = ensemble.get_iso_features(df_test)
    X_test_lgb = ensemble.get_lgb_features(df_test)

    lgb_test           = ensemble.score_lgb(X_test_lgb)
    iso_test           = ensemble.score_iso(X_test_iso)
    beh_test, reasons  = ensemble.score_beh(df_test, profiler)
    fused_test         = fusion.fuse(lgb_test, iso_test, beh_test)

    # Binary predictions using empirically tuned threshold
    y_pred = (
        fused_test >= fusion.approve_threshold
    ).astype(int)

    # Build results dataframe
    df_results = df_test[['transaction_id', 'is_fraud']].copy()
    df_results['lgb_risk_score']      = lgb_test
    df_results['iso_risk_score']      = iso_test
    df_results['beh_risk_score']      = beh_test
    df_results['ensemble_risk_score'] = fused_test
    df_results['ensemble_prediction'] = y_pred
    df_results['risk_tier']           = fusion.get_decisions(
        fused_test
    )
    df_results['beh_reasons']         = reasons

    # Tier summary
    tier_summary = df_results.groupby('risk_tier').agg(
        transaction_count=('is_fraud', 'count'),
        fraud_count=('is_fraud', 'sum'),
        fraud_rate=('is_fraud', 'mean')
    ).round(4)
    print("\n=== RISK TIER SUMMARY (TEST SET) ===")
    print(tier_summary)
    tier_summary.to_csv(
        os.path.join(OUT_RESULTS, 'risk_tier_summary.csv')
    )

    # STEP 9: EVALUATE ON TEST SET ONLY
    evaluate_ensemble(
        y_test, y_pred, df_results,
        fusion.approve_threshold,
        fusion.flag_threshold,
        OUT_PLOTS, OUT_RESULTS
    )

    # STEP 10: SAVE ENSEMBLE CONFIG
    # All values empirically derived — nothing hardcoded
    fusion.save_config(
        path=os.path.join(OUT_MODEL, 'ensemble_config.json'),
        val_precision=val_precision,
        val_recall=val_recall,
        val_f1=val_f1
    )

    # Final summary
    print("\n" + "="*60)
    print("ENSEMBLE PIPELINE COMPLETE")
    print("="*60)
    print(f"Optimal weights (grid search on val set):")
    print(f"  LightGBM:    {fusion.w_lgb:.2f}")
    print(f"  IsoForest:   {fusion.w_iso:.2f}")
    print(f"  Behavioural: {fusion.w_beh:.2f}")
    print(f"\nOptimal threshold: {fusion.approve_threshold:.2f} "
          f"← PR curve argmax F1 on val set")
    print(f"Flag threshold:    {fusion.flag_threshold:.2f} "
          f"← derived from optimal")
    print(f"\n✅ All outputs saved under: {BASE_DIR}/outputs/")

if __name__ == "__main__":
    main()