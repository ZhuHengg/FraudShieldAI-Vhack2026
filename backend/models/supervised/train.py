"""
Supervised LightGBM Pipeline
Mirrors structure of unsupervised/isolation_forest/train.py
All thresholds empirically derived from PR curve on validation set
No threshold values are hardcoded anywhere in this pipeline
"""
import os
import json
import joblib

from data_loader import load_data, split_data
from preprocessing import preprocess_features, calculate_recipient_risk
from model import LightGBMModel
from evaluation import (
    evaluate_predictions,
    tune_threshold,
    plot_feature_importance
)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(
    BASE_DIR, '..', '..', 'data', 'ewallet_transaction.csv'
)

OUT_MODEL   = os.path.join(BASE_DIR, 'outputs', 'model')
OUT_PLOTS   = os.path.join(BASE_DIR, 'outputs', 'plots')
OUT_RESULTS = os.path.join(BASE_DIR, 'outputs', 'results')

for d in [OUT_MODEL, OUT_PLOTS, OUT_RESULTS]:
    os.makedirs(d, exist_ok=True)

def main():

    # STEP 1: LOAD
    df = load_data(DATA_PATH)

    # STEP 2: WALL — split before everything else
    # Must happen before any feature computation
    df_train, df_val, df_test = split_data(df)

    # STEP 3: RECIPIENT RISK FROM TRAIN ONLY
    # Identical to Isolation Forest approach
    risk_lookup = calculate_recipient_risk(df_train)
    for split in [df_train, df_val, df_test]:
        split['recipient_risk_profile_score'] = (
            split['name_recipient']
            .map(risk_lookup)
            .fillna(0.0)
        )
    coverage = df_test['name_recipient'].isin(
        risk_lookup
    ).mean()
    print(f"Test recipient coverage: {coverage:.2%}")
    print(f"Unseen recipients → score 0.0")

    # STEP 4: PREPROCESSING
    # Critical order inside preprocess_features:
    #   1. Fit scaler on original X_train
    #   2. Apply SMOTE to get X_train_res
    #   3. Transform X_train_res, X_val, X_test with fitted scaler
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     scaler, all_features) = preprocess_features(
        df_train, df_val, df_test
    )

    # STEP 5: TRAIN LIGHTGBM ON TRAIN ONLY
    # X_train is SMOTE-resampled
    # X_val is original unaugmented — used for early stopping
    lgb_model = LightGBMModel()
    lgb_model.fit(X_train, y_train, X_val, y_val)

    # STEP 6: EMPIRICAL THRESHOLD TUNING ON VALIDATION SET ONLY
    # val_risk_scores on 0-100 scale to match IsoForest convention
    # tune_threshold finds argmax(F1) across all threshold points
    # This is the ONLY legitimate source of threshold values
    val_risk_scores = lgb_model.predict_scaled(X_val)
    best_threshold, best_precision, best_recall, best_f1 = (
        tune_threshold(y_val, val_risk_scores, OUT_PLOTS)
    )
    # set_threshold receives empirically derived value only
    lgb_model.set_threshold(best_threshold)

    # STEP 7: PREDICT ON TEST SET ONLY
    # Test set has never been seen by model or threshold tuning
    y_pred, risk_scores_test = lgb_model.predict_with_tier(X_test)

    df_results = df_test[['transaction_id', 'is_fraud']].copy()
    df_results['lgb_risk_score'] = risk_scores_test
    df_results['lgb_prediction'] = y_pred
    df_results['risk_tier']      = (
        df_results['lgb_risk_score'].apply(lgb_model.assign_tier)
    )

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

    # STEP 8: EVALUATION ON TEST SET ONLY
    evaluate_predictions(
        y_test, y_pred, df_results,
        all_features, OUT_PLOTS, OUT_RESULTS
    )

    # STEP 9: FEATURE IMPORTANCE
    # Native LightGBM importances — no SHAP here
    # SHAP belongs in ensemble/explainability.py
    importance = lgb_model.get_feature_importance(all_features)
    plot_feature_importance(importance, OUT_PLOTS)
    print("\nTop 5 most important features:")
    print(importance.head(5).to_string())

    # STEP 10: SAVE ALL OUTPUTS
    print("\n" + "="*60)
    print("STEP 10: SAVE OUTPUTS")
    print("="*60)

    lgb_model.save(
        os.path.join(OUT_MODEL, 'lgb_model.pkl'),
        os.path.join(OUT_MODEL, 'feature_columns.pkl')
    )
    joblib.dump(scaler, os.path.join(OUT_MODEL, 'scaler.pkl'))
    joblib.dump(
        all_features,
        os.path.join(OUT_MODEL, 'feature_columns.pkl')
    )

    # threshold_config.json stores empirically tuned values only
    # These numbers come from PR curve maximisation on validation set
    # They must NEVER be hardcoded or manually set
    # If you are tempted to write approve_threshold = 0.35
    # that is WRONG — the value must come from tune_threshold()
    threshold_config = {
        'optimal_threshold': float(best_threshold),
        # ↑ from argmax(F1) on validation set PR curve
        'flag_threshold':    float(lgb_model.flag_threshold),
        # ↑ derived: optimal + (100 - optimal) * 0.5
        'val_precision':     float(best_precision),
        # ↑ precision at optimal threshold on validation set
        'val_recall':        float(best_recall),
        # ↑ recall at optimal threshold on validation set
        'val_f1':            float(best_f1)
        # ↑ maximised F1 value — what the threshold was optimised for
    }
    threshold_path = os.path.join(OUT_MODEL, 'threshold_config.json')
    with open(threshold_path, 'w') as f:
        json.dump(threshold_config, f, indent=2)
    print(f"Threshold config saved: {threshold_path}")
    print(f"  optimal_threshold: {best_threshold:.2f} "
          f"← empirically derived from PR curve")
    print(f"  flag_threshold:    {lgb_model.flag_threshold:.2f} "
          f"← derived from optimal")

    print("\n✅ Supervised pipeline complete")
    print(f"All outputs saved under: {BASE_DIR}/outputs/")

if __name__ == "__main__":
    main()
