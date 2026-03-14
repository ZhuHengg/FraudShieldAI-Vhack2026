"""
Main Isolation Forest Pipeline execution script.
Refactored from monolithic train_isolation_forest.py.
"""
import os
import joblib

# Local imports
from data_loader import load_data, split_data
from preprocessing import preprocess_features, calculate_recipient_risk
from model import IsolationForestModel, generate_tier_summary
from evaluation import evaluate_predictions, analyze_bias

# ──────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "..", "..", "..", "data", "raw", "ewallet_transaction.csv")

OUT_MODEL   = os.path.join(BASE_DIR, "outputs", "model")
OUT_PLOTS   = os.path.join(BASE_DIR, "outputs", "plots")
OUT_RESULTS = os.path.join(BASE_DIR, "outputs", "results")

for d in [OUT_MODEL, OUT_PLOTS, OUT_RESULTS]:
    os.makedirs(d, exist_ok=True)

def main():
    # STEP 1: LOAD
    try:
        df = load_data(DATA_PATH)
    except FileNotFoundError as e:
        print(f"Error loading dataset: {e}")
        return

    # STEP 2: SPLIT (BUILD THE WALL)
    df_train, df_test = split_data(df)

    # STEP 3: RECIPIENT RISK (TRAIN ONLY)
    risk_lookup = calculate_recipient_risk(df_train)
    df_train['recipient_risk_profile_score'] = df_train['name_recipient'].map(risk_lookup).fillna(0.0)
    df_test['recipient_risk_profile_score'] = df_test['name_recipient'].map(risk_lookup).fillna(0.0)

    # STEP 4: PREPROCESSING & SCALING (FIT ON TRAIN, TRANSFORM BOTH)
    X_train, X_test, y_train, y_test, scaler, all_features = preprocess_features(df_train, df_test)

    # Calculate contamination from TRAIN only
    train_fraud_rate = y_train.mean()
    contamination = min(train_fraud_rate * 5, 0.10)
    print(f"Train fraud rate:      {train_fraud_rate:.4f}")
    print(f"Contamination used:   {contamination:.4f}\n")

    # STEP 5: MODEL TRAINING (TRAIN ONLY)
    iso_model = IsolationForestModel(
        contamination=contamination,
        n_estimators=200,
        max_samples=512,
        max_features=0.8,
    )
    iso_model.fit(X_train)

    # STEP 6: PREDICTION (TEST ONLY)
    y_pred_test, risk_scores_test = iso_model.predict(X_test)

    # Prepare results for evaluation
    df_results_test = df_test[['transaction_id', 'is_fraud']].copy()
    df_results_test['iso_risk_score'] = risk_scores_test
    df_results_test['iso_prediction'] = y_pred_test
    df_results_test['risk_tier'] = df_results_test['iso_risk_score'].apply(iso_model.assign_tier)

    # Summary
    tier_summary = generate_tier_summary(df_results_test)
    print("\n=== RISK TIER SUMMARY (TEST SET) ===")
    print(tier_summary)
    print()

    # STEP 7: EVALUATION (TEST ONLY)
    df_model_test = df_test[all_features] # Helper for evaluation function which expects individual features
    evaluate_predictions(y_test, y_pred_test, df_results_test, df_model_test, all_features, OUT_PLOTS, OUT_RESULTS)

    # STEP 8: BIAS DETECTION
    analyze_bias(iso_model.model, X_test, all_features, OUT_PLOTS, OUT_RESULTS)

    # STEP 9: SAVE OUTPUTS
    print("\n" + "=" * 60)
    print("STEP 9: SAVE OUTPUTS")
    print("=" * 60)
    
    model_path = os.path.join(OUT_MODEL, "isolation_forest_model.pkl")
    mms_path = os.path.join(OUT_MODEL, "minmax_scaler.pkl")
    scaler_path = os.path.join(OUT_MODEL, "scaler.pkl")
    
    iso_model.save(model_path, mms_path)
    joblib.dump(scaler, scaler_path)
    print(f"Standard Scaler saved to: {scaler_path}")
    
    tier_csv_path = os.path.join(OUT_RESULTS, "risk_tier_summary.csv")
    tier_summary.to_csv(tier_csv_path)
    print(f"Saved: {tier_csv_path}")

    print("\n✅  Modular pipeline complete — all outputs saved under outputs/")

if __name__ == "__main__":
    main()
