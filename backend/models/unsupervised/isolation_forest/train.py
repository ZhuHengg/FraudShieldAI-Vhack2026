"""
Main Isolation Forest Pipeline execution script.
Refactored from monolithic train_isolation_forest.py.
"""
import os
import joblib

# Local imports
from data_loader import load_data
from preprocessing import preprocess_features
from model import IsolationForestModel, generate_tier_summary
from evaluation import evaluate_predictions

# ──────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
# dataset is at backend/data/raw/ewallet_transaction.csv
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

    # STEP 2 & 3: PREPROCESSING & SCALING
    X, df_model, y_true, all_features, scaler = preprocess_features(df)

    # Calculate actual contamination
    fraud_rate = y_true.mean()
    contamination = min(fraud_rate * 5, 0.10)
    
    print(f"True fraud rate:      {fraud_rate:.4f}")
    print(f"Contamination used:   {contamination:.4f}")
    print(f"Aggressive factor:    5x\n")

    # STEP 4: MODEL INIT & TRAINING
    iso_model = IsolationForestModel(
        contamination=contamination,
        n_estimators=200,
        max_samples=512,
        max_features=0.8,
    )
    
    # Fit and get predictions / risk scores
    iso_preds, iso_risk_score = iso_model.fit_predict(X)

    # STEP 5: RISK TIERING
    df_results = df[['transaction_id', 'is_fraud']].copy()
    df_results['iso_risk_score'] = iso_risk_score
    df_results['iso_prediction'] = iso_preds
    df_results['risk_tier'] = df_results['iso_risk_score'].apply(iso_model.assign_tier)

    tier_summary = generate_tier_summary(df_results)
    print("\n=== RISK TIER SUMMARY ===")
    print(tier_summary)
    print()

    # STEP 6: EVALUATION
    top10 = evaluate_predictions(y_true, iso_preds, df_results, df_model, all_features, OUT_PLOTS, OUT_RESULTS)

    # STEP 7: SAVE OUTPUTS
    print("\n" + "=" * 60)
    print("STEP 7: SAVE OUTPUTS")
    print("=" * 60)
    
    # Save Model & Scalers
    model_path = os.path.join(OUT_MODEL, "isolation_forest_model.pkl")
    mms_path = os.path.join(OUT_MODEL, "minmax_scaler.pkl")
    scaler_path = os.path.join(OUT_MODEL, "scaler.pkl")
    
    iso_model.save(model_path, mms_path)
    joblib.dump(scaler, scaler_path)
    print(f"Standard Scaler saved to: {scaler_path}")
    
    # Save Tier Summary
    tier_csv_path = os.path.join(OUT_RESULTS, "risk_tier_summary.csv")
    tier_summary.to_csv(tier_csv_path)
    print(f"Saved: {tier_csv_path}")

    print("\n✅  Pipeline complete — all modular outputs saved under outputs/")

if __name__ == "__main__":
    main()
