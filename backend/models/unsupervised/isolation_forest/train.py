"""
Main Isolation Forest Pipeline execution script.
Refactored from monolithic isolation_forest.py.
"""
import os
import warnings

# Local imports
from data_loader import load_data
from preprocessing import preprocess_features
from model import IsolationForestModel, generate_tier_summary, assign_tier
from evaluation import evaluate_predictions

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
# dataset is at backend/data/paysim.csv according to original root path logic
DATA_PATH   = os.path.join(BASE_DIR, "..", "..", "..", "data", "raw", "paysim.csv")

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
        print("Please place paysim.csv at backend/data/raw/paysim.csv")
        return

    # STEP 2 & 3: PREPROCESSING & SCALING
    X_scaled, df_filtered, y_true, fraud_rate = preprocess_features(df)

    # STEP 4: MODEL INIT & TRAINING
    iso_model = IsolationForestModel(contamination=fraud_rate, n_estimators=100)
    
    # Fit and get predictions / risk scores
    iso_preds, iso_risk_score = iso_model.fit_predict(X_scaled)

    # Attach predictions back to dataframe
    df_filtered["iso_risk_score"] = iso_risk_score
    df_filtered["iso_prediction"] = iso_preds

    # STEP 5: RISK TIERING
    df_filtered["risk_tier"] = df_filtered["iso_risk_score"].apply(assign_tier)
    tier_summary = generate_tier_summary(df_filtered)

    print("\nRisk Tier Summary")
    print(tier_summary.to_string())
    print()

    # STEP 6: EVALUATION (saves plots/txt/csv as well)
    top10 = evaluate_predictions(y_true, iso_preds, df_filtered, OUT_PLOTS, OUT_RESULTS)

    # STEP 7: SAVE MODEL & TIER SUMMARY
    print("\n" + "=" * 60)
    print("STEP 7: SAVE OUTPUTS")
    print("=" * 60)
    
    # Save Model
    model_path = os.path.join(OUT_MODEL, "isolation_forest_model.pkl")
    iso_model.save(model_path)
    
    # Save Tier Summary
    tier_csv_path = os.path.join(OUT_RESULTS, "risk_tier_summary.csv")
    tier_summary.to_csv(tier_csv_path)
    print(f"Saved: {tier_csv_path}")

    print("\n✅  Pipeline complete — all modular outputs saved under outputs/")

if __name__ == "__main__":
    main()
