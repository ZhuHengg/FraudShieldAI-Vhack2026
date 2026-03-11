import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_features(df: pd.DataFrame):
    """
    Applies feature engineering, filters transaction types, and scales the features.
    
    Returns:
        X_scaled (np.ndarray): The scaled feature matrix ready for Isolation Forest.
        df_filtered (pd.DataFrame): The filtered dataframe for risk tier mapping.
        y_true (np.ndarray): The ground truth label array.
        fraud_rate (float): The actual fraud rate in the filtered dataset.
    """
    print("=" * 60)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 60)

    # 1. Extract isMerchant_Dest before dropping name columns
    df["isMerchant_Dest"] = df["nameDest"].str.startswith("M").astype(int)

    # 2. Balance discrepancies
    df["balanceDiscrepancy_Orig"] = (df["oldbalanceOrg"] - df["amount"]) - df["newbalanceOrig"]
    df["balanceDiscrepancy_Dest"] = (df["oldbalanceDest"] + df["amount"]) - df["newbalanceDest"]

    # 3. isEmpty_Orig: was the sender account drained to zero?
    df["isEmpty_Orig"] = (df["newbalanceOrig"] == 0).astype(int)

    # 4. amount_to_balance_ratio: relative transaction size
    df["amount_to_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)

    # 5. Filter to TRANSFER and CASH_OUT (fraud only occurs here)
    df_filtered = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()
    print(f"Rows after filtering to TRANSFER & CASH_OUT: {len(df_filtered)}")
    print(f"Fraud cases in filtered set: {df_filtered['isFraud'].sum()}")

    fraud_rate = df_filtered["isFraud"].mean()
    print(f"Fraud rate (filtered): {fraud_rate*100:.4f}%\n")

    # 6. Label-encode 'type'
    le = LabelEncoder()
    df_filtered["type"] = le.fit_transform(df_filtered["type"])
    print(f"Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # 7. Drop high-cardinality strings and naive flag
    df_filtered.drop(columns=["nameOrig", "nameDest", "isFlaggedFraud"], inplace=True)

    # 8. Separate target label and feature matrix
    y_true = df_filtered["isFraud"].values
    feature_cols = [c for c in df_filtered.columns if c != "isFraud" and c != "iso_risk_score" and c != "iso_prediction" and c != "risk_tier"]
    X = df_filtered[feature_cols].copy()
    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}\n")

    print("=" * 60)
    print("STEP 3: STANDARD SCALING (Z-score normalisation)")
    print("=" * 60)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sanity = pd.DataFrame({
        "feature": feature_cols,
        "mean":    np.mean(X_scaled, axis=0).round(6),
        "std":     np.std(X_scaled, axis=0).round(6),
    })
    print(sanity.to_string(index=False))
    print()

    return X_scaled, df_filtered, y_true, fraud_rate
