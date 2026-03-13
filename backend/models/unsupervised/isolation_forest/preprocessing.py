import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_features(df: pd.DataFrame):
    """
    Applies feature engineering, drops irrelevant columns, and scales the features.
    
    Returns:
        X (np.ndarray): The scaled feature matrix ready for Isolation Forest.
        df_model (pd.DataFrame): The filtered dataframe containing the features and encoded types.
        y_true (np.ndarray): The ground truth label array.
        all_features (list): List of feature names in order.
        scaler: The fitted StandardScaler object.
    """
    print("=" * 60)
    print("STEP 2 & 3: PREPROCESSING & SCALING")
    print("=" * 60)

    # Drop identifier columns — unique strings with no signal
    # Drop raw amount columns — summarised by amount_vs_avg_ratio
    # Drop is_fraud — target label, never feed into unsupervised model
    # Drop transfer_type — will encode separately below
    DROP_COLUMNS = [
        'transaction_id',
        'name_sender',
        'name_recipient',
        'amount',
        'avg_transaction_amount_30d',
        'is_fraud',
        'transfer_type'
    ]

    # Store labels separately before dropping
    y_true = df['is_fraud'].values

    # Store transfer_type before dropping for encoding
    transfer_type = df['transfer_type'].copy()

    df_model = df.drop(columns=DROP_COLUMNS)
    print(f"Shape after dropping: {df_model.shape}")

    # Label encode transfer_type
    le = LabelEncoder()
    df_model['transfer_type_encoded'] = le.fit_transform(transfer_type)
    print(f"Transfer type encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Binary features — leave as 0/1
    binary_features = [
        'transfer_type_encoded',
        'is_weekend',
        'sender_account_fully_drained',
        'is_new_device',
        'is_proxy_ip',
        'country_mismatch',
        'is_new_recipient',
        'established_user_new_recipient'
    ]

    # Continuous features — must scale
    continuous_features = [
        'amount_vs_avg_ratio',
        'transaction_hour',
        'session_duration_seconds',
        'failed_login_attempts',
        'ip_risk_score',
        'account_age_days',
        'tx_count_24h',
        'recipient_risk_profile_score'
    ]

    print(f"Binary features ({len(binary_features)}): {binary_features}")
    print(f"Continuous features ({len(continuous_features)}): {continuous_features}")
    print(f"Total features: {len(binary_features) + len(continuous_features)}")

    print("\nClipping and scaling continuous features...")

    # Clip at 1st and 99th percentile before scaling
    for col in continuous_features:
        lower = df_model[col].quantile(0.01)
        upper = df_model[col].quantile(0.99)
        df_model[col] = df_model[col].clip(lower=lower, upper=upper)

    # Scale continuous features only
    scaler = StandardScaler()
    df_model[continuous_features] = scaler.fit_transform(df_model[continuous_features])

    all_features = binary_features + continuous_features
    X = df_model[all_features].values

    print(f"Final feature matrix shape: {X.shape}\n")

    return X, df_model, y_true, all_features, scaler
