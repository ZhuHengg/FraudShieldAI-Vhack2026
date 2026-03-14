import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

def calculate_recipient_risk(df_train: pd.DataFrame):
    """
    Computes risk profile scores for recipients based on training data only.
    """
    print("="*50)
    print("STEP 3 — RECIPIENT RISK SCORE FROM TRAIN ONLY")
    print("="*50)

    recipient_stats = df_train.groupby('name_recipient').agg(
        unique_sender_count   = ('name_sender',     'nunique'),
        avg_received_amount   = ('amount',          'mean'),
        min_account_age       = ('account_age_days','min'),
        avg_incoming_velocity = ('tx_count_24h',    'mean'),
        total_received        = ('amount',          'count')
    ).reset_index()

    signal_scaler = MinMaxScaler()
    signals = ['unique_sender_count', 'avg_received_amount', 'avg_incoming_velocity', 'total_received']
    recipient_stats[signals] = signal_scaler.fit_transform(recipient_stats[signals])
    
    recipient_stats['age_risk'] = np.where(
        recipient_stats['min_account_age'] < 90,
        1 - (recipient_stats['min_account_age'] / 90),
        0.0
    )
    
    recipient_stats['recipient_risk_profile_score'] = (
        recipient_stats['unique_sender_count']   * 0.35 +
        recipient_stats['avg_received_amount']   * 0.25 +
        recipient_stats['age_risk']              * 0.20 +
        recipient_stats['avg_incoming_velocity'] * 0.15 +
        recipient_stats['total_received']        * 0.05
    ).round(3).clip(0.0, 1.0)

    # Merchants get 0.0
    merchant_mask = recipient_stats['name_recipient'].str.startswith('M')
    recipient_stats.loc[merchant_mask, 'recipient_risk_profile_score'] = 0.00

    lookup = dict(zip(recipient_stats['name_recipient'], recipient_stats['recipient_risk_profile_score']))
    print(f"Recipient risk lookup built for {len(lookup)} unique recipients.")
    return lookup

def preprocess_features(df_train: pd.DataFrame, df_test: pd.DataFrame = None):
    """
    Applies feature engineering, drops irrelevant columns, and scales the features.
    If df_test is provided, it fits on train and transforms both.
    
    Returns:
        X_train, X_test (if provided)
        y_train, y_test (if provided)
        scaler
        all_features
    """
    print("=" * 60)
    print("STEP 6: CLIPPING & SCALING")
    print("=" * 60)

    DROP_COLUMNS = [
        'transaction_id', 'name_sender', 'name_recipient', 'amount',
        'avg_transaction_amount_30d', 'is_fraud', 'transfer_type'
    ]

    # Handle Encoding
    le = LabelEncoder()
    df_train['transfer_type_encoded'] = le.fit_transform(df_train['transfer_type'])
    if df_test is not None:
        df_test['transfer_type_encoded'] = le.transform(df_test['transfer_type'])

    # Feature Interactions
    for d in [df_train, df_test]:
        if d is not None:
            # Action 2 & 3: Replace raw country_mismatch with interaction version
            d['country_mismatch_suspicious'] = (
                (d['country_mismatch'] == 1) &
                (
                    (d['ip_risk_score'] > 0.50)        |  # suspicious IP
                    (d['session_duration_seconds'] < 30)|  # very fast session
                    (d['account_age_days'] < 30)        |  # very new account
                    (d['is_new_device'] == 1)              # unrecognised device
                )
            ).astype(int)

    binary_features = [
        'transfer_type_encoded', 'is_weekend', 'sender_account_fully_drained',
        'is_new_device', 'is_proxy_ip', 'country_mismatch_suspicious',
        'established_user_new_recipient'
    ]

    continuous_features = [
        'amount_vs_avg_ratio', 'transaction_hour', 'session_duration_seconds',
        'failed_login_attempts', 'ip_risk_score', 'account_age_days',
        'tx_count_24h', 'recipient_risk_profile_score'
    ]
    all_features = binary_features + continuous_features

    # Clipping and Scaling
    scaler = StandardScaler()
    for col in continuous_features:
        lower = df_train[col].quantile(0.01)
        upper = df_train[col].quantile(0.99)
        df_train[col] = df_train[col].clip(lower=lower, upper=upper)
        if df_test is not None:
            df_test[col] = df_test[col].clip(lower=lower, upper=upper)

    df_train[continuous_features] = scaler.fit_transform(df_train[continuous_features])
    X_train = df_train[all_features].values
    y_train = df_train['is_fraud'].values

    if df_test is not None:
        df_test[continuous_features] = scaler.transform(df_test[continuous_features])
        X_test = df_test[all_features].values
        y_test = df_test['is_fraud'].values
        print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
        return X_train, X_test, y_train, y_test, scaler, all_features
    
    return X_train, y_train, scaler, all_features
