import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE

def calculate_recipient_risk(df_train: pd.DataFrame):
    """
    Computes risk profile scores for recipients based on training data only.
    Identical logic to isolation_forest/preprocessing.py
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
    signals = [
        'unique_sender_count', 'avg_received_amount',
        'avg_incoming_velocity', 'total_received'
    ]
    recipient_stats[signals] = signal_scaler.fit_transform(
        recipient_stats[signals]
    )

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

    # Merchants always get 0.0
    merchant_mask = recipient_stats['name_recipient'].str.startswith('M')
    recipient_stats.loc[merchant_mask, 'recipient_risk_profile_score'] = 0.00

    lookup = dict(zip(
        recipient_stats['name_recipient'],
        recipient_stats['recipient_risk_profile_score']
    ))
    print(f"Recipient risk lookup built for {len(lookup):,} unique recipients.")
    return lookup


def preprocess_features(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame = None
):
    """
    Applies feature engineering, logs amount columns, drops irrelevant columns,
    and scales features ensuring precise ordering for SMOTE.
    """
    print("=" * 60)
    print("STEP 4: FEATURE ENGINEERING, SCALING & SMOTE")
    print("=" * 60)

    splits = {
        'train': df_train,
        'val':   df_val,
        'test':  df_test
    }
    active_splits = {
        k: v for k, v in splits.items() if v is not None
    }

    # ── LABEL ENCODE transfer_type ────────────────────────────────
    le = LabelEncoder()
    df_train['transfer_type_encoded'] = le.fit_transform(
        df_train['transfer_type']
    )
    for name, split in active_splits.items():
        if name != 'train':
            split['transfer_type_encoded'] = le.transform(
                split['transfer_type']
            )
    print(f"Transfer type encoding: "
          f"{dict(zip(le.classes_, le.transform(le.classes_)))}")

    # ── COUNTRY MISMATCH INTERACTION ──────────────────────────────
    for name, split in active_splits.items():
        split['country_mismatch_suspicious'] = (
            (split['country_mismatch'] == 1) &
            (
                (split['ip_risk_score'] > 0.50)         |
                (split['session_duration_seconds'] < 30) |
                (split['account_age_days'] < 30)         |
                (split['is_new_device'] == 1)
            )
        ).astype(int)

    # ── LIGHTGBM ADDED PRE-DROP LOGIC (Logs before dropping) ──────
    for name, split in active_splits.items():
        split['log_amount'] = np.log1p(split['amount'])
        split['log_avg_30d'] = np.log1p(split['avg_transaction_amount_30d'])

    # ── FEATURE DEFINITIONS ───────────────────────────────────────
    binary_features = [
        'transfer_type_encoded',
        'is_weekend',
        'sender_account_fully_drained',
        'is_new_device',
        'is_proxy_ip',
        'country_mismatch_suspicious', 
        'established_user_new_recipient'
    ]

    continuous_features = [
        'log_amount',                      # added for LGBM
        'log_avg_30d',                     # added for LGBM
        'amount_vs_avg_ratio',
        'transaction_hour',
        'session_duration_seconds',
        'failed_login_attempts',
        'ip_risk_score',
        'account_age_days',
        'tx_count_24h',
        'recipient_risk_profile_score'
    ]

    all_features = binary_features + continuous_features
    print(f"\nBinary features ({len(binary_features)}):     {binary_features}")
    print(f"Continuous features ({len(continuous_features)}): {continuous_features}")
    print(f"Total features: {len(all_features)}")

    # ── CLIP OUTLIERS ─────────────────────────────────────────────
    clip_bounds = {}
    for col in continuous_features:
        lower = df_train[col].quantile(0.01)
        upper = df_train[col].quantile(0.99)
        clip_bounds[col] = (lower, upper)

    for col in continuous_features:
        lower, upper = clip_bounds[col]
        for name, split in active_splits.items():
            split[col] = split[col].clip(lower=lower, upper=upper)

    # ── PREPARE DATA MATRICES ─────────────────────────────────────
    X_train_raw = df_train[all_features].values
    y_train = df_train['is_fraud'].values

    X_val_raw   = df_val[all_features].values
    y_val   = df_val['is_fraud'].values

    X_test_raw = df_test[all_features].values if df_test is not None else None
    y_test = df_test['is_fraud'].values if df_test is not None else None

    # CRITICAL ORDER 1: SCALE ON ORIGINAL X_TRAIN (NO SMOTE YET)
    scaler = StandardScaler()
    X_train_scaled = X_train_raw.copy()
    X_train_scaled[:, len(binary_features):] = scaler.fit_transform(
        X_train_raw[:, len(binary_features):]
    )
    
    # CRITICAL ORDER 2: SMOTE ON SCALED TRAIN DATA
    print("\nApplying SMOTE on TRAIN ONLY...")
    smote = SMOTE(sampling_strategy=0.10, random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    print(f"Train before SMOTE: {X_train_scaled.shape} - Fraud rate: {y_train.mean():.4f}")
    print(f"Train after SMOTE:  {X_train_res.shape} - Fraud rate: {y_train_res.mean():.4f}")

    # CRITICAL ORDER 3: TRANSFORM VAL AND TEST
    X_val = X_val_raw.copy()
    X_val[:, len(binary_features):] = scaler.transform(X_val_raw[:, len(binary_features):])

    X_test = None
    if X_test_raw is not None:
         X_test = X_test_raw.copy()
         X_test[:, len(binary_features):] = scaler.transform(X_test_raw[:, len(binary_features):])


    # ── SANITY CHECK ──────────────────────────────────────────────
    print("\n=== SCALING SANITY CHECK ===")
    print("Train means (expect ~0 pre-SMOTE):")
    print(np.mean(X_train_scaled[:, len(binary_features):], axis=0).round(4).tolist())
    
    if df_test is not None:
         return X_train_res, X_val, X_test, y_train_res, y_val, y_test, scaler, all_features
    return X_train_res, X_val, y_train_res, y_val, scaler, all_features
