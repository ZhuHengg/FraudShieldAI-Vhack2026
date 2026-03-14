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
    df_val: pd.DataFrame,       # CHANGED: added df_val parameter
    df_test: pd.DataFrame = None
):
    """
    Applies feature engineering, drops irrelevant columns, and scales features.
    Fits all transformers on train only.
    Transforms train, val, and test using train-fitted transformers.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, scaler, all_features
    """
    print("=" * 60)
    print("STEP 4: FEATURE ENGINEERING, CLIPPING & SCALING")
    print("=" * 60)

    # CHANGED: include df_val in all operations
    # all splits processed together to avoid code duplication
    splits = {
        'train': df_train,
        'val':   df_val,
        'test':  df_test
    }
    active_splits = {
        k: v for k, v in splits.items() if v is not None
    }

    # ── LABEL ENCODE transfer_type ────────────────────────────────
    # Fit on train only, transform all splits
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
    # Compute on raw unscaled values before any scaling
    # Replace raw country_mismatch — too noisy for ASEAN migrant workers
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

    # ── FEATURE DEFINITIONS ───────────────────────────────────────
    # is_new_recipient DROPPED — 0.94 correlation with
    # established_user_new_recipient, dominated permutation importance
    binary_features = [
        'transfer_type_encoded',
        'is_weekend',
        'sender_account_fully_drained',
        'is_new_device',
        'is_proxy_ip',
        'country_mismatch_suspicious',   # replaces raw country_mismatch
        'established_user_new_recipient'
        # 'is_new_recipient' DROPPED
    ]

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

    all_features = binary_features + continuous_features
    print(f"\nBinary features ({len(binary_features)}):     {binary_features}")
    print(f"Continuous features ({len(continuous_features)}): {continuous_features}")
    print(f"Total features: {len(all_features)}")

    # ── CLIP OUTLIERS ─────────────────────────────────────────────
    # Clip bounds computed from TRAIN only
    # Applied to all splits using train bounds
    clip_bounds = {}
    for col in continuous_features:
        lower = df_train[col].quantile(0.01)
        upper = df_train[col].quantile(0.99)
        clip_bounds[col] = (lower, upper)

    for col in continuous_features:
        lower, upper = clip_bounds[col]
        for name, split in active_splits.items():
            split[col] = split[col].clip(lower=lower, upper=upper)

    # ── SCALE CONTINUOUS FEATURES ─────────────────────────────────
    # Fit StandardScaler on TRAIN only
    # Transform all splits using train-fitted scaler
    scaler = StandardScaler()
    df_train[continuous_features] = scaler.fit_transform(
        df_train[continuous_features]
    )
    for name, split in active_splits.items():
        if name != 'train':
            split[continuous_features] = scaler.transform(
                split[continuous_features]
            )

    # ── SANITY CHECK ──────────────────────────────────────────────
    print("\n=== SCALING SANITY CHECK ===")
    print("Train means (expect ~0):")
    print(df_train[continuous_features].mean().round(4).to_dict())
    print("Train stds (expect ~1):")
    print(df_train[continuous_features].std().round(4).to_dict())
    print("Val means (expect close to 0):")
    print(df_val[continuous_features].mean().round(4).to_dict())
    print("Binary ranges (expect min=0, max=1):")
    print(df_train[binary_features].describe().loc[['min','max']].to_dict())

    # ── BUILD FEATURE MATRICES ────────────────────────────────────
    X_train = df_train[all_features].values
    y_train = df_train['is_fraud'].values

    # CHANGED: always build X_val y_val — val is always required now
    X_val   = df_val[all_features].values
    y_val   = df_val['is_fraud'].values

    print(f"\nX_train: {X_train.shape}")
    print(f"X_val:   {X_val.shape}")

    if df_test is not None:
        X_test = df_test[all_features].values
        y_test = df_test['is_fraud'].values
        print(f"X_test:  {X_test.shape}")
        # CHANGED: returns X_val and y_val between train and test
        return X_train, X_val, X_test, y_train, y_val, y_test, scaler, all_features

    # CHANGED: returns X_val y_val even when no test set provided
    return X_train, X_val, y_train, y_val, scaler, all_features