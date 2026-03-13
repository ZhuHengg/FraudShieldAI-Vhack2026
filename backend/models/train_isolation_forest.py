import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import time

np.random.seed(42)

DATA_FILE    = '../data/raw/ewallet_transaction.csv'
OUTPUT_MODEL = 'outputs/model/isolation_forest_model.pkl'
OUTPUT_SCALER = 'outputs/model/scaler.pkl'
OUTPUT_MMS    = 'outputs/model/minmax_scaler.pkl'

os.makedirs('outputs/model',   exist_ok=True)
os.makedirs('outputs/plots',   exist_ok=True)
os.makedirs('outputs/results', exist_ok=True)

print("Loading dataset...")
start = time.time()

df = pd.read_csv(DATA_FILE)

print(f"Shape: {df.shape}")
print(f"Fraud cases: {df['is_fraud'].sum():,}")
print(f"Legit cases: {(df['is_fraud']==0).sum():,}")
print(f"Fraud rate: {df['is_fraud'].mean():.4f}")
print(f"Loaded in {time.time()-start:.1f}s")


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
print(f"\nShape after dropping: {df_model.shape}")
print(f"Remaining columns: {df_model.columns.tolist()}")


# Label encode transfer_type
# CASH_OUT = 0, TRANSFER = 1
le = LabelEncoder()
df_model['transfer_type_encoded'] = le.fit_transform(transfer_type)

print(f"\nTransfer type encoding:")
print(dict(zip(le.classes_, le.transform(le.classes_))))


# Binary features — leave as 0/1
# Scaling destroys the meaning of binary flags
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
# Without scaling, account_age_days (1-5000) dominates
# over ip_risk_score (0-1) in random splits
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

print(f"\nBinary features ({len(binary_features)}): {binary_features}")
print(f"Continuous features ({len(continuous_features)}): {continuous_features}")
print(f"Total features: {len(binary_features) + len(continuous_features)}")


print("\nClipping and scaling continuous features...")

# Clip at 1st and 99th percentile before scaling
# Prevents extreme outliers from dominating StandardScaler
for col in continuous_features:
    lower = df_model[col].quantile(0.01)
    upper = df_model[col].quantile(0.99)
    df_model[col] = df_model[col].clip(lower=lower, upper=upper)
    print(f"  {col}: clipped to [{lower:.3f}, {upper:.3f}]")

# Scale continuous features only
scaler = StandardScaler()
df_model[continuous_features] = scaler.fit_transform(
    df_model[continuous_features]
)

# Sanity check — means must be ~0, stds must be ~1
print("\n=== SCALING SANITY CHECK ===")
print("Continuous feature means (expect ~0):")
print(df_model[continuous_features].mean().round(4))
print("\nContinuous feature stds (expect ~1):")
print(df_model[continuous_features].std().round(4))
print("\nBinary feature ranges (expect min=0, max=1 only):")
print(df_model[binary_features].describe().loc[['min', 'max']])


all_features = binary_features + continuous_features
X = df_model[all_features].values

print(f"\nFinal feature matrix shape: {X.shape}")
print(f"Features in order: {all_features}")


# Use actual fraud rate from dataset
# Multiply by aggressive factor to improve recall
# Isolation Forest needs slightly higher contamination
# than true rate to catch enough fraud cases
fraud_rate       = y_true.mean()
contamination    = min(fraud_rate * 5, 0.10)

print(f"\nTrue fraud rate:      {fraud_rate:.4f}")
print(f"Contamination used:   {contamination:.4f}")
print(f"Aggressive factor:    5x")


print("\nTraining Isolation Forest...")
start = time.time()

iso_forest = IsolationForest(
    n_estimators=200,       # enough trees for stable anomaly scores
    contamination=contamination,
    max_samples=512,        # subsample size per tree
                            # smaller = faster, still effective at 2M rows
    max_features=0.8,       # use 80% of features per tree
                            # prevents over-reliance on single features
    random_state=42,
    n_jobs=-1               # use all CPU cores
)

iso_forest.fit(X)
print(f"Training complete in {time.time()-start:.1f}s")


print("\nGenerating predictions and risk scores...")

# Raw predictions: -1 = anomaly (fraud), 1 = normal (legitimate)
# Convert to: 1 = fraud, 0 = legitimate
raw_predictions  = iso_forest.predict(X)
y_pred           = np.where(raw_predictions == -1, 1, 0)

# Anomaly score: lower = more anomalous
# Invert so that higher score = more suspicious
anomaly_scores   = iso_forest.decision_function(X)
risk_scores_raw  = -anomaly_scores

# Normalize to 0-100 range for interpretability
mms = MinMaxScaler(feature_range=(0, 100))
risk_scores = mms.fit_transform(
    risk_scores_raw.reshape(-1, 1)
).flatten()

print(f"Risk score range: {risk_scores.min():.2f} to {risk_scores.max():.2f}")
print(f"Risk score mean:  {risk_scores.mean():.2f}")


# Percentile-based thresholds — adapts to actual score distribution
# Better than fixed values which assume balanced distribution
approve_threshold = np.percentile(risk_scores, 85)
flag_threshold    = np.percentile(risk_scores, 95)

print(f"\nThresholds:")
print(f"  Approve -> below {approve_threshold:.2f}")
print(f"  Flag    -> {approve_threshold:.2f} to {flag_threshold:.2f}")
print(f"  Block   -> above {flag_threshold:.2f}")

def assign_tier(score):
    if score <= approve_threshold:
        return 'Approve'
    elif score <= flag_threshold:
        return 'Flag'
    else:
        return 'Block'

df_results = df[['transaction_id', 'is_fraud']].copy()
df_results['iso_risk_score'] = risk_scores
df_results['iso_prediction'] = y_pred
df_results['risk_tier']      = df_results['iso_risk_score'].apply(assign_tier)

# Risk tier summary
print("\n=== RISK TIER SUMMARY ===")
tier_summary = df_results.groupby('risk_tier').agg(
    transaction_count = ('is_fraud', 'count'),
    fraud_count       = ('is_fraud', 'sum'),
    fraud_rate        = ('is_fraud', 'mean')
).round(4)
print(tier_summary)

# Save the models (joblib)
joblib.dump(iso_forest, OUTPUT_MODEL)
joblib.dump(scaler, OUTPUT_SCALER)
joblib.dump(mms, OUTPUT_MMS)
print(f"Models saved to {os.path.dirname(OUTPUT_MODEL)}")
