"""
Synthetic ASEAN Mobile E-Wallet Fraud Detection Dataset Generator
=================================================================
Generates:
  - data/accounts.csv           (account registry for debugging)
  - data/synthetic_ewallet_fraud.csv  (2M transactions for model training)

Usage:
    python generate_synthetic_data.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import time

np.random.seed(42)

TOTAL_RECORDS = 2_000_000
N_FRAUD       = 30_000
N_LEGIT       = 1_970_000
RANDOM_SEED   = 42

os.makedirs('data', exist_ok=True)
ACCOUNTS_FILE     = 'data/accounts.csv'
TRANSACTIONS_FILE = 'data/synthetic_ewallet_fraud.csv'

# ═══════════════════════════════════════════════════════════════════
# STEP 1 — GENERATE ACCOUNT POOLS AND BUILD ACCOUNTS REGISTRY
# ═══════════════════════════════════════════════════════════════════

print("Generating account pools...")
start = time.time()

used_ids          = set()
accounts_registry = []


def generate_pool(prefix, size, used_ids):
    pool = []
    while len(pool) < size:
        batch = [
            f"{prefix}{np.random.randint(100000000, 999999999)}"
            for _ in range(size * 2)
        ]
        for acc_id in batch:
            if acc_id not in used_ids:
                used_ids.add(acc_id)
                pool.append(acc_id)
            if len(pool) == size:
                break
    return pool


def register_accounts(pool, account_type, archetype, pool_name,
                      age_range, amt_min, amt_max):
    rep_age = int((age_range[0] + age_range[1]) / 2)
    for acc_id in pool:
        accounts_registry.append({
            'account_id':     acc_id,
            'account_type':   account_type,
            'archetype':      archetype,
            'pool':           pool_name,
            'account_age_days_representative': rep_age,
            'avg_amount_range_min': amt_min,
            'avg_amount_range_max': amt_max,
            'recipient_risk_profile_score': None
        })


# ── FRAUD POOLS ───────────────────────────────────────────────────
FRAUD_SENDER_POOL = generate_pool('C', 500, used_ids)
register_accounts(
    FRAUD_SENDER_POOL, 'fraudster', 'mixed',
    'fraud_sender', (1, 2000), 0, 1000000
)

MULE_POOL      = generate_pool('C', 150, used_ids)
HEAVY_MULES    = MULE_POOL[:30]
REGULAR_MULES  = MULE_POOL[30:]

register_accounts(
    HEAVY_MULES, 'mule', 'money_mule',
    'heavy_mule', (30, 180), 10000, 1000000
)
register_accounts(
    REGULAR_MULES, 'mule', 'money_mule',
    'regular_mule', (30, 180), 10000, 1000000
)

# ── LEGITIMATE POOLS ──────────────────────────────────────────────
LEGIT_SENDER_POOL = generate_pool('C', 500_000, used_ids)
chunk = len(LEGIT_SENDER_POOL)
c1 = int(chunk * 0.35)
c2 = int(chunk * 0.25)
c3 = int(chunk * 0.20)
c4 = chunk - c1 - c2 - c3

register_accounts(LEGIT_SENDER_POOL[:c1],
    'legitimate', 'low_income_user', 'legit_sender',
    (180, 1825), 50, 500)
register_accounts(LEGIT_SENDER_POOL[c1:c1+c2],
    'legitimate', 'migrant_worker', 'legit_sender',
    (90, 730), 200, 2000)
register_accounts(LEGIT_SENDER_POOL[c1+c2:c1+c2+c3],
    'legitimate', 'sme_merchant', 'legit_sender',
    (365, 3650), 1000, 50000)
register_accounts(LEGIT_SENDER_POOL[c1+c2+c3:],
    'legitimate', 'elderly_user', 'legit_sender',
    (730, 5000), 50, 800)

LEGIT_RECIPIENT_POOL = generate_pool('C', 500_000, used_ids)
register_accounts(
    LEGIT_RECIPIENT_POOL, 'legitimate', 'mixed',
    'legit_recipient', (90, 3650), 50, 50000
)

MERCHANT_POOL = generate_pool('M', 50_000, used_ids)
register_accounts(
    MERCHANT_POOL, 'legitimate', 'merchant',
    'merchant', (365, 5000), 500, 100000
)

# High volume merchant pool — small number of accounts
# that receive from many different customers legitimately
# These will naturally score high on unique_sender_count
# creating realistic false positives in the score distribution
HIGH_VOLUME_MERCHANT_POOL = MERCHANT_POOL[:500]
# First 500 merchants are designated high volume
# Same accounts, just flagged for routing logic in Step 5

print(f"Account pools generated in {time.time()-start:.1f}s")
print(f"Total unique accounts: {len(used_ids):,}")
print(f"Accounts registry entries: {len(accounts_registry):,}")


# ═══════════════════════════════════════════════════════════════════
# STEP 2 — DEFINE ARCHETYPES
# ═══════════════════════════════════════════════════════════════════

LEGIT_ARCHETYPES = {

    'low_income_user': {
        'weight': 0.35,
        'account_age_days':        (180, 1825),
        'avg_amount_range':        (50, 500),
        'amount_multiplier':       (0.3, 2.0),
        'tx_count_24h_range':      (1, 2),
        'country_mismatch_prob':   0.05,
        'is_new_device_prob':      0.05,
        'is_proxy_ip_prob':        0.02,
        'session_range':           (90, 600),
        'failed_login_weights':    [0.85, 0.12, 0.02, 0.01],
        'drain_prob':              0.02,
        'transfer_type_weights':   [0.50, 0.50],
        'night_transaction_prob':  0.05,
    },

    'migrant_worker': {
        'weight': 0.25,
        'account_age_days':        (90, 730),
        'avg_amount_range':        (200, 2000),
        'amount_multiplier':       (0.5, 3.0),
        'tx_count_24h_range':      (1, 3),
        'country_mismatch_prob':   0.70,
        'is_new_device_prob':      0.15,
        'is_proxy_ip_prob':        0.05,
        'session_range':           (60, 400),
        'failed_login_weights':    [0.80, 0.14, 0.04, 0.02],
        'drain_prob':              0.03,
        'transfer_type_weights':   [0.70, 0.30],
        'night_transaction_prob':  0.08,
    },

    'sme_merchant': {
        'weight': 0.20,
        'account_age_days':        (365, 3650),
        'avg_amount_range':        (1000, 50000),
        'amount_multiplier':       (0.4, 2.0),
        'tx_count_24h_range':      (3, 10),
        'country_mismatch_prob':   0.10,
        'is_new_device_prob':      0.03,
        'is_proxy_ip_prob':        0.03,
        'session_range':           (120, 480),
        'failed_login_weights':    [0.90, 0.08, 0.015, 0.005],
        'drain_prob':              0.01,
        'transfer_type_weights':   [0.45, 0.55],
        'night_transaction_prob':  0.03,
    },

    'elderly_user': {
        'weight': 0.20,
        'account_age_days':        (730, 5000),
        'avg_amount_range':        (50, 800),
        'amount_multiplier':       (0.3, 1.5),
        'tx_count_24h_range':      (1, 2),
        'country_mismatch_prob':   0.02,
        'is_new_device_prob':      0.25,
        'is_proxy_ip_prob':        0.01,
        'session_range':           (180, 900),
        'failed_login_weights':    [0.65, 0.20, 0.10, 0.05],
        'drain_prob':              0.02,
        'transfer_type_weights':   [0.40, 0.60],
        'night_transaction_prob':  0.03,
    },
}

FRAUD_ARCHETYPES = {

    'account_takeover': {
        'weight': 0.45,
        'account_age_days':        (365, 2000),
        'avg_amount_range':        (5000, 500000),
        'amount_multiplier':       (0.8, 1.0),
        'tx_count_24h_range':      (3, 10),
        'country_mismatch_prob':   0.80,
        'is_new_device_prob':      0.95,
        'is_proxy_ip_prob':        0.80,
        'session_range':           (3, 20),
        'failed_login_weights':    [0.10, 0.20, 0.30, 0.40],
        'drain_prob':              0.85,
        'transfer_type_weights':   [0.55, 0.45],
        'night_transaction_prob':  0.65,
    },

    'new_account_fraud': {
        'weight': 0.30,
        'account_age_days':        (1, 30),
        'avg_amount_range':        (1000, 100000),
        'amount_multiplier':       (0.8, 5.0),
        'tx_count_24h_range':      (1, 4),
        'country_mismatch_prob':   0.75,
        'is_new_device_prob':      0.92,
        'is_proxy_ip_prob':        0.70,
        'session_range':           (5, 45),
        'failed_login_weights':    [0.60, 0.25, 0.10, 0.05],
        'drain_prob':              0.60,
        'transfer_type_weights':   [0.40, 0.60],
        'night_transaction_prob':  0.45,
    },

    'money_mule': {
        'weight': 0.25,
        'account_age_days':        (30, 180),
        'avg_amount_range':        (10000, 1000000),
        'amount_multiplier':       (0.5, 0.9),
        'tx_count_24h_range':      (2, 8),
        'country_mismatch_prob':   0.55,
        'is_new_device_prob':      0.60,
        'is_proxy_ip_prob':        0.55,
        'session_range':           (10, 60),
        'failed_login_weights':    [0.70, 0.20, 0.07, 0.03],
        'drain_prob':              0.30,
        'transfer_type_weights':   [0.35, 0.65],
        'night_transaction_prob':  0.35,
    },
}


# ═══════════════════════════════════════════════════════════════════
# STEP 3 — IP RISK SCORE FUNCTION
# ═══════════════════════════════════════════════════════════════════

def generate_ip_risk_score(is_fraud, is_proxy):
    if is_fraud:
        tier = np.random.choice(
            ['vpn', 'known_bad', 'residential'],
            p=[0.45, 0.35, 0.20]
        )
        if tier == 'vpn':
            return round(np.random.uniform(0.50, 0.75), 3)
        elif tier == 'known_bad':
            return round(np.random.uniform(0.75, 1.00), 3)
        else:
            return round(np.random.uniform(0.20, 0.50), 3)
    else:
        if is_proxy:
            return round(np.random.uniform(0.10, 0.35), 3)
        else:
            return round(np.random.uniform(0.00, 0.20), 3)


# ═══════════════════════════════════════════════════════════════════
# STEP 4 — TRANSACTION HOUR FUNCTION
# ═══════════════════════════════════════════════════════════════════

_BUSINESS_HOUR_PROBS_RAW = [
    0.01, 0.01, 0.01, 0.01, 0.01, 0.02,
    0.03, 0.05, 0.07, 0.08, 0.08, 0.08,
    0.08, 0.08, 0.07, 0.07, 0.06, 0.06,
    0.05, 0.04, 0.04, 0.03, 0.02, 0.01
]
# Normalize to sum exactly to 1.0
_s = sum(_BUSINESS_HOUR_PROBS_RAW)
BUSINESS_HOUR_PROBS = [p / _s for p in _BUSINESS_HOUR_PROBS_RAW]
LATE_HOURS = list(range(0, 7)) + list(range(22, 24))
DAY_HOURS  = list(range(7, 22))


def generate_transaction_hour(is_fraud, archetype):
    night_prob = archetype['night_transaction_prob']
    if is_fraud:
        if np.random.random() < night_prob:
            return int(np.random.choice(LATE_HOURS))
        else:
            return int(np.random.choice(DAY_HOURS))
    else:
        return int(np.random.choice(range(24), p=BUSINESS_HOUR_PROBS))


# ═══════════════════════════════════════════════════════════════════
# STEP 5 — CORE RECORD GENERATION FUNCTION
# ═══════════════════════════════════════════════════════════════════

def generate_record(idx, is_fraud, archetype):

    tx = {}
    tx['transaction_id'] = f'TXN_{idx:08d}'
    tx['is_fraud']       = int(is_fraud)

    # ── SENDER AND RECIPIENT ──────────────────────────────────────
    if is_fraud:
        tx['name_sender'] = FRAUD_SENDER_POOL[
            np.random.randint(0, len(FRAUD_SENDER_POOL))
        ]
        if np.random.random() < 0.60:
            tx['name_recipient'] = HEAVY_MULES[
                np.random.randint(0, len(HEAVY_MULES))
            ]
        else:
            tx['name_recipient'] = REGULAR_MULES[
                np.random.randint(0, len(REGULAR_MULES))
            ]
    else:
        tx['name_sender'] = LEGIT_SENDER_POOL[
            np.random.randint(0, len(LEGIT_SENDER_POOL))
        ]

        routing = np.random.random()

        if routing < 0.10:
            # FIX 1: 10% of legitimate transactions route to mule accounts
            # Mules receive legitimate transactions to appear normal
            # This prevents perfect score separation at 0.40
            if np.random.random() < 0.60:
                tx['name_recipient'] = HEAVY_MULES[
                    np.random.randint(0, len(HEAVY_MULES))
                ]
            else:
                tx['name_recipient'] = REGULAR_MULES[
                    np.random.randint(0, len(REGULAR_MULES))
                ]

        elif routing < 0.15:
            # FIX 2: 5% of legitimate transactions route to high volume merchants
            # These merchants receive from many different customers legitimately
            # High unique_sender_count pushes their score above 0.40
            tx['name_recipient'] = HIGH_VOLUME_MERCHANT_POOL[
                np.random.randint(0, len(HIGH_VOLUME_MERCHANT_POOL))
            ]

        elif routing < 0.30:
            # Remaining merchant transactions — normal volume merchants
            tx['name_recipient'] = MERCHANT_POOL[500:][
                np.random.randint(0, len(MERCHANT_POOL) - 500)
            ]

        else:
            # Standard legitimate recipient
            tx['name_recipient'] = LEGIT_RECIPIENT_POOL[
                np.random.randint(0, len(LEGIT_RECIPIENT_POOL))
            ]

    # ── TRANSFER TYPE ─────────────────────────────────────────────
    tx['transfer_type'] = np.random.choice(
        ['TRANSFER', 'CASH_OUT'],
        p=archetype['transfer_type_weights']
    )

    # ── ACCOUNT AGE ───────────────────────────────────────────────
    tx['account_age_days'] = int(np.random.randint(
        *archetype['account_age_days']
    ))

    # ── AMOUNT CHAIN ──────────────────────────────────────────────
    avg = round(np.random.uniform(*archetype['avg_amount_range']), 2)
    tx['avg_transaction_amount_30d'] = avg

    if is_fraud:
        if np.random.random() < 0.60:
            tx['amount'] = avg
        else:
            multiplier = np.random.uniform(3.0, 10.0)
            tx['amount'] = round(avg * multiplier, 2)
    else:
        multiplier = np.random.uniform(*archetype['amount_multiplier'])
        tx['amount'] = round(avg * multiplier, 2)

    tx['amount_vs_avg_ratio'] = round(
        tx['amount'] / (tx['avg_transaction_amount_30d'] + 1), 3
    )

    # ── TIME ──────────────────────────────────────────────────────
    tx['transaction_hour'] = generate_transaction_hour(is_fraud, archetype)
    tx['is_weekend']       = int(np.random.random() < (
        0.32 if is_fraud else 0.28
    ))

    # ── DRAIN FLAG ────────────────────────────────────────────────
    tx['sender_account_fully_drained'] = int(
        np.random.random() < archetype['drain_prob']
    )

    # ── DEVICE AND SESSION ────────────────────────────────────────
    tx['is_new_device'] = int(
        np.random.random() < archetype['is_new_device_prob']
    )

    if is_fraud:
        tx['session_duration_seconds'] = int(
            np.random.randint(*archetype['session_range'])
        )
    elif tx['is_new_device'] == 1:
        tx['session_duration_seconds'] = int(np.random.randint(120, 900))
    else:
        tx['session_duration_seconds'] = int(
            np.random.randint(*archetype['session_range'])
        )

    tx['failed_login_attempts'] = int(np.random.choice(
        [0, 1, 2, 3],
        p=archetype['failed_login_weights']
    ))

    # ── LOCATION AND IP ───────────────────────────────────────────
    tx['is_proxy_ip'] = int(
        np.random.random() < archetype['is_proxy_ip_prob']
    )
    tx['ip_risk_score'] = generate_ip_risk_score(
        is_fraud, tx['is_proxy_ip']
    )
    tx['country_mismatch'] = int(
        np.random.random() < archetype['country_mismatch_prob']
    )

    # ── VELOCITY ──────────────────────────────────────────────────
    tx['tx_count_24h'] = int(np.random.randint(
        *archetype['tx_count_24h_range']
    ))
    tx['is_new_recipient'] = int(np.random.random() < (
        0.90 if is_fraud else 0.35
    ))

    tx['established_user_new_recipient'] = int(
        tx['account_age_days'] > 90 and tx['is_new_recipient'] == 1
    )
    if is_fraud:
        tx['established_user_new_recipient'] = int(
            tx['is_new_recipient'] == 1
        )

    # recipient_risk_profile_score filled in Step 7
    tx['recipient_risk_profile_score'] = None

    return tx


# ═══════════════════════════════════════════════════════════════════
# STEP 6 — GENERATE ALL RECORDS
# ═══════════════════════════════════════════════════════════════════

print("Generating 2,000,000 records...")
start = time.time()

records = []

legit_archetype_names  = list(LEGIT_ARCHETYPES.keys())
legit_weights          = [a['weight'] for a in LEGIT_ARCHETYPES.values()]
fraud_archetype_names  = list(FRAUD_ARCHETYPES.keys())
fraud_weights          = [a['weight'] for a in FRAUD_ARCHETYPES.values()]

legit_archetype_choices = np.random.choice(
    legit_archetype_names, size=N_LEGIT, p=legit_weights
)
fraud_archetype_choices = np.random.choice(
    fraud_archetype_names, size=N_FRAUD, p=fraud_weights
)

for i in range(N_LEGIT):
    if i % 200_000 == 0:
        print(f"  Legit: {i:,} / {N_LEGIT:,}")
    archetype = LEGIT_ARCHETYPES[legit_archetype_choices[i]]
    records.append(generate_record(i, False, archetype))

for i in range(N_LEGIT, N_LEGIT + N_FRAUD):
    archetype = FRAUD_ARCHETYPES[
        fraud_archetype_choices[i - N_LEGIT]
    ]
    records.append(generate_record(i, True, archetype))

print(f"Records generated in {time.time()-start:.1f}s")

print("Converting to DataFrame...")
df = pd.DataFrame(records)
print(f"DataFrame shape: {df.shape}")


# ═══════════════════════════════════════════════════════════════════
# STEP 7 — COMPUTE RECIPIENT RISK PROFILE SCORE
# ═══════════════════════════════════════════════════════════════════

print("Computing recipient_risk_profile_score...")

recipient_stats = df.groupby('name_recipient').agg(
    unique_sender_count   = ('name_sender',    'nunique'),
    avg_received_amount   = ('amount',         'mean'),
    min_account_age       = ('account_age_days','min'),
    avg_incoming_velocity = ('tx_count_24h',   'mean'),
    total_received        = ('amount',         'count')
).reset_index()

scaler  = MinMaxScaler()
signals = [
    'unique_sender_count',
    'avg_received_amount',
    'avg_incoming_velocity',
    'total_received'
]
recipient_stats[signals] = scaler.fit_transform(
    recipient_stats[signals]
)
recipient_stats['age_risk'] = 1 - MinMaxScaler().fit_transform(
    recipient_stats[['min_account_age']]
)

recipient_stats['recipient_risk_profile_score'] = (
    recipient_stats['unique_sender_count']   * 0.35 +
    recipient_stats['avg_received_amount']   * 0.25 +
    recipient_stats['age_risk']              * 0.20 +
    recipient_stats['avg_incoming_velocity'] * 0.15 +
    recipient_stats['total_received']        * 0.05
).round(3)

# Merchant accounts always score 0.00
merchant_mask = recipient_stats['name_recipient'].str.startswith('M')
recipient_stats.loc[merchant_mask, 'recipient_risk_profile_score'] = 0.00

# FIX 3: Inject gaussian noise to blur the clean boundary
# Without noise: legit max=0.35, fraud min=0.40 — perfect separation
# With noise: overlap zone created between 0.30 and 0.55
# Forces model to use OTHER features alongside the score

np.random.seed(RANDOM_SEED)

# Legitimate recipients get upward noise — some will cross 0.40
legit_recipient_mask = (
    ~recipient_stats['name_recipient'].str.startswith('M') &
    ~recipient_stats['name_recipient'].isin(MULE_POOL)
)
n_legit_recipients = legit_recipient_mask.sum()
legit_noise = np.random.normal(
    loc=0.0,     # centered at zero — symmetric noise
    scale=0.06,  # std of 0.06 pushes ~8% above 0.40
    size=n_legit_recipients
)
recipient_stats.loc[
    legit_recipient_mask, 'recipient_risk_profile_score'
] = (
    recipient_stats.loc[
        legit_recipient_mask, 'recipient_risk_profile_score'
    ] + legit_noise
).clip(0.0, 1.0).round(3)

# Mule accounts get downward noise — some will drop below 0.40
# because they received legitimate transactions in Fix 1
mule_mask = recipient_stats['name_recipient'].isin(MULE_POOL)
n_mules   = mule_mask.sum()
mule_noise = np.random.normal(
    loc=-0.05,   # slight downward bias from legitimate transactions
    scale=0.07,  # std of 0.07 pulls ~15% below 0.40
    size=n_mules
)
recipient_stats.loc[
    mule_mask, 'recipient_risk_profile_score'
] = (
    recipient_stats.loc[
        mule_mask, 'recipient_risk_profile_score'
    ] + mule_noise
).clip(0.0, 1.0).round(3)

# Merchant accounts stay at 0.00 — override any noise applied
recipient_stats.loc[merchant_mask, 'recipient_risk_profile_score'] = 0.00

print("Gaussian noise applied to recipient_risk_profile_score")

# Join back to transactions
df = df.merge(
    recipient_stats[['name_recipient', 'recipient_risk_profile_score']],
    on='name_recipient',
    how='left',
    suffixes=('_old', '')
)
df = df.drop(columns=['recipient_risk_profile_score_old'], errors='ignore')
df['recipient_risk_profile_score'] = df[
    'recipient_risk_profile_score'
].fillna(0.00)

# Join risk score back to accounts registry too
risk_lookup = dict(zip(
    recipient_stats['name_recipient'],
    recipient_stats['recipient_risk_profile_score']
))
for entry in accounts_registry:
    entry['recipient_risk_profile_score'] = risk_lookup.get(
        entry['account_id'], None
    )

print("recipient_risk_profile_score computed and joined")


# ═══════════════════════════════════════════════════════════════════
# STEP 8 — SAVE ACCOUNTS.CSV
# ═══════════════════════════════════════════════════════════════════

print("Saving accounts.csv...")
df_accounts = pd.DataFrame(accounts_registry)

df_accounts = df_accounts[[
    'account_id',
    'account_type',
    'archetype',
    'pool',
    'account_age_days_representative',
    'avg_amount_range_min',
    'avg_amount_range_max',
    'recipient_risk_profile_score'
]]

df_accounts.to_csv(ACCOUNTS_FILE, index=False)
print(f"accounts.csv saved — {len(df_accounts):,} accounts")
print(f"File size: {os.path.getsize(ACCOUNTS_FILE)/1024/1024:.1f} MB")

print("\n=== ACCOUNTS.CSV SANITY CHECK ===")
print(df_accounts['account_type'].value_counts())
print(df_accounts['archetype'].value_counts())
print(df_accounts['pool'].value_counts())

print("\nAvg risk score by pool:")
print(df_accounts.groupby('pool')[
    'recipient_risk_profile_score'
].mean().round(3).sort_values(ascending=False))


# ═══════════════════════════════════════════════════════════════════
# STEP 9 — INJECT NOISE EDGE CASES
# ═══════════════════════════════════════════════════════════════════

print("Injecting noise edge cases...")

n_noise   = int(len(df) * 0.08)
noise_idx = df.sample(n_noise, random_state=42).index
third     = n_noise // 3

# Edge Case 1: Migrant worker false positive
migrant_idx = noise_idx[:third]
legit_mask  = df.loc[migrant_idx, 'is_fraud'] == 0
df.loc[migrant_idx[legit_mask], 'country_mismatch'] = 1
df.loc[migrant_idx[legit_mask], 'is_new_device']    = 1

# Edge Case 2: Sophisticated fraudster with clean IP
clean_fraud_idx = noise_idx[third: 2*third]
fraud_mask      = df.loc[clean_fraud_idx, 'is_fraud'] == 1
n_clean_fraud   = fraud_mask.sum()
df.loc[clean_fraud_idx[fraud_mask], 'ip_risk_score'] = np.round(
    np.random.uniform(0.05, 0.25, n_clean_fraud), 3
)
df.loc[clean_fraud_idx[fraud_mask], 'is_proxy_ip'] = 0

# Edge Case 3: New legitimate user
new_user_idx = noise_idx[2*third:]
legit_mask2  = df.loc[new_user_idx, 'is_fraud'] == 0
n_new_legit  = legit_mask2.sum()
df.loc[new_user_idx[legit_mask2], 'account_age_days'] = np.random.randint(
    1, 14, n_new_legit
)
df.loc[new_user_idx[legit_mask2], 'is_new_recipient'] = 1
df.loc[new_user_idx[legit_mask2], 'is_new_device']    = 1
df.loc[new_user_idx[legit_mask2], 'established_user_new_recipient'] = 0

print(f"Noise injected into {n_noise:,} records")


# ═══════════════════════════════════════════════════════════════════
# STEP 10 — SHUFFLE AND ENFORCE COLUMN ORDER
# ═══════════════════════════════════════════════════════════════════

print("Shuffling dataset...")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

FINAL_COLUMNS = [
    'transaction_id',
    'name_sender',
    'name_recipient',
    'transfer_type',
    'amount',
    'avg_transaction_amount_30d',
    'amount_vs_avg_ratio',
    'transaction_hour',
    'is_weekend',
    'sender_account_fully_drained',
    'is_new_device',
    'session_duration_seconds',
    'failed_login_attempts',
    'is_proxy_ip',
    'ip_risk_score',
    'country_mismatch',
    'account_age_days',
    'tx_count_24h',
    'is_new_recipient',
    'established_user_new_recipient',
    'recipient_risk_profile_score',
    'is_fraud'
]

df = df[FINAL_COLUMNS]
print(f"Final shape: {df.shape}")


# ═══════════════════════════════════════════════════════════════════
# STEP 11 — SANITY CHECKS
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*50)
print("TRANSACTIONS SANITY CHECKS")
print("="*50)

print(f"\nTotal records:  {len(df):,}")
print(f"Fraud cases:    {df['is_fraud'].sum():,}")
print(f"Legit cases:    {(df['is_fraud']==0).sum():,}")
print(f"Fraud rate:     {df['is_fraud'].mean():.4f} (expect 0.0150)")

print("\nTop correlations with is_fraud:")
corr = df.corr(numeric_only=True)['is_fraud'].sort_values(ascending=False)
print(corr[corr.index != 'is_fraud'].head(8).round(3))

fraud_hours = df[df['is_fraud']==1]['transaction_hour']
night_pct   = ((fraud_hours <= 6) | (fraud_hours >= 22)).mean()
print(f"\n% fraud 10PM-6AM: {night_pct:.2%} (expect >50%)")

print(f"\nFraud avg amount_vs_avg_ratio:  {df[df['is_fraud']==1]['amount_vs_avg_ratio'].mean():.2f} (expect >3.0)")
print(f"Legit avg amount_vs_avg_ratio:  {df[df['is_fraud']==0]['amount_vs_avg_ratio'].mean():.2f} (expect 0.8-1.2)")

print(f"\nFraud avg session: {df[df['is_fraud']==1]['session_duration_seconds'].mean():.1f}s (expect <45s)")
print(f"Legit avg session: {df[df['is_fraud']==0]['session_duration_seconds'].mean():.1f}s (expect >90s)")

top_recipients = df[df['is_fraud']==1]['name_recipient'].value_counts()
top5_share = top_recipients.head(5).sum() / N_FRAUD
print(f"\nTop 5 recipients handle {top5_share:.1%} of fraud (expect >30%)")
print(f"Most used recipient: {top_recipients.iloc[0]:,} transactions")

print(f"\nFraud avg recipient_risk_profile_score: {df[df['is_fraud']==1]['recipient_risk_profile_score'].mean():.3f} (expect >0.60)")
print(f"Legit avg recipient_risk_profile_score: {df[df['is_fraud']==0]['recipient_risk_profile_score'].mean():.3f} (expect <0.15)")

print(f"\nFraud avg ip_risk_score: {df[df['is_fraud']==1]['ip_risk_score'].mean():.3f} (expect >0.55)")
print(f"Legit avg ip_risk_score: {df[df['is_fraud']==0]['ip_risk_score'].mean():.3f} (expect <0.15)")

impossible = df[
    (df['account_age_days'] < 7) &
    (df['avg_transaction_amount_30d'] > 10000)
].shape[0]
print(f"\nImpossible combinations: {impossible} (expect 0)")

print(f"\nTransfer type distribution:")
print(df['transfer_type'].value_counts())

# Score overlap verification — critical check after fixes
print("\n=== SCORE OVERLAP CHECK ===")
fraud_scores = df[df['is_fraud']==1]['recipient_risk_profile_score']
legit_scores = df[df['is_fraud']==0]['recipient_risk_profile_score']

print(f"Fraud score — mean: {fraud_scores.mean():.3f}, "
      f"min: {fraud_scores.min():.3f}, "
      f"max: {fraud_scores.max():.3f}")
print(f"Legit score — mean: {legit_scores.mean():.3f}, "
      f"min: {legit_scores.min():.3f}, "
      f"max: {legit_scores.max():.3f}")

# Overlap zone 0.30 to 0.55
overlap_fraud = ((fraud_scores >= 0.30) & (fraud_scores <= 0.55)).sum()
overlap_legit = ((legit_scores >= 0.30) & (legit_scores <= 0.55)).sum()
print(f"\nIn overlap zone (0.30-0.55):")
print(f"  Fraud transactions: {overlap_fraud:,} "
      f"({overlap_fraud/N_FRAUD:.1%} of all fraud)")
print(f"  Legit transactions: {overlap_legit:,} "
      f"({overlap_legit/N_LEGIT:.1%} of all legit)")

# Confirm legitimate scores can exceed 0.40
legit_above_40 = (legit_scores > 0.40).sum()
print(f"\nLegit transactions with score > 0.40: {legit_above_40:,} "
      f"(expect >5,000)")

# Confirm fraud scores can drop below 0.40
fraud_below_40 = (fraud_scores < 0.40).sum()
print(f"Fraud transactions with score < 0.40: {fraud_below_40:,} "
      f"(expect >1,000)")


# ═══════════════════════════════════════════════════════════════════
# STEP 12 — SAVE TRANSACTIONS.CSV
# ═══════════════════════════════════════════════════════════════════

print(f"\nSaving transactions to {TRANSACTIONS_FILE}...")
start = time.time()
df.to_csv(TRANSACTIONS_FILE, index=False)
print(f"Saved in {time.time()-start:.1f}s")
print(f"File size: {os.path.getsize(TRANSACTIONS_FILE)/1024/1024:.1f} MB")

print("\n=== GENERATION COMPLETE ===")
print(f"accounts.csv    → {os.path.getsize(ACCOUNTS_FILE)/1024/1024:.1f} MB")
print(f"transactions.csv → {os.path.getsize(TRANSACTIONS_FILE)/1024/1024:.1f} MB")
print(f"\nSample fraud record:\n{df[df['is_fraud']==1].iloc[0].to_dict()}")
print(f"\nSample legit record:\n{df[df['is_fraud']==0].iloc[0].to_dict()}")
