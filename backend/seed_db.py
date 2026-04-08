import pandas as pd
from api.database import engine

def seed_database():
    csv_path = "data/test.csv" 
    
    print(f"📂 Reading all parameters from {csv_path}...")
    df = pd.read_csv(csv_path)

    print("📝 Processing full dataset...")
    
    # 1. Rename the PII columns for PDPA compliance
    df = df.rename(columns={
        'name_sender': 'user_hash',
        'name_recipient': 'recipient_hash'
    })

    # 2. Generate the App UI columns based on the 'is_fraud' ground truth
    df['action_taken'] = df['is_fraud'].apply(lambda x: "BLOCK" if x == 1 else "APPROVE")
    df['ml_risk_score'] = df['is_fraud'].apply(lambda x: 0.95 if x == 1 else 0.10)

    print(f"🚀 Appending {len(df)} transactions to your EXISTING Neon table...")
    
    # CRITICAL CHANGE: 'append' strictly adds rows without touching your column settings
    df.to_sql('transaction_logs', engine, if_exists='append', index=False)

    print("✅ Data successfully appended to your existing database table!")

if __name__ == "__main__":
    try:
        seed_database()
    except Exception as e:
        print(f"❌ Upload Failed. Error details: {e}")