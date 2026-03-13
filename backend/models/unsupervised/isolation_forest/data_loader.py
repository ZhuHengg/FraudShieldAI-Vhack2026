import pandas as pd
import time
import os

def load_data(data_path: str) -> pd.DataFrame:
    """
    Loads the e-wallet transaction dataset from the specified path.
    """
    print("=" * 60)
    print("STEP 1: LOAD DATA")
    print("=" * 60)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    print("Loading dataset...")
    start = time.time()
    df = pd.read_csv(data_path)

    print(f"Shape: {df.shape}")
    print(f"Fraud cases: {df['is_fraud'].sum():,}")
    print(f"Legit cases: {(df['is_fraud']==0).sum():,}")
    print(f"Fraud rate: {df['is_fraud'].mean():.4f}")
    print(f"Loaded in {time.time()-start:.1f}s\n")
    
    return df
