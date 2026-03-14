import pandas as pd
import time
import os
from sklearn.model_selection import train_test_split

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Splits the dataframe into training and testing sets, stratified by is_fraud.
    """
    print("\n" + "="*50)
    print("BUILDING THE WALL — TRAIN/TEST SPLIT")
    print("="*50)

    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['is_fraud']
    )

    df_train = df_train.reset_index(drop=True)
    df_test  = df_test.reset_index(drop=True)

    print(f"Train: {len(df_train):,} records | Fraud: {df_train['is_fraud'].sum():,}")
    print(f"Test:  {len(df_test):,} records  | Fraud: {df_test['is_fraud'].sum():,}")
    print("Wall built — test set locked until evaluation\n")
    
    return df_train, df_test

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
