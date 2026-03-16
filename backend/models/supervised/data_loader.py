import pandas as pd
import time
import os
from sklearn.model_selection import train_test_split

def split_data(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.2, random_state: int = 42):
    """
    Splits the dataframe into train, validation and test sets.
    Stratified by is_fraud to preserve fraud rate across all three sets.

    Split allocation:
        test = 20% of total
        val  = 20% of remaining 80% = 16% of total
        train = 64% of total
    """
    print("\n" + "="*50)
    print("BUILDING THE WALL — TRAIN/VAL/TEST SPLIT")
    print("="*50)

    # First split: carve out test set
    # Test set is locked — never touched until final evaluation
    df_train_full, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['is_fraud']
    )

    # Second split: carve validation from remaining train
    # Validation set used only for threshold tuning
    df_train, df_val = train_test_split(
        df_train_full,
        test_size=val_size,
        random_state=random_state,
        stratify=df_train_full['is_fraud']
    )

    df_train = df_train.reset_index(drop=True)
    df_val   = df_val.reset_index(drop=True)
    df_test  = df_test.reset_index(drop=True)

    total = len(df)
    print(f"Train: {len(df_train):,} ({len(df_train)/total*100:.0f}%) | "
          f"Fraud: {df_train['is_fraud'].sum():,} "
          f"(rate: {df_train['is_fraud'].mean():.4f})")
    print(f"Val:   {len(df_val):,} ({len(df_val)/total*100:.0f}%)   | "
          f"Fraud: {df_val['is_fraud'].sum():,} "
          f"(rate: {df_val['is_fraud'].mean():.4f})")
    print(f"Test:  {len(df_test):,} ({len(df_test)/total*100:.0f}%) | "
          f"Fraud: {df_test['is_fraud'].sum():,} "
          f"(rate: {df_test['is_fraud'].mean():.4f})")

    # Verify fraud rate is preserved across all three sets
    print(f"\nFraud rate check:")
    print(f"  Original: {df['is_fraud'].mean():.4f}")
    print(f"  Train:    {df_train['is_fraud'].mean():.4f}")
    print(f"  Val:      {df_val['is_fraud'].mean():.4f}")
    print(f"  Test:     {df_test['is_fraud'].mean():.4f}")
    print(f"  (all four must be close to equal)")

    print("\nWall built — val set for threshold tuning only, "
          "test set locked until final evaluation")
    print("="*50 + "\n")

    return df_train, df_val, df_test


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
