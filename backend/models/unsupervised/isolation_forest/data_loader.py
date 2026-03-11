import pandas as pd
import os

def load_data(data_path: str) -> pd.DataFrame:
    """
    Loads the PaySim dataset from the specified path and prints initial statistics.
    """
    print("=" * 60)
    print("STEP 1: LOAD DATA")
    print("=" * 60)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = pd.read_csv(data_path)
    
    print(f"Dataset shape : {df.shape}")
    print(f"Columns       : {list(df.columns)}")
    print(f"Transaction types:\n{df['type'].value_counts()}")
    print(f"\nisFraud distribution:\n{df['isFraud'].value_counts()}")
    print(f"Overall fraud rate: {df['isFraud'].mean()*100:.4f}%\n")
    
    return df
