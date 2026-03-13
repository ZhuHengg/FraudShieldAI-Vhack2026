import pandas as pd
import time

input_file = 'data/synthetic_ewallet_fraud.csv'
output_file = 'data/filtered_fraud_data.csv'

print(f"Loading {input_file}...")
start_time = time.time()
df = pd.read_csv(input_file)
print(f"Loaded {len(df):,} rows in {time.time() - start_time:.2f} seconds.")

# Filter: is_fraud == 1 OR recipient_risk_profile_score > 0.5
print("Filtering data...")
filtered_df = df[(df['is_fraud'] == 1) | (df['recipient_risk_profile_score'] > 0.5)]

print(f"Filtered down to {len(filtered_df):,} rows.")

print(f"Saving to {output_file}...")
filtered_df.to_csv(output_file, index=False)
print("Done!")
