import pandas as pd

df = pd.read_parquet("data/kkbox/processed/model_table.parquet")
df_subset = df.sample(frac=0.10, random_state=42)
df_subset.to_parquet(
    "data/kkbox/processed/model_table_sagemaker_subset.parquet", index=False
)
print(df_subset.shape)
