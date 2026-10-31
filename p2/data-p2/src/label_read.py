import pandas as pd

labels = pd.read_parquet("labels_v1.parquet")

print("Shape:", labels.shape)
print("\nColumns:")
print(labels.columns.tolist())

print("\nFirst 5 rows:")
print(labels.head())

print("\nData types:")
print(labels.dtypes)

print("\nMissing values:")
print(labels.isna().sum())

print("\nSplit counts:")
print(labels["split"].value_counts())