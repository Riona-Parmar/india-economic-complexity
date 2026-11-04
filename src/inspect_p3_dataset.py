import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_parquet(
    ROOT / "data" / "processed" / "p3_dataset_v1.parquet"
)

print("DATASET SHAPE:", df.shape)

print("\nALL COLUMNS:")
print(df.columns.tolist())

print("\nMISSING VALUES:")
print(df.isna().sum()[df.isna().sum() > 0])

print("\nLABEL DISTRIBUTION:")
print(df["label"].value_counts())

print("\nSPLIT DISTRIBUTION:")
print(df["split"].value_counts())

print("\nDENSITY STATISTICS:")
print(df["density"].describe())

print("\nPATHWAY DISTRIBUTION:")
print(df["pathway"].value_counts())

print("\nAS-OF YEAR DISTRIBUTION:")
print(df["asof_year"].value_counts().sort_index())