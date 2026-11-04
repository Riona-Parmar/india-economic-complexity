import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_parquet(
    ROOT / "data" / "processed" / "p3_dataset_v2.parquet"
)

# Features planned for XGBoost
features = [
    "density",
    "pci_current",
    "rca3_current",
    "rca3_change_3y",
    "eci_current",
    "f_ubiquity",
    "f_world_growth"
]

print("FEATURE STATISTICS:")
print(df[features].describe().T)

print("\nMISSING VALUES:")
print(df[features].isna().sum())

print("\nINFINITE VALUES:")
print(
    np.isinf(df[features].select_dtypes(include=np.number))
    .sum()
)

print("\nDUPLICATE KEYS:")
print(
    df.duplicated(
        ["asof_year", "country", "product"]
    ).sum()
)

print("\nFEATURES BY YEAR:")
print(
    df.groupby("asof_year")[features].mean().round(4)
)

print("\nLABEL DISTRIBUTION:")
print(df["label"].value_counts())

print("\nVALIDATION COMPLETE")