
import pandas as pd
from pathlib import Path

base = Path("data/processed")

labels = pd.read_parquet(base / "labels_v1.parquet")
features = pd.read_parquet(base / "model_features_v1.parquet")

# Keep only historical labels
historical = labels[labels["label"].notna()].copy()

# Keep only the new emergence pathway
emergence = features[
    features["pathway"] == "new_emergence"
].copy()

keys = ["asof_year", "country", "product"]

print("Historical label rows:", len(historical))
print("New emergence feature rows:", len(emergence))

# Check which label records are missing from features
comparison = historical[keys].merge(
    emergence[keys],
    on=keys,
    how="outer",
    indicator=True
)

print("\nAlignment results:")
print(comparison["_merge"].value_counts())

print("\nFeature rows by split:")
print(emergence["split"].value_counts())

print("\nFeature rows by year and pathway:")
print(features.groupby(["asof_year", "pathway"]).size())

print("\nFeature label distribution:")
print(emergence["label"].value_counts(dropna=False))
