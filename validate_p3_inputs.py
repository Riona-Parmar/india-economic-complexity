
import pandas as pd
from pathlib import Path

base = Path("data/processed")

labels = pd.read_parquet(base / "labels_v1.parquet")
features = pd.read_parquet(base / "model_features_v1.parquet")

print("\n========== LABEL DISTRIBUTION ==========")
print(labels["label"].value_counts(dropna=False))

print("\n========== SPLIT DISTRIBUTION ==========")
print(labels["split"].value_counts(dropna=False))

print("\n========== AS-OF YEAR DISTRIBUTION ==========")
print(labels.groupby("asof_year", dropna=False).size())

print("\n========== LABELS BY YEAR AND SPLIT ==========")
print(
    labels.groupby(["asof_year", "split"], dropna=False)
    .agg(rows=("product", "size"),
         positive_labels=("label", lambda x: (x == 1).sum()),
         missing_labels=("label", lambda x: x.isna().sum()))
)

print("\n========== EXISTING FEATURE PATHWAYS ==========")
print(features["pathway"].value_counts(dropna=False))

print("\n========== FEATURE ROW COUNTS ==========")
print("Labels:", len(labels))
print("Features:", len(features))

print("\n========== FEATURE YEARS ==========")
print(features.groupby("asof_year").size())

print("\n========== DUPLICATE KEYS ==========")
keys = ["asof_year", "country", "product"]

print("Duplicate label keys:", labels.duplicated(keys).sum())
print("Duplicate feature keys:", features.duplicated(keys).sum())

print("\n========== FEATURE COLUMNS ==========")
print(features.columns.tolist())
