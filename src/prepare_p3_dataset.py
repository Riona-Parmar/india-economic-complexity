import pandas as pd
from pathlib import Path

# Find the main project directory
ROOT = Path(__file__).resolve().parents[1]

# STEP 1: Load P1 features
features = pd.read_parquet(
    ROOT / "data" / "processed" / "model_features_v1.parquet"
)

# STEP 2: Keep only new emergence rows
features = features[
    features["pathway"] == "new_emergence"
].copy()

print("P1 features shape:", features.shape)

# STEP 3: Load all four P2 density files
density_folder = ROOT / "p2" / "data-p2" / "density"

density_list = []

for year in [2000, 2005, 2010, 2015]:

    file_path = density_folder / f"density_A{year}.parquet"

    density = pd.read_parquet(file_path)

    density_list.append(density)

# Combine all density files
density_all = pd.concat(density_list, ignore_index=True)

print("P2 density shape:", density_all.shape)

# STEP 4: Merge using common keys
keys = ["asof_year", "country", "product"]

dataset = features.merge(
    density_all,
    on=keys,
    how="left",
    validate="one_to_one",
    indicator=True
)

# STEP 5: Validate the merge
print("\nMerge results:")
print(dataset["_merge"].value_counts())

print("\nMissing density values:", dataset["density"].isna().sum())

# Stop if any density values are missing
if dataset["density"].isna().any():
    raise ValueError("Some rows are missing density values!")

# Remove merge indicator
dataset = dataset.drop(columns=["_merge"])

# STEP 6: Save the combined dataset
output_path = ROOT / "data" / "processed" / "p3_dataset_v1.parquet"

dataset.to_parquet(output_path, index=False)

print("\nFinal dataset shape:", dataset.shape)
print("Saved:", output_path)