import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

YEARS = [2000, 2005, 2010, 2015]

# STEP 1: Load existing P3 dataset
dataset = pd.read_parquet(
    ROOT / "data" / "processed" / "p3_dataset_v1.parquet"
)

# STEP 2: Calculate product ubiquity
ubiquity_list = []

matrix_folder = ROOT / "p2" / "data-p2" / "matrices"

for year in YEARS:

    print(f"Calculating ubiquity for {year}")

    matrix = pd.read_parquet(
        matrix_folder / f"matrix_A{year}.parquet"
    )

    # Count competitive countries for each product
    counts = matrix.sum(axis=0)

    ubiquity = pd.DataFrame({
        "product": counts.index,
        "asof_year": year,
        "f_ubiquity": counts.values
    })

    ubiquity_list.append(ubiquity)

ubiquity_all = pd.concat(
    ubiquity_list,
    ignore_index=True
)

# STEP 3: Load historical trade data
trade = pd.read_parquet(
    ROOT / "data" / "processed" / "trade_rca_v1.parquet"
)

# Calculate total global exports for each product-year
world_exports = (
    trade.groupby(["year", "product"])["export_value"]
    .sum()
)

# STEP 4: Calculate three-year world export growth
growth_list = []

for year in YEARS:

    previous_year = year - 3

    print(f"Calculating world export growth: {previous_year} to {year}")

    if year not in world_exports.index.get_level_values("year"):
        raise ValueError(f"Missing global export data for {year}")

    if previous_year not in world_exports.index.get_level_values("year"):
        raise ValueError(
            f"Missing global export data for {previous_year}"
        )

    current = world_exports.loc[year]
    previous = world_exports.loc[previous_year]

    products = dataset.loc[
        dataset["asof_year"] == year, "product"
    ].unique()

    current = current.reindex(products, fill_value=0)
    previous = previous.reindex(products, fill_value=0)

    # Log change in global export value
    growth = (
        np.log1p(current) - np.log1p(previous)
    )

    growth_df = pd.DataFrame({
        "product": products,
        "asof_year": year,
        "f_world_growth": growth.values
    })

    growth_list.append(growth_df)

growth_all = pd.concat(
    growth_list,
    ignore_index=True
)

# STEP 5: Merge both engineered features
keys = ["asof_year", "product"]

dataset = dataset.merge(
    ubiquity_all,
    on=keys,
    how="left",
    validate="many_to_one"
)

dataset = dataset.merge(
    growth_all,
    on=keys,
    how="left",
    validate="many_to_one"
)

# STEP 6: Validate
print("\nMissing ubiquity:", dataset["f_ubiquity"].isna().sum())
print("Missing world growth:", dataset["f_world_growth"].isna().sum())

if dataset[["f_ubiquity", "f_world_growth"]].isna().any().any():
    raise ValueError("Missing engineered feature values!")

# STEP 7: Save
output_path = (
    ROOT / "data" / "processed" / "p3_dataset_v2.parquet"
)

dataset.to_parquet(output_path, index=False)

print("\nFinal dataset shape:", dataset.shape)
print("Saved:", output_path)