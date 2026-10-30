
from pathlib import Path
import gc
import json
import time

import numpy as np
import pandas as pd
import yaml


# ------------------------------------------------------------
# 1. Paths and configuration
# ------------------------------------------------------------
ROOT = Path.cwd()

CONFIG_PATH = ROOT / "config.yaml"
RAW_PATH = ROOT / "data/raw/hs92_country_product_year_4.csv"
KEEP_PATH = ROOT / "data/interim/keep_countries.csv"
OUTPUT_DIR = ROOT / "data/processed"
INTERIM_DIR = ROOT / "data/interim"

OUTPUT_PATH = OUTPUT_DIR / "trade_rca_v1.parquet"
AUDIT_PATH = INTERIM_DIR / "trade_rca_source_comparison_v1.parquet"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

started = time.time()

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

first_year = int(config["first_year"])
last_year = int(config["last_year"])
india = str(config["india"])
peers = set(map(str, config["peers"]))

required_columns = [
    "country_iso3_code",
    "product_hs92_code",
    "year",
    "export_value",
    "export_rca",
    "pci",
    "distance",
    "cog",
]

print("Loading raw trade data...")
raw = pd.read_csv(
    RAW_PATH,
    usecols=required_columns,
    dtype={
        "country_iso3_code": "string",
        "product_hs92_code": "string",
    },
    low_memory=False,
)

# Normalize identifiers and numeric fields.
raw["country"] = raw["country_iso3_code"].str.strip().str.upper()
raw["product"] = (
    raw["product_hs92_code"]
    .str.strip()
    .str.upper()
    .str.zfill(4)
)
raw["year"] = pd.to_numeric(raw["year"], errors="raise").astype("int16")

for col in ["export_value", "export_rca", "pci", "distance", "cog"]:
    raw[col] = pd.to_numeric(raw[col], errors="coerce")

raw = raw.loc[
    raw["year"].between(first_year, last_year)
].copy()

if raw.duplicated(["country", "product", "year"]).any():
    raise ValueError("Duplicate country-product-year keys found in raw data.")

if not raw["product"].str.len().eq(4).all():
    raise ValueError("At least one product code is not four characters long.")

print(f"Raw rows in configured years: {len(raw):,}")
print(f"Years: {raw['year'].min()}–{raw['year'].max()}")
print(f"Countries: {raw['country'].nunique():,}")
print(f"Products: {raw['product'].nunique():,}")


# ------------------------------------------------------------
# 2. Calculate standard Balassa RCA using ALL countries first
# ------------------------------------------------------------
print("\nCalculating RCA using the full global dataset...")

# X_cp: country-product exports in a given year.
# X_c: total exports of a country in a given year.
# X_p: global exports of a product in a given year.
# X_world: total global exports in a given year.

raw["export_value"] = raw["export_value"].fillna(0.0).astype("float64")

country_totals = (
    raw.groupby(["country", "year"], observed=True)["export_value"]
    .sum()
    .rename("country_total")
)

product_totals = (
    raw.groupby(["product", "year"], observed=True)["export_value"]
    .sum()
    .rename("product_total")
)

world_totals = (
    raw.groupby("year", observed=True)["export_value"]
    .sum()
    .rename("world_total")
)

raw = raw.join(country_totals, on=["country", "year"])
raw = raw.join(product_totals, on=["product", "year"])
raw = raw.join(world_totals, on="year")

country_share = np.divide(
    raw["export_value"].to_numpy(),
    raw["country_total"].to_numpy(),
    out=np.zeros(len(raw), dtype="float64"),
    where=raw["country_total"].to_numpy() > 0,
)

world_product_share = np.divide(
    raw["product_total"].to_numpy(),
    raw["world_total"].to_numpy(),
    out=np.zeros(len(raw), dtype="float64"),
    where=raw["world_total"].to_numpy() > 0,
)

calculated_rca = np.divide(
    country_share,
    world_product_share,
    out=np.zeros(len(raw), dtype="float64"),
    where=world_product_share > 0,
)

raw["rca_calculated"] = calculated_rca

# Save a compact audit before filtering countries.
print("Saving source-RCA comparison...")
raw[
    [
        "country",
        "product",
        "year",
        "export_value",
        "export_rca",
        "rca_calculated",
    ]
].to_parquet(AUDIT_PATH, index=False)

print(f"RCA audit saved: {AUDIT_PATH}")


# ------------------------------------------------------------
# 3. Keep only countries explicitly marked keep=True
# ------------------------------------------------------------
print("\nApplying country filter...")

keep_df = pd.read_csv(KEEP_PATH, low_memory=False)

if "keep" not in keep_df.columns:
    raise ValueError("Country filter file has no 'keep' column.")

keep_values = (
    keep_df["keep"]
    .astype(str)
    .str.strip()
    .str.lower()
)

selected_countries = set(
    keep_df.loc[
        keep_values.isin(["true", "1", "yes"]),
        "country_iso3_code",
    ]
    .astype(str)
    .str.strip()
    .str.upper()
)

if not selected_countries:
    raise ValueError("No countries selected by keep=True.")

required_countries = {india} | peers
missing_required = required_countries - selected_countries

if missing_required:
    raise ValueError(
        f"Required countries missing from keep=True filter: "
        f"{sorted(missing_required)}"
    )

print(f"Selected countries: {len(selected_countries)}")
print(f"India included: {india in selected_countries}")
print(f"Peers included: {sorted(peers & selected_countries)}")

# Keep the global RCA calculation intact; filtering happens only now.
panel_source = raw.loc[
    raw["country"].isin(selected_countries),
    [
        "country",
        "product",
        "year",
        "export_value",
        "rca_calculated",
        "pci",
        "distance",
        "cog",
    ],
].copy()

# Capture the global product universe before building the zero-filled grid.
all_products = sorted(raw["product"].dropna().unique().tolist())

# Release large intermediate objects before creating the panel.
del raw, country_totals, product_totals, world_totals
del country_share, world_product_share, calculated_rca
gc.collect()


# ------------------------------------------------------------
# 4. Create a complete country × product × year panel
# ------------------------------------------------------------
print("\nBuilding the zero-filled panel...")

years = list(range(first_year, last_year + 1))
countries = sorted(selected_countries)

grid = pd.MultiIndex.from_product(
    [countries, all_products, years],
    names=["country", "product", "year"],
)

panel_source = panel_source.set_index(
    ["country", "product", "year"]
).sort_index()

panel = panel_source.reindex(grid)

# Absent trade records represent zero exports.
panel["export_value"] = panel["export_value"].fillna(0.0)
panel["rca"] = panel["rca_calculated"].fillna(0.0)

# Preserve known product-year attributes, filling gaps from the
# product-year median and then the overall product median.
for col in ["pci", "distance", "cog"]:
    product_year_median = (
        panel.groupby(["product", "year"], observed=True)[col]
        .transform("median")
    )
    product_median = (
        panel.groupby("product", observed=True)[col]
        .transform("median")
    )
    panel[col] = panel[col].fillna(product_year_median)
    panel[col] = panel[col].fillna(product_median)

del panel_source, grid
gc.collect()


# ------------------------------------------------------------
# 5. Compute rca3, m, ECI proxy and rename Atlas features
# ------------------------------------------------------------
print("\nComputing rolling RCA and ECI proxy...")

panel = panel.reset_index()
panel = panel.sort_values(
    ["country", "product", "year"],
    kind="mergesort",
).reset_index(drop=True)

panel["rca3"] = (
    panel.groupby(["country", "product"], sort=False)["rca"]
    .transform(
        lambda s: s.rolling(window=3, min_periods=1).mean()
    )
)

panel["m"] = (panel["rca3"] >= 1.0).astype("int8")

panel["eci"] = (
    panel["pci"].where(panel["m"].eq(1))
    .groupby([panel["country"], panel["year"]])
    .transform("mean")
)

panel = panel.rename(
    columns={
        "pci": "pci",
        "distance": "atlas_distance",
        "cog": "atlas_cog",
    }
)

final_columns = [
    "country",
    "product",
    "year",
    "export_value",
    "rca",
    "rca3",
    "m",
    "pci",
    "eci",
    "atlas_distance",
    "atlas_cog",
]

panel = panel[final_columns]

# Required P1 self-checks.
if panel[["rca", "rca3", "m"]].isna().any().any():
    raise ValueError("NaN found in required RCA fields.")

if not panel["product"].str.len().eq(4).all():
    raise ValueError("Final panel contains non-four-character products.")

if panel.duplicated(["country", "product", "year"]).any():
    raise ValueError("Duplicate keys found in final panel.")

if set(panel["country"].unique()) != selected_countries:
    raise ValueError("Final panel country set does not match selected countries.")

panel.to_parquet(OUTPUT_PATH, index=False)

print("\nBUILD COMPLETE")
print(f"Output: {OUTPUT_PATH}")
print(f"Rows: {len(panel):,}")
print(f"Countries: {panel['country'].nunique():,}")
print(f"Products: {panel['product'].nunique():,}")
print(f"Years: {panel['year'].min()}–{panel['year'].max()}")
print(f"NaN in rca: {panel['rca'].isna().sum()}")
print(f"NaN in rca3: {panel['rca3'].isna().sum()}")
print(f"NaN in m: {panel['m'].isna().sum()}")
print(f"Elapsed seconds: {time.time() - started:.1f}")
