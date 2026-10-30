
from pathlib import Path
import pandas as pd

# Project root: parent of the src directory
ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = ROOT / "data/raw/hs92_country_product_year_4.csv"
MAPPING_FILE = ROOT / "data/interim/hs92_hs4_sector_mapping_v2.csv"
OUTPUT_FILE = ROOT / "data/processed/country_year_sector_trade.csv"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

print("Loading product-to-sector mapping...")

mapping = pd.read_csv(
    MAPPING_FILE,
    dtype={"product_hs92_code": str}
)

mapping = mapping[[
    "product_hs92_code",
    "sector_id",
    "sector_name"
]].copy()

mapping["product_hs92_code"] = (
    mapping["product_hs92_code"].str.zfill(4)
)

print("Loading merchandise trade data...")

trade = pd.read_csv(
    RAW_FILE,
    dtype={"product_hs92_code": str},
    low_memory=False
)

trade["product_hs92_code"] = (
    trade["product_hs92_code"].str.zfill(4)
)

print("Joining trade records to sector assignments...")

trade = trade.merge(
    mapping,
    on="product_hs92_code",
    how="left",
    validate="many_to_one"
)

# The special code XXXX is not assigned to a merchandise sector.
unmapped_rows = trade["sector_id"].isna().sum()
print("Trade rows without a sector assignment:", unmapped_rows)

# Exclude unassigned records from sector totals.
trade = trade.dropna(subset=["sector_id"]).copy()
trade["sector_id"] = trade["sector_id"].astype(int)

# Aggregate export and import values across HS4 products.
sector_trade = (
    trade.groupby(
        [
            "country_id",
            "country_iso3_code",
            "year",
            "sector_id",
            "sector_name"
        ],
        as_index=False,
        dropna=False
    )
    .agg(
        export_value=("export_value", "sum"),
        import_value=("import_value", "sum"),
        product_count=("product_hs92_code", "nunique")
    )
)

sector_trade["trade_balance"] = (
    sector_trade["export_value"] - sector_trade["import_value"]
)

sector_trade.to_csv(OUTPUT_FILE, index=False)

print("\nSector-level dataset created.")
print("Output:", OUTPUT_FILE)
print("Rows:", len(sector_trade))
print("Countries:", sector_trade["country_iso3_code"].nunique())
print("Years:", sector_trade["year"].min(), "-", sector_trade["year"].max())
print("Sectors:", sector_trade["sector_id"].nunique())
print("\nSample:")
print(sector_trade.head(10).to_string(index=False))
