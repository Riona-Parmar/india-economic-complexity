
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = ROOT / "data" / "raw" / "hs92_country_product_year_4.csv"

print("Loading India's historical product records...")

df = pd.read_csv(
    RAW_FILE,
    dtype={
        "country_iso3_code": "string",
        "product_hs92_code": "string",
    },
    low_memory=False,
)

# Keep India and the years needed for historical analysis.
df = df.loc[
    (df["country_iso3_code"] == "IND")
    & df["year"].between(1995, 2024)
].copy()

# Normalize HS4 product codes.
df["product_hs92_code"] = (
    df["product_hs92_code"]
    .str.strip()
    .str.replace(r"\.0$", "", regex=True)
    .str.zfill(4)
)

print("\nINDIA HISTORICAL PRODUCT DATA")
print("=" * 45)
print(f"Records: {len(df):,}")
print(f"Years: {df['year'].min()}–{df['year'].max()}")
print(f"Unique product codes: {df['product_hs92_code'].nunique():,}")
print(f"Unique years: {df['year'].nunique()}")
print(f"Missing export values: {df['export_value'].isna().sum():,}")
print(f"Missing RCA values: {df['export_rca'].isna().sum():,}")
print(f"Missing PCI values: {df['pci'].isna().sum():,}")

print("\nRecords per year:")
print(df.groupby("year").size().to_string())
