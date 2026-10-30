
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = ROOT / "data" / "raw" / "hs92_country_product_year_4.csv"

df = pd.read_csv(
    RAW_FILE,
    usecols=[
        "country_iso3_code",
        "product_hs92_code",
        "year",
        "export_value",
        "export_rca",
        "pci",
    ],
    dtype={
        "country_iso3_code": "string",
        "product_hs92_code": "string",
    },
    low_memory=False,
)

df = df.loc[df["country_iso3_code"] == "IND"].copy()

df["product_hs92_code"] = (
    df["product_hs92_code"]
    .str.strip()
    .str.replace(r"\.0$", "", regex=True)
    .str.zfill(4)
)

# A product-year record is present even if its exports are zero.
coverage = (
    df.groupby("product_hs92_code")["year"]
    .nunique()
    .rename("years_present")
    .reset_index()
)

print("PRODUCT HISTORY COVERAGE")
print("=" * 45)
print(f"Products in dataset: {len(coverage):,}")
print(f"Present in all 30 years: {(coverage['years_present'] == 30).sum():,}")
print(f"Present in 25–29 years: {coverage['years_present'].between(25, 29).sum():,}")
print(f"Present in 10–24 years: {coverage['years_present'].between(10, 24).sum():,}")
print(f"Present in fewer than 10 years: {(coverage['years_present'] < 10).sum():,}")

print("\nProducts with the fewest years present:")
print(
    coverage.sort_values(["years_present", "product_hs92_code"])
    .head(15)
    .to_string(index=False)
)

print("\nSpecial-code coverage:")
print(
    coverage.loc[
        coverage["product_hs92_code"] == "XXXX"
    ].to_string(index=False)
)

output_file = ROOT / "data" / "interim" / "india_product_history_coverage.csv"
coverage.to_csv(output_file, index=False)

print(f"\nCoverage report saved to: {output_file}")
