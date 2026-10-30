
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "hs92_country_product_year_4.csv"
OUTPUT_FILE = ROOT / "data" / "interim" / "country_coverage_audit.csv"

print("Loading country and year coverage from the Atlas data...")

df = pd.read_csv(
    RAW_FILE,
    usecols=[
        "country_iso3_code",
        "year",
        "export_value",
    ],
    dtype={"country_iso3_code": "string"},
    low_memory=False,
)

df = df.loc[
    df["year"].between(1995, 2024)
].copy()

country_year = (
    df.groupby(["country_iso3_code", "year"], as_index=False)
    ["export_value"]
    .sum()
)

coverage = (
    country_year.groupby("country_iso3_code")
    .agg(
        years_present=("year", "nunique"),
        first_year=("year", "min"),
        last_year=("year", "max"),
        total_exports=("export_value", "sum"),
    )
    .reset_index()
    .sort_values("total_exports", ascending=False)
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
coverage.to_csv(OUTPUT_FILE, index=False)

print("\nCOUNTRY COVERAGE AUDIT")
print("=" * 70)
print(f"Countries: {len(coverage):,}")
print(f"Coverage period: {df['year'].min()}–{df['year'].max()}")
print("\nTop 20 countries by cumulative exports:")
print(coverage.head(20).to_string(index=False))

print("\nCoverage distribution:")
print(coverage["years_present"].describe().to_string())

print("\nCountries with at least 25 years of coverage:")
print(
    int((coverage["years_present"] >= 25).sum())
)

print("\nRequired countries:")
print(
    coverage.loc[
        coverage["country_iso3_code"].isin(
            ["IND", "VNM", "BGD", "MEX", "IDN", "TUR"]
        )
    ].to_string(index=False)
)

print(f"\nFull report saved to: {OUTPUT_FILE}")
