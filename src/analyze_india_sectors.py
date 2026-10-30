
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data/processed/country_year_sector_trade.csv"
OUTPUT_DIR = ROOT / "outputs/reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

# India is identified using its ISO-3 country code.
india = df[df["country_iso3_code"] == "IND"].copy()

if india.empty:
    raise ValueError("India (IND) was not found in the dataset.")

india["year"] = india["year"].astype(int)

# Summarize each sector over the full available period.
summary = (
    india.groupby(["sector_id", "sector_name"], as_index=False)
    .agg(
        first_year=("year", "min"),
        last_year=("year", "max"),
        total_exports=("export_value", "sum"),
        total_imports=("import_value", "sum"),
        average_annual_exports=("export_value", "mean"),
        average_annual_imports=("import_value", "mean"),
        years_observed=("year", "nunique"),
    )
)

summary["trade_balance"] = (
    summary["total_exports"] - summary["total_imports"]
)

# Compare the earliest and latest years available for each sector.
first = (
    india.sort_values("year")
    .groupby("sector_id")
    .first()[["year", "export_value"]]
    .rename(columns={
        "year": "first_observed_year",
        "export_value": "first_year_exports"
    })
)

last = (
    india.sort_values("year")
    .groupby("sector_id")
    .last()[["year", "export_value"]]
    .rename(columns={
        "year": "last_observed_year",
        "export_value": "last_year_exports"
    })
)

summary = summary.merge(first, on="sector_id")
summary = summary.merge(last, on="sector_id")

summary["export_growth_pct"] = (
    (summary["last_year_exports"] / summary["first_year_exports"] - 1) * 100
).where(summary["first_year_exports"] > 0)

summary = summary.sort_values("total_exports", ascending=False)

output_file = OUTPUT_DIR / "india_sector_performance_1995_2024.csv"
summary.to_csv(output_file, index=False)

print("India sector analysis completed.")
print("Years in source:", india["year"].min(), "-", india["year"].max())
print("Sector count:", india["sector_id"].nunique())
print("Output:", output_file)

print("\nTop 10 sectors by cumulative exports:")
print(
    summary[
        ["sector_id", "sector_name", "total_exports",
         "total_imports", "trade_balance", "years_observed"]
    ].head(10).to_string(index=False)
)

print("\nTop 10 sectors by export growth:")
print(
    summary.dropna(subset=["export_growth_pct"])
    .sort_values("export_growth_pct", ascending=False)[
        ["sector_id", "sector_name", "first_observed_year",
         "last_observed_year", "export_growth_pct"]
    ].head(10).to_string(index=False)
)
