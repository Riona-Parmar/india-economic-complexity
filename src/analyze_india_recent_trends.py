
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data/processed/country_year_sector_trade.csv"
OUTPUT_FILE = ROOT / "outputs/reports/india_sector_recent_trends.csv"

df = pd.read_csv(INPUT_FILE)
india = df[df["country_iso3_code"] == "IND"].copy()

india["year"] = india["year"].astype(int)

# Compare the most recent five available years with the preceding five.
latest_year = int(india["year"].max())
recent_start = latest_year - 4
previous_start = latest_year - 9
previous_end = latest_year - 5

recent = india[india["year"].between(recent_start, latest_year)]
previous = india[india["year"].between(previous_start, previous_end)]

recent_summary = (
    recent.groupby(["sector_id", "sector_name"], as_index=False)
    .agg(
        recent_exports=("export_value", "mean"),
        recent_imports=("import_value", "mean"),
        recent_years=("year", "nunique")
    )
)

previous_summary = (
    previous.groupby("sector_id", as_index=False)
    .agg(
        previous_exports=("export_value", "mean"),
        previous_imports=("import_value", "mean"),
        previous_years=("year", "nunique")
    )
)

result = recent_summary.merge(previous_summary, on="sector_id")

result["export_change_pct"] = (
    (result["recent_exports"] / result["previous_exports"] - 1) * 100
).where(result["previous_exports"] > 0)

result["recent_trade_balance"] = (
    result["recent_exports"] - result["recent_imports"]
)

result["previous_trade_balance"] = (
    result["previous_exports"] - result["previous_imports"]
)

result["trade_balance_change"] = (
    result["recent_trade_balance"] - result["previous_trade_balance"]
)

result = result.sort_values("recent_exports", ascending=False)
result.to_csv(OUTPUT_FILE, index=False)

print(f"Recent period: {recent_start}-{latest_year}")
print(f"Comparison period: {previous_start}-{previous_end}")
print("Output:", OUTPUT_FILE)

print("\nTop 10 sectors by recent average exports:")
print(
    result[
        ["sector_name", "recent_exports", "export_change_pct",
         "recent_trade_balance", "recent_years", "previous_years"]
    ].head(10).to_string(index=False)
)

print("\nTop 10 sectors by export growth between the two periods:")
print(
    result.dropna(subset=["export_change_pct"])
    .sort_values("export_change_pct", ascending=False)[
        ["sector_name", "previous_exports", "recent_exports",
         "export_change_pct"]
    ].head(10).to_string(index=False)
)
