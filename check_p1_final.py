from pathlib import Path
import pandas as pd

root = Path.cwd()
paths = {
    "trade_rca": root / "data/processed/trade_rca_v1.parquet",
    "labels": root / "data/processed/labels_v1.parquet",
    "products": root / "data/processed/products_v1.parquet",
    "report": root / "outputs/reports/data_report.md",
}

print("=== P1 FILE CHECK ===")
for name, path in paths.items():
    print(f"{name}: {'FOUND' if path.is_file() else 'MISSING'}")

for name in ("trade_rca", "labels", "products"):
    path = paths[name]
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")

trade = pd.read_parquet(paths["trade_rca"])
labels = pd.read_parquet(paths["labels"])
products = pd.read_parquet(paths["products"])

print("\n=== ROW COUNTS ===")
for name, df in [
    ("trade_rca", trade),
    ("labels", labels),
    ("products", products),
]:
    print(f"{name}: {len(df):,} rows")

print("\n=== REQUIRED COLUMNS ===")
required = {
    "trade_rca": {
        "country", "product", "year", "export_value", "rca",
        "rca3", "m", "pci", "eci", "atlas_distance", "atlas_cog"
    },
    "labels": {
        "asof_year", "target_year", "country",
        "product", "label", "split"
    },
    "products": {
        "product", "name", "hs_section", "sector", "pci_latest"
    },
}
for name, df in [
    ("trade_rca", trade),
    ("labels", labels),
    ("products", products),
]:
    missing = sorted(required[name] - set(df.columns))
    print(f"{name}: {'PASS' if not missing else 'FAIL'}")
    if missing:
        print("  Missing:", missing)

print("\n=== TRADE DATA CHECKS ===")
for col in ["rca", "rca3", "m"]:
    print(f"{col} missing values: {trade[col].isna().sum():,}")

codes = trade["product"].astype(str)
print("Product codes not 4 characters:", int((codes.str.len() != 4).sum()))
print("Year range:", trade["year"].min(), "-", trade["year"].max())
print("Countries:", trade["country"].nunique())
print("Products:", trade["product"].nunique())
print("India 2024 exports:", trade.loc[
    (trade["country"] == "IND") & (trade["year"] == 2024),
    "export_value"
].sum())

print("\n=== LABEL CHECKS ===")
historical = labels[labels["target_year"] > labels["asof_year"]].copy()
print("Historical rows:", f"{len(historical):,}")
print("Historical labels missing:", int(historical["label"].isna().sum()))
print("Forecast rows:", int((labels["split"] == "forecast").sum()))

print("\nIndia positive labels by window:")
india = historical[
    (historical["country"] == "IND") & (historical["label"] == 1)
]
print(india.groupby(["asof_year", "target_year"]).size().to_string())

print("\n=== LABEL ELIGIBILITY CHECK ===")
# Efficiently validate all historical labels against the RCA3 panel.
panel = trade.pivot(
    index=["country", "product"],
    columns="year",
    values="rca3"
)
first_year = int(trade["year"].min())
violations = 0

for asof_year, group in historical.groupby("asof_year"):
    years = list(range(first_year, int(asof_year) + 1))
    if any(year not in panel.columns for year in years):
        raise ValueError(f"Missing years needed to check as-of {asof_year}")

    keys = pd.MultiIndex.from_frame(group[["country", "product"]])
    history = panel.reindex(keys)[years]
    bad = history.isna().any(axis=1) | ~(history < 0.25).all(axis=1)
    count = int(bad.sum())
    violations += count
    print(f"As-of {asof_year}: {count:,} eligibility violations")

print("Total historical eligibility violations:", f"{violations:,}")
print("\nValidation script finished.")
