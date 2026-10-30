from pathlib import Path
import pandas as pd

ROOT = Path(".")
trade_path = ROOT / "data/processed/trade_rca_v1.parquet"
labels_path = ROOT / "data/processed/labels_v1.parquet"
products_path = ROOT / "data/processed/products_v1.parquet"

peers = {"BGD", "IDN", "MEX", "TUR", "VNM"}
india = "IND"
first_year = 1995
horizon = 5
eligibility_threshold = 0.25
emergence_threshold = 1.0

trade = pd.read_parquet(trade_path)
labels = pd.read_parquet(labels_path)
products = pd.read_parquet(products_path)

errors = []

def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        errors.append(name)

print("=== 1. FILES AND KEYS ===")
check(
    "Trade panel key is unique",
    not trade.duplicated(["country", "product", "year"]).any(),
    f"duplicate keys: {trade.duplicated(['country','product','year']).sum():,}"
)
check(
    "Label key is unique",
    not labels.duplicated(["asof_year", "country", "product"]).any(),
    f"duplicate keys: {labels.duplicated(['asof_year','country','product']).sum():,}"
)
check(
    "Product key is unique",
    not products.duplicated(["product"]).any(),
    f"duplicate keys: {products.duplicated(['product']).sum():,}"
)

print("\n=== 2. TRADE PANEL ===")
check("RCA has no missing values", trade["rca"].notna().all())
check("RCA3 has no missing values", trade["rca3"].notna().all())
check("m has no missing values", trade["m"].notna().all())
check(
    "Years are within configured range",
    trade["year"].min() == first_year and trade["year"].max() == 2024,
    f"{trade['year'].min()}–{trade['year'].max()}"
)
check(
    "Product codes are four characters",
    trade["product"].astype(str).str.len().eq(4).all()
)
print("Countries:", trade["country"].nunique())
print("Products:", trade["product"].nunique())
print("Rows:", f"{len(trade):,}")

print("\n=== 3. SPLIT ASSIGNMENTS ===")
expected_split = labels["country"].map(
    lambda c: "test_india" if c == india
    else "test_peer" if c in peers
    else "train"
)
historical = labels[labels["label"].notna()].copy()
forecast = labels[labels["label"].isna()].copy()

bad_historical_split = historical["split"].ne(
    expected_split.loc[historical.index]
)
check(
    "Historical split matches country assignment",
    not bad_historical_split.any(),
    f"mismatches: {bad_historical_split.sum():,}"
)
check(
    "Forecast rows have forecast split",
    forecast["split"].eq("forecast").all(),
    f"forecast rows: {len(forecast):,}"
)
check(
    "Forecast labels are unobserved",
    forecast["label"].isna().all()
)
check(
    "Historical labels are binary",
    historical["label"].isin([0, 1]).all()
)
check(
    "Historical target year equals as-of year plus horizon",
    (historical["target_year"] == historical["asof_year"] + horizon).all()
)
print("Historical rows:", f"{len(historical):,}")
print("Forecast rows:", f"{len(forecast):,}")

print("\n=== 4. LABEL TRUTH ===")
# Compare each historical label with RCA3 at its target year.
target = trade[["country", "product", "year", "rca3"]].rename(
    columns={"year": "target_year", "rca3": "target_rca3"}
)
truth = historical.merge(
    target,
    on=["country", "product", "target_year"],
    how="left",
    validate="many_to_one"
)
check(
    "Every historical row has target RCA3",
    truth["target_rca3"].notna().all(),
    f"missing targets: {truth['target_rca3'].isna().sum():,}"
)
truth_checkable = truth["target_rca3"].notna()
expected_label = (
    truth.loc[truth_checkable, "target_rca3"] >= emergence_threshold
).astype("int8")
actual_label = truth.loc[truth_checkable, "label"].astype("int8")
truth_mismatches = int(expected_label.ne(actual_label).sum())
check(
    "Labels match target RCA3 >= 1.0",
    truth_mismatches == 0,
    f"mismatches: {truth_mismatches:,}"
)

print("\n=== 5. HISTORICAL ELIGIBILITY ===")
# For each country-product-year, cumulative flag indicates whether
# RCA3 has ever reached/exceeded 0.25 since the first data year.
panel = trade[["country", "product", "year", "rca3"]].sort_values(
    ["country", "product", "year"]
).copy()
panel["ever_ineligible"] = (
    panel["rca3"].ge(eligibility_threshold)
    .groupby([panel["country"], panel["product"]])
    .cummax()
)
eligibility_at_asof = panel.rename(
    columns={"year": "asof_year", "ever_ineligible": "ever_ineligible_by_asof"}
)[["country", "product", "asof_year", "ever_ineligible_by_asof"]]

elig = historical.merge(
    eligibility_at_asof,
    on=["country", "product", "asof_year"],
    how="left",
    validate="many_to_one"
)
eligibility_missing = elig["ever_ineligible_by_asof"].isna().sum()
eligibility_violations = (
    elig["ever_ineligible_by_asof"].fillna(True).astype(bool).sum()
)
check(
    "Historical rows satisfy strict RCA3 eligibility",
    eligibility_missing == 0 and eligibility_violations == 0,
    f"missing checks: {eligibility_missing:,}; violations: {eligibility_violations:,}"
)

print("\n=== 6. INDIA POSITIVE LABELS ===")
india_hist = historical[historical["country"] == india]
print(
    india_hist.groupby(["asof_year", "target_year"])["label"]
    .sum().astype(int).to_string()
)

print("\n=== FINAL RESULT ===")
if errors:
    print(f"FAIL: {len(errors)} check(s) need investigation:")
    for error in errors:
        print(" -", error)
else:
    print("All checks passed.")
