from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(".")
TRADE_PATH = ROOT / "data/processed/trade_rca_v1.parquet"
LABEL_PATH = ROOT / "data/processed/labels_v1.parquet"
SOURCE_PATH = ROOT / "data/interim/trade_rca_source_comparison_v1.parquet"
OUT_DIR = ROOT / "outputs/reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOW = 0.25
HIGH = 1.0
FIRST_YEAR = 1995

print("Loading data...")
trade = pd.read_parquet(TRADE_PATH)
labels = pd.read_parquet(LABEL_PATH)
source = pd.read_parquet(SOURCE_PATH)

required = {"country", "product", "year", "export_rca"}
if not required.issubset(source.columns):
    raise ValueError(f"Source audit file is missing columns: {required - set(source.columns)}")

# Only compare countries included in the validated project trade panel.
countries = set(trade["country"].unique())
source = source[source["country"].isin(countries)].copy()

print("Project trade rows:", f"{len(trade):,}")
print("Source rows for project countries:", f"{len(source):,}")

# Join the source's annual RCA to the complete project grid.
# Rows absent from the source file are treated as zero source RCA,
# consistent with a missing country-product-year trade observation.
# For source rows with an explicitly missing export_rca, use calculated
# RCA as a fallback and report that assumption separately.
source = source[["country", "product", "year", "export_rca"]].copy()
source["source_row_present"] = True

panel = trade[
    ["country", "product", "year", "rca", "rca3"]
].merge(
    source,
    on=["country", "product", "year"],
    how="left",
    validate="one_to_one"
)

absent_source_rows = int(panel["source_row_present"].isna().sum())
explicit_missing = int(
    (panel["source_row_present"].notna() & panel["export_rca"].isna()).sum()
)

panel["source_rca_sensitivity"] = panel["export_rca"]
panel.loc[panel["source_row_present"].isna(), "source_rca_sensitivity"] = 0.0
panel.loc[
    panel["source_row_present"].notna() & panel["export_rca"].isna(),
    "source_rca_sensitivity"
] = panel.loc[
    panel["source_row_present"].notna() & panel["export_rca"].isna(),
    "rca"
]

print("Grid rows absent from source file (assumed zero RCA):", f"{absent_source_rows:,}")
print("Explicitly missing source RCA values (fallback to calculated RCA):", f"{explicit_missing:,}")

panel = panel.sort_values(["country", "product", "year"]).reset_index(drop=True)

# Match the project's RCA3 convention: rolling mean of up to three years,
# with min_periods=1.

panel["source_rca3_sensitivity"] = (
    panel.groupby(["country", "product"], sort=False)["source_rca_sensitivity"]
    .rolling(window=3, min_periods=1)
    .mean()
    .reset_index(level=[0, 1], drop=True)
    .reindex(panel.index)
)


print("\n=== A. ANNUAL RCA THRESHOLD DISAGREEMENTS ===")
for threshold in [LOW, HIGH]:
    a = panel["source_rca_sensitivity"].ge(threshold)
    b = panel["rca"].ge(threshold)
    disagree = a.ne(b)
    print(
        f"Threshold {threshold}: {int(disagree.sum()):,} / {len(panel):,} "
        f"rows disagree ({disagree.mean():.4%})"
    )

print("\n=== B. RCA3 THRESHOLD DISAGREEMENTS ===")
for threshold in [LOW, HIGH]:
    a = panel["source_rca3_sensitivity"].ge(threshold)
    b = panel["rca3"].ge(threshold)
    disagree = a.ne(b)
    print(
        f"Threshold {threshold}: {int(disagree.sum()):,} / {len(panel):,} "
        f"rows disagree ({disagree.mean():.4%})"
    )

# Construct source-RCA sensitivity eligibility: RCA3 must be strictly
# below 0.25 in every year from 1995 through the as-of year.
panel["source_ineligible"] = panel["source_rca3_sensitivity"].ge(LOW)
panel["calc_ineligible"] = panel["rca3"].ge(LOW)

panel["source_ever_ineligible"] = (
    panel.groupby(["country", "product"])["source_ineligible"].cummax()
)
panel["calc_ever_ineligible"] = (
    panel.groupby(["country", "product"])["calc_ineligible"].cummax()
)

historical = labels[labels["label"].notna()].copy()
panel_for_asof = panel.rename(columns={"year": "asof_year"})

asof_cols = [
    "country", "product", "asof_year",
    "source_rca3_sensitivity", "rca3",
    "source_ever_ineligible", "calc_ever_ineligible"
]
asof = historical.merge(
    panel_for_asof[asof_cols],
    on=["country", "product", "asof_year"],
    how="left",
    validate="many_to_one"
)

target = panel[[
    "country", "product", "year",
    "source_rca3_sensitivity", "rca3"
]].rename(columns={
    "year": "target_year",
    "source_rca3_sensitivity": "source_target_rca3",
    "rca3": "calculated_target_rca3"
})

comparison = asof.merge(
    target,
    on=["country", "product", "target_year"],
    how="left",
    validate="many_to_one"
)

# Existing calculated label: eligible at as-of year AND target RCA3 >= 1.
comparison["source_eligible"] = ~comparison["source_ever_ineligible"].astype(bool)
comparison["calculated_eligible"] = ~comparison["calc_ever_ineligible"].astype(bool)

comparison["source_sensitivity_label"] = (
    comparison["source_eligible"]
    & comparison["source_target_rca3"].ge(HIGH)
).astype("int8")

comparison["calculated_rebuilt_label"] = (
    comparison["calculated_eligible"]
    & comparison["calculated_target_rca3"].ge(HIGH)
).astype("int8")

print("\n=== C. HISTORICAL LABEL SENSITIVITY ===")
eligibility_disagreements = comparison["source_eligible"].ne(
    comparison["calculated_eligible"]
)
label_disagreements = comparison["source_sensitivity_label"].ne(
    comparison["calculated_rebuilt_label"]
)
existing_label_disagreements = comparison["source_sensitivity_label"].ne(
    comparison["label"].astype("int8")
)

print(
    "Eligibility status disagreements:",
    f"{int(eligibility_disagreements.sum()):,} / {len(comparison):,}"
)
print(
    "Source-vs-calculated rebuilt label disagreements:",
    f"{int(label_disagreements.sum()):,} / {len(comparison):,}"
)
print(
    "Sensitivity labels vs existing project labels:",
    f"{int(existing_label_disagreements.sum()):,} / {len(comparison):,}"
)

print("\nDisagreements by as-of year:")
summary = comparison.groupby("asof_year").agg(
    historical_rows=("label", "size"),
    eligibility_disagreements=("source_eligible", lambda x: 0),
)
summary["eligibility_disagreements"] = comparison.groupby("asof_year").apply(
    lambda g: int(g["source_eligible"].ne(g["calculated_eligible"]).sum()),
    include_groups=False
)
summary["label_disagreements"] = comparison.groupby("asof_year").apply(
    lambda g: int(g["source_sensitivity_label"].ne(g["calculated_rebuilt_label"]).sum()),
    include_groups=False
)
print(summary.to_string())

# Save row-level discrepancies for review.
detail_cols = [
    "asof_year", "target_year", "country", "product", "label",
    "source_eligible", "calculated_eligible",
    "source_target_rca3", "calculated_target_rca3",
    "source_sensitivity_label", "calculated_rebuilt_label"
]
details = comparison[
    eligibility_disagreements | label_disagreements | existing_label_disagreements
][detail_cols].copy()

detail_path = ROOT / "data/interim/rca_sensitivity_label_disagreements.csv"
details.to_csv(detail_path, index=False)

# Save a concise summary report.
report_path = OUT_DIR / "rca_sensitivity_audit_v1.md"
report = f"""# RCA Sensitivity Audit

## Scope

Compared the project's calculated standard Balassa RCA and rolling RCA3
against the Atlas source `export_rca` field, for the countries in the project
trade panel.

## Assumptions and limitations

- The source comparison file has no row for some country-product-year keys
  in the complete project grid. These absent rows were assigned source RCA = 0
  for this sensitivity exercise.
- Explicitly missing source `export_rca` values were replaced with the
  calculated RCA for those records. This affects {explicit_missing:,} rows.
- These are sensitivity assumptions, not proof of the Atlas source's intended
  missing-value semantics.
- RCA3 uses a rolling mean of up to three years, with `min_periods=1`.
- The sensitivity labels apply the project's strict historical eligibility
  rule and target RCA3 >= {HIGH:.1f}.

## Results

- Project trade-panel rows: {len(panel):,}
- Source rows absent from the project grid: {absent_source_rows:,}
- Explicitly missing source RCA values using calculated-RCA fallback: {explicit_missing:,}
- Historical labeled rows compared: {len(comparison):,}
- Eligibility status disagreements: {int(eligibility_disagreements.sum()):,}
- Source-vs-calculated rebuilt label disagreements: {int(label_disagreements.sum()):,}
- Sensitivity labels vs existing project labels: {int(existing_label_disagreements.sum()):,}

See `data/interim/rca_sensitivity_label_disagreements.csv` for affected rows.
"""
report_path.write_text(report, encoding="utf-8")

print("\nSaved detail:", detail_path)
print("Saved report:", report_path)
print("\nIMPORTANT: Review the missing-value assumptions before interpreting results.")
