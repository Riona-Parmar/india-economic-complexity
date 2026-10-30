
from pathlib import Path
import time

import pandas as pd
import yaml


ROOT = Path.cwd()

with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

PANEL_PATH = ROOT / "data/processed/trade_rca_v1.parquet"
PRODUCTS_PATH = ROOT / "data/processed/products_v1.parquet"
EMERGENCE_PATH = ROOT / "data/processed/labels_v1.parquet"
RECOVERY_PATH = ROOT / "data/processed/recovery_labels_v1.parquet"

REPORT_DIR = ROOT / "outputs/reports"
INTERIM_DIR = ROOT / "data/interim"

REPORT_PATH = REPORT_DIR / "feature_readiness_report.md"
LABEL_COUNTS_PATH = INTERIM_DIR / "feature_audit_label_counts.csv"
FEATURE_COUNTS_PATH = INTERIM_DIR / "feature_audit_feature_coverage.csv"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

start = time.time()

india = str(cfg["india"])
peers = set(map(str, cfg["peers"]))
first_year = int(cfg["first_year"])
last_year = int(cfg["last_year"])
horizon = int(cfg["horizon"])

windows = sorted(
    int(y) for y in cfg["windows"]
    if int(y) + horizon <= last_year
)

feature_columns = [
    "export_value",
    "rca",
    "rca3",
    "m",
    "pci",
    "eci",
    "atlas_distance",
    "atlas_cog",
]

print("FEATURE READINESS AUDIT")
print("-" * 60)

# ------------------------------------------------------------
# 1. Check that all required inputs exist
# ------------------------------------------------------------
required_paths = [
    PANEL_PATH,
    PRODUCTS_PATH,
    EMERGENCE_PATH,
    RECOVERY_PATH,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(f"Required input missing: {path}")
    print(f"FOUND: {path.relative_to(ROOT)}")


# ------------------------------------------------------------
# 2. Load product metadata and check sector coverage
# ------------------------------------------------------------
print("\nChecking product metadata...")

products = pd.read_parquet(PRODUCTS_PATH)

required_product_columns = {
    "product", "name", "hs_section", "sector", "pci_latest"
}
missing_product_columns = required_product_columns - set(products.columns)

if missing_product_columns:
    raise ValueError(
        f"Product metadata missing columns: {sorted(missing_product_columns)}"
    )

if products["product"].duplicated().any():
    raise ValueError("Duplicate product codes in products_v1.parquet.")

print(f"Product metadata rows: {len(products):,}")
print(f"Missing names: {products['name'].isna().sum():,}")
print(f"Missing sectors: {products['sector'].isna().sum():,}")
print(f"Distinct mapped goods sectors: {products['sector'].nunique():,}")
print(
    "Unmapped product codes:",
    products.loc[products["sector"].isna(), "product"].tolist()
)


# ------------------------------------------------------------
# 3. Load only features needed at historical as-of dates
# ------------------------------------------------------------
print("\nLoading panel features...")

panel = pd.read_parquet(
    PANEL_PATH,
    columns=[
        "country",
        "product",
        "year",
        *feature_columns,
    ],
)

panel["country"] = panel["country"].astype(str)
panel["product"] = panel["product"].astype(str)
panel["year"] = panel["year"].astype(int)

if panel.duplicated(["country", "product", "year"]).any():
    raise ValueError("Duplicate country-product-year rows in trade panel.")

# Features are selected only for the historical prediction dates.
asof_panel = panel[panel["year"].isin(windows)].copy()
asof_panel = asof_panel.rename(columns={"year": "asof_year"})

print(f"Full panel rows loaded: {len(panel):,}")
print(f"Rows at configured as-of years: {len(asof_panel):,}")


# ------------------------------------------------------------
# 4. Load both label datasets and audit class balance
# ------------------------------------------------------------
print("\nLoading labels...")

label_frames = []

for pathway, path in [
    ("new_emergence", EMERGENCE_PATH),
    ("recovery", RECOVERY_PATH),
]:
    labels = pd.read_parquet(path)

    required_label_columns = {
        "asof_year", "target_year", "country",
        "product", "label", "split",
    }

    missing = required_label_columns - set(labels.columns)
    if missing:
        raise ValueError(
            f"{path.name} missing columns: {sorted(missing)}"
        )

    # Forecast rows have no observed target and are not evaluation labels.
    labels = labels[labels["split"] != "forecast"].copy()

    labels["pathway"] = pathway
    labels["asof_year"] = labels["asof_year"].astype(int)
    labels["country"] = labels["country"].astype(str)
    labels["product"] = labels["product"].astype(str)

    if labels["label"].isna().any():
        raise ValueError(
            f"Unexpected missing historical labels in {path.name}"
        )

    labels["label"] = labels["label"].astype(int)
    label_frames.append(labels)

all_labels = pd.concat(label_frames, ignore_index=True)

if all_labels.duplicated(
    ["pathway", "asof_year", "country", "product"]
).any():
    raise ValueError("Duplicate label keys within a pathway.")

label_summary = (
    all_labels.groupby(
        ["pathway", "asof_year", "target_year", "split"],
        dropna=False,
        observed=True,
    )
    .agg(
        rows=("label", "size"),
        positives=("label", "sum"),
    )
    .reset_index()
)

label_summary["negatives"] = (
    label_summary["rows"] - label_summary["positives"]
)
label_summary["positive_rate_pct"] = (
    100 * label_summary["positives"] / label_summary["rows"]
)

label_summary.to_csv(LABEL_COUNTS_PATH, index=False)

print("\nLABEL COUNTS")
print(label_summary.to_string(index=False))


# ------------------------------------------------------------
# 5. Join labels to features available at the as-of year
# ------------------------------------------------------------
print("\nChecking as-of feature coverage...")

audit = all_labels.merge(
    asof_panel,
    on=["country", "product", "asof_year"],
    how="left",
    validate="many_to_one",
    indicator=True,
)

audit["feature_row_found"] = audit["_merge"].eq("both")

feature_rows = []

for (pathway, asof_year, split), group in audit.groupby(
    ["pathway", "asof_year", "split"],
    observed=True,
):
    row = {
        "pathway": pathway,
        "asof_year": int(asof_year),
        "split": split,
        "label_rows": len(group),
        "feature_rows_found": int(group["feature_row_found"].sum()),
        "feature_rows_missing": int((~group["feature_row_found"]).sum()),
    }

    for feature in feature_columns:
        row[f"{feature}_missing"] = int(group[feature].isna().sum())
        row[f"{feature}_missing_pct"] = round(
            100 * group[feature].isna().mean(), 3
        )

    feature_rows.append(row)

feature_summary = pd.DataFrame(feature_rows)
feature_summary.to_csv(FEATURE_COUNTS_PATH, index=False)

print("\nFEATURE COVERAGE")
print(
    feature_summary[
        [
            "pathway",
            "asof_year",
            "split",
            "label_rows",
            "feature_rows_found",
            "feature_rows_missing",
        ]
    ].to_string(index=False)
)


# ------------------------------------------------------------
# 6. Check the missingness of each feature overall
# ------------------------------------------------------------
print("\nOVERALL FEATURE MISSINGNESS")

overall_missing = []

for feature in feature_columns:
    missing_count = int(audit[feature].isna().sum())
    overall_missing.append({
        "feature": feature,
        "missing_values": missing_count,
        "total_label_rows": len(audit),
        "missing_pct": round(100 * missing_count / len(audit), 3),
    })

overall_missing = pd.DataFrame(overall_missing)
print(overall_missing.to_string(index=False))


# ------------------------------------------------------------
# 7. Check split definitions and temporal alignment
# ------------------------------------------------------------
print("\nCHECKING SPLITS AND YEARS...")

bad_india_split = (
    audit["country"].eq(india)
    & audit["split"].ne("test_india")
)

bad_peer_split = (
    audit["country"].isin(peers)
    & ~audit["country"].eq(india)
    & audit["split"].ne("test_peer")
)

bad_train_split = (
    ~audit["country"].eq(india)
    & ~audit["country"].isin(peers)
    & audit["split"].ne("train")
)

print("India split violations:", int(bad_india_split.sum()))
print("Peer split violations:", int(bad_peer_split.sum()))
print("Training split violations:", int(bad_train_split.sum()))

bad_horizon = (
    audit["target_year"].astype(int) - audit["asof_year"]
) != horizon

print("Incorrect prediction horizon rows:", int(bad_horizon.sum()))

missing_feature_rows = int((~audit["feature_row_found"]).sum())
print("Label rows without matching as-of panel row:", missing_feature_rows)


# ------------------------------------------------------------
# 8. Create the audit report
# ------------------------------------------------------------
lines = [
    "# Feature Readiness Audit",
    "",
    "## Configuration",
    "",
    f"- Data period: {first_year}–{last_year}",
    f"- Historical as-of years: {windows}",
    f"- Prediction horizon: {horizon} years",
    f"- India evaluation country: {india}",
    f"- Peer evaluation countries: {', '.join(sorted(peers))}",
    "",
    "## Product metadata",
    "",
    f"- Product rows: {len(products):,}",
    f"- Missing names: {products['name'].isna().sum():,}",
    f"- Missing sectors: {products['sector'].isna().sum():,}",
    f"- Distinct goods sectors represented: {products['sector'].nunique():,}",
    "- Software and other commercial services are not covered by this merchandise panel.",
    "",
    "## Label balance",
    "",
    label_summary.to_markdown(index=False),
    "",
    "## As-of feature coverage",
    "",
    feature_summary.to_markdown(index=False),
    "",
    "## Overall feature missingness",
    "",
    overall_missing.to_markdown(index=False),
    "",
    "## Validation checks",
    "",
    f"- India split violations: {int(bad_india_split.sum())}",
    f"- Peer split violations: {int(bad_peer_split.sum())}",
    f"- Training split violations: {int(bad_train_split.sum())}",
    f"- Incorrect prediction horizon rows: {int(bad_horizon.sum())}",
    f"- Labels without matching as-of panel row: {missing_feature_rows}",
    "",
    "## Interpretation and safeguards",
    "",
    "- Features are joined using country, product, and as-of year.",
    "- Target-year values are not joined as predictive features.",
    "- Missing feature values must be handled using training-only imputation rules.",
    "- The label eligibility rules differ between new emergence and recovery; compare metrics within each pathway.",
    "- India-only metrics may be unstable because of small positive counts in some windows.",
    "- ECI is a proxy based on the current panel construction, not an official national ECI series.",
    "- Atlas source RCA and calculated Balassa RCA have a known unresolved discrepancy.",
    "- This is a readiness audit, not evidence of predictive performance or causal effects.",
    "",
]

REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

print("\nAUDIT COMPLETE")
print(f"Report: {REPORT_PATH}")
print(f"Label summary CSV: {LABEL_COUNTS_PATH}")
print(f"Feature coverage CSV: {FEATURE_COUNTS_PATH}")
print(f"Elapsed seconds: {time.time() - start:.1f}")
