
from pathlib import Path
import time

import numpy as np
import pandas as pd
import yaml


ROOT = Path.cwd()

with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

PANEL_PATH = ROOT / "data/processed/trade_rca_v1.parquet"
PRODUCTS_PATH = ROOT / "data/processed/products_v1.parquet"
EMERGENCE_PATH = ROOT / "data/processed/labels_v1.parquet"
RECOVERY_PATH = ROOT / "data/processed/recovery_labels_v1.parquet"

OUTPUT_DIR = ROOT / "data/processed"
REPORT_DIR = ROOT / "outputs/reports"

OUTPUT_PATH = OUTPUT_DIR / "model_features_v1.parquet"
DICTIONARY_PATH = REPORT_DIR / "feature_dictionary_v1.csv"
REPORT_PATH = REPORT_DIR / "model_feature_build_report.md"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

start = time.time()

first_year = int(cfg["first_year"])
last_year = int(cfg["last_year"])
horizon = int(cfg["horizon"])
windows = sorted(set(map(int, cfg["windows"])))
india = str(cfg["india"])
peers = set(map(str, cfg["peers"]))

windows = [
    year for year in windows
    if first_year <= year and year + horizon <= last_year
]

print("MODEL FEATURE BUILD")
print("-" * 60)
print("Historical as-of years:", windows)


# ------------------------------------------------------------
# 1. Load historical labels, excluding forecast-only rows
# ------------------------------------------------------------
print("\nLoading labels...")

label_frames = []

for pathway, path in [
    ("new_emergence", EMERGENCE_PATH),
    ("recovery", RECOVERY_PATH),
]:
    labels = pd.read_parquet(path)

    labels = labels[labels["split"] != "forecast"].copy()

    labels["pathway"] = pathway
    labels["country"] = labels["country"].astype(str)
    labels["product"] = (
        labels["product"].astype(str).str.strip().str.upper().str.zfill(4)
    )
    labels["asof_year"] = labels["asof_year"].astype(int)
    labels["target_year"] = labels["target_year"].astype(int)
    labels["label"] = labels["label"].astype(int)

    label_frames.append(
        labels[
            [
                "pathway",
                "asof_year",
                "target_year",
                "country",
                "product",
                "label",
                "split",
            ]
        ]
    )

labels = pd.concat(label_frames, ignore_index=True)

if labels.duplicated(
    ["pathway", "asof_year", "country", "product"]
).any():
    raise ValueError("Duplicate label keys found.")

if not labels["asof_year"].isin(windows).all():
    raise ValueError("Unexpected as-of year found in labels.")

if not (labels["target_year"] - labels["asof_year"]).eq(horizon).all():
    raise ValueError("Some labels have an unexpected prediction horizon.")

print(f"Historical label rows: {len(labels):,}")


# ------------------------------------------------------------
# 2. Load the panel and create as-of and lagged features
# ------------------------------------------------------------
print("\nLoading historical panel features...")

panel_columns = [
    "country",
    "product",
    "year",
    "export_value",
    "rca",
    "rca3",
    "m",
    "pci",
    "eci",
    "atlas_distance",
    "atlas_cog",
]

panel = pd.read_parquet(PANEL_PATH, columns=panel_columns)

panel["country"] = panel["country"].astype(str)
panel["product"] = (
    panel["product"].astype(str).str.strip().str.upper().str.zfill(4)
)
panel["year"] = panel["year"].astype(int)

if panel.duplicated(["country", "product", "year"]).any():
    raise ValueError("Duplicate country-product-year keys in panel.")

# Only years needed for as-of features and three-year lags.
needed_years = set()
for year in windows:
    needed_years.update([year, year - 1, year - 3])

needed_years = {
    year for year in needed_years
    if first_year <= year <= last_year
}

panel = panel[panel["year"].isin(needed_years)].copy()

base_features = [
    "export_value",
    "rca",
    "rca3",
    "m",
    "pci",
    "eci",
    "atlas_distance",
    "atlas_cog",
]

key_columns = ["country", "product"]

def year_snapshot(year, suffix):
    """Return country-product values from one historical year."""
    snap = panel.loc[panel["year"] == year, key_columns + base_features].copy()
    snap = snap.rename(
        columns={col: f"{col}_{suffix}" for col in base_features}
    )
    return snap


feature_blocks = []

for asof in windows:
    print(f"Building features for as-of year {asof}...")

    current = year_snapshot(asof, "current")
    previous = year_snapshot(asof - 1, "lag1")
    three_years_ago = year_snapshot(asof - 3, "lag3")

    block = current.merge(
        previous,
        on=key_columns,
        how="left",
        validate="one_to_one",
    )

    block = block.merge(
        three_years_ago,
        on=key_columns,
        how="left",
        validate="one_to_one",
    )

    block["asof_year"] = asof

    # All calculations use only the as-of year or earlier.
    # log1p makes zero-export observations valid.
    block["log_export_value"] = np.log1p(
        block["export_value_current"].clip(lower=0)
    )

    block["log_export_value_lag3"] = np.log1p(
        block["export_value_lag3"].clip(lower=0)
    )

    block["log_export_change_3y"] = (
        block["log_export_value"] - block["log_export_value_lag3"]
    )

    block["rca3_change_3y"] = (
        block["rca3_current"] - block["rca3_lag3"]
    )

    block["rca3_lag1"] = block["rca3_lag1"]
    block["export_value_lag1"] = block["export_value_lag1"]

    block["log_export_change_1y"] = (
        np.log1p(block["export_value_current"].clip(lower=0))
        - np.log1p(block["export_value_lag1"].clip(lower=0))
    )

    # At the beginning of the data period, a lag may not exist.
    # We do not invent historical observations; use explicit
    # availability indicators and neutral numeric placeholders.
    block["lag1_available"] = block["rca3_lag1"].notna().astype("int8")
    block["lag3_available"] = block["rca3_lag3"].notna().astype("int8")

    lag_numeric_columns = [
        "rca_lag1",
        "rca3_lag1",
        "m_lag1",
        "export_value_lag1",
        "pci_lag1",
        "eci_lag1",
        "atlas_distance_lag1",
        "atlas_cog_lag1",
        "rca_lag3",
        "rca3_lag3",
        "m_lag3",
        "export_value_lag3",
        "pci_lag3",
        "eci_lag3",
        "atlas_distance_lag3",
        "atlas_cog_lag3",
        "log_export_value_lag3",
        "log_export_change_3y",
        "rca3_change_3y",
        "log_export_change_1y",
    ]

    for col in lag_numeric_columns:
        if col in block.columns:
            block[col] = block[col].fillna(0)

    feature_blocks.append(block)

feature_panel = pd.concat(feature_blocks, ignore_index=True)

del panel, feature_blocks
print(f"As-of feature rows: {len(feature_panel):,}")


# ------------------------------------------------------------
# 3. Attach product metadata and labels
# ------------------------------------------------------------
print("\nJoining product metadata and labels...")

products = pd.read_parquet(PRODUCTS_PATH)
products["product"] = (
    products["product"].astype(str).str.strip().str.upper().str.zfill(4)
)

if products["product"].duplicated().any():
    raise ValueError("Duplicate product metadata keys.")

feature_panel = feature_panel.merge(
    products[["product", "name", "hs_section", "sector"]],
    on="product",
    how="left",
    validate="many_to_one",
)

features = labels.merge(
    feature_panel,
    on=["country", "product", "asof_year"],
    how="left",
    validate="many_to_one",
    indicator=True,
)

missing_feature_rows = int(features["_merge"].ne("both").sum())
print("Labels without matching feature rows:", missing_feature_rows)

if missing_feature_rows:
    raise ValueError("Some labels have no matching as-of feature row.")

features = features.drop(columns=["_merge"])

# Save an indicator rather than assigning a made-up sector to XXXX.
features["sector_missing"] = features["sector"].isna().astype("int8")


# ------------------------------------------------------------
# 4. Define the model feature list
# ------------------------------------------------------------
model_feature_columns = [
    "export_value_current",
    "log_export_value",
    "rca_current",
    "rca3_current",
    "m_current",
    "pci_current",
    "eci_current",
    "atlas_distance_current",
    "atlas_cog_current",
    "export_value_lag1",
    "rca_lag1",
    "rca3_lag1",
    "m_lag1",
    "rca_lag3",
    "rca3_lag3",
    "m_lag3",
    "log_export_change_1y",
    "log_export_change_3y",
    "rca3_change_3y",
    "lag1_available",
    "lag3_available",
]

metadata_columns = [
    "pathway",
    "asof_year",
    "target_year",
    "country",
    "product",
    "label",
    "split",
    "name",
    "hs_section",
    "sector",
    "sector_missing",
]

output_columns = metadata_columns + model_feature_columns

missing_columns = set(output_columns) - set(features.columns)
if missing_columns:
    raise ValueError(
        f"Output columns missing from assembled features: {sorted(missing_columns)}"
    )

features = features[output_columns]

# Check numeric features. Lag fields have explicit missing-history indicators.
for col in model_feature_columns:
    if features[col].isna().any():
        raise ValueError(
            f"Unexpected missing values in model feature {col!r}."
        )

if features.duplicated(
    ["pathway", "asof_year", "country", "product"]
).any():
    raise ValueError("Duplicate feature rows detected.")

if not features["target_year"].sub(features["asof_year"]).eq(horizon).all():
    raise ValueError("Feature rows have an incorrect prediction horizon.")

features.to_parquet(OUTPUT_PATH, index=False)
print(f"Saved feature dataset: {OUTPUT_PATH}")


# ------------------------------------------------------------
# 5. Save a feature dictionary
# ------------------------------------------------------------
dictionary = [
    ("export_value_current", "Current-year exports", "USD", "As-of year"),
    ("log_export_value", "Natural log of 1 + current exports", "log(USD+1)", "As-of year"),
    ("rca_current", "Calculated Balassa RCA", "Ratio", "As-of year"),
    ("rca3_current", "Three-year rolling average RCA", "Ratio", "As-of year"),
    ("m_current", "Binary indicator: RCA3 >= 1", "0/1", "As-of year"),
    ("pci_current", "Product complexity index", "Atlas index", "As-of year"),
    ("eci_current", "Country ECI proxy", "Mean PCI proxy", "As-of year"),
    ("atlas_distance_current", "Atlas distance feature", "Atlas index", "As-of year"),
    ("atlas_cog_current", "Atlas COG feature", "Atlas index", "As-of year"),
    ("export_value_lag1", "Exports one year before as-of year", "USD", "As-of year - 1"),
    ("rca_lag1", "Calculated RCA one year before as-of year", "Ratio", "As-of year - 1"),
    ("rca3_lag1", "RCA3 one year before as-of year", "Ratio", "As-of year - 1"),
    ("m_lag1", "Binary RCA3 specialization indicator one year earlier", "0/1", "As-of year - 1"),
    ("rca_lag3", "Calculated RCA three years before as-of year", "Ratio", "As-of year - 3"),
    ("rca3_lag3", "RCA3 three years before as-of year", "Ratio", "As-of year - 3"),
    ("m_lag3", "Binary RCA3 specialization indicator three years earlier", "0/1", "As-of year - 3"),
    ("log_export_change_1y", "Difference in log(1 + exports) over one year", "Log difference", "As-of year and prior year"),
    ("log_export_change_3y", "Difference in log(1 + exports) over three years", "Log difference", "As-of year and three years earlier"),
    ("rca3_change_3y", "Change in RCA3 over three years", "RCA difference", "As-of year and three years earlier"),
    ("lag1_available", "Whether one-year lag exists", "0/1", "Historical availability"),
    ("lag3_available", "Whether three-year lag exists", "0/1", "Historical availability"),
    ("name", "Product description", "Text", "Static metadata"),
    ("hs_section", "HS chapter field in project mapping", "Code", "Static metadata"),
    ("sector", "Project-defined goods sector", "Category", "Static metadata"),
    ("sector_missing", "Whether sector metadata is missing", "0/1", "Static metadata"),
]

dictionary_df = pd.DataFrame(
    dictionary,
    columns=["feature", "definition", "units", "availability"],
)
dictionary_df.to_csv(DICTIONARY_PATH, index=False)


# ------------------------------------------------------------
# 6. Write summary report
# ------------------------------------------------------------
summary = (
    features.groupby(["pathway", "asof_year", "split"], observed=True)
    .agg(
        rows=("label", "size"),
        positives=("label", "sum"),
    )
    .reset_index()
)
summary["negatives"] = summary["rows"] - summary["positives"]

lines = [
    "# Model Feature Build Report",
    "",
    f"- Output rows: {len(features):,}",
    f"- Output columns: {len(features.columns):,}",
    f"- Historical windows: {windows}",
    f"- Prediction horizon: {horizon} years",
    f"- India: {india}",
    f"- Peers: {', '.join(sorted(peers))}",
    "",
    "## Label counts",
    "",
    summary.to_string(index=False),
    "",
    "## Feature columns",
    "",
    ", ".join(model_feature_columns),
    "",
    "## Leakage safeguards",
    "",
    "- Each label is joined to features for its as-of year only.",
    "- Lag features use the as-of year or earlier.",
    "- Target-year RCA, exports and other target-year variables are not model features.",
    "- Product names and sector classifications are static metadata.",
    "- Labels from new emergence and recovery remain separated by the pathway column.",
    "",
    "## Important limitations",
    "",
    "- Lag values unavailable at the beginning of the dataset are set to zero and accompanied by availability indicators.",
    "- The current ECI variable is a project proxy, not an official ECI series.",
    "- Atlas source RCA and calculated Balassa RCA have an unresolved discrepancy.",
    "- The product mapping covers 20 merchandise sectors; software and other commercial services need a separate dataset.",
    "- This build prepares features but does not train or evaluate models.",
    "",
]

REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

print(f"Feature dictionary: {DICTIONARY_PATH}")
print(f"Build report: {REPORT_PATH}")
print("\nFEATURE DATASET SUMMARY")
print("Rows:", f"{len(features):,}")
print("Columns:", len(features.columns))
print("Missing numeric model features:", int(features[model_feature_columns].isna().sum().sum()))
print("\nLABEL SUMMARY")
print(summary.to_string(index=False))
print(f"\nElapsed seconds: {time.time() - start:.1f}")
