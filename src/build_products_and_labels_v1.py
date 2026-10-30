
from pathlib import Path
import time

import numpy as np
import pandas as pd
import yaml


ROOT = Path.cwd()
CONFIG_PATH = ROOT / "config.yaml"
PANEL_PATH = ROOT / "data/processed/trade_rca_v1.parquet"
MAPPING_PATH = ROOT / "data/interim/hs92_hs4_sector_mapping_v2.csv"
OUTPUT_DIR = ROOT / "data/processed"
REPORT_DIR = ROOT / "outputs/reports"

PRODUCTS_PATH = OUTPUT_DIR / "products_v1.parquet"
LABELS_PATH = OUTPUT_DIR / "labels_v1.parquet"
REPORT_PATH = REPORT_DIR / "data_report.md"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

start = time.time()

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

first_year = int(cfg["first_year"])
last_year = int(cfg["last_year"])
horizon = int(cfg["horizon"])
india = str(cfg["india"])
peers = set(map(str, cfg["peers"]))
windows = sorted(set(map(int, cfg["windows"])))
low_threshold = float(cfg["rca_low"])
high_threshold = float(cfg["rca_high"])

print("Loading trade panel...")
panel = pd.read_parquet(
    PANEL_PATH,
    columns=["country", "product", "year", "rca3", "pci"],
)

panel["country"] = panel["country"].astype(str)
panel["product"] = (
    panel["product"].astype(str).str.strip().str.upper().str.zfill(4)
)
panel["year"] = panel["year"].astype(int)

if panel.duplicated(["country", "product", "year"]).any():
    raise ValueError("Duplicate country-product-year keys in panel.")

products_in_panel = set(panel["product"].unique())


# ------------------------------------------------------------
# 1. Build products_v1.parquet
# ------------------------------------------------------------
print("Building product metadata...")

mapping = pd.read_csv(
    MAPPING_PATH,
    dtype={"product_hs92_code": "string"},
    low_memory=False,
)

mapping["product"] = (
    mapping["product_hs92_code"]
    .astype("string")
    .str.strip()
    .str.upper()
    .str.zfill(4)
)

required_mapping_columns = {
    "product", "product_description", "sector_id", "sector_name"
}
missing_cols = required_mapping_columns - set(mapping.columns)
if missing_cols:
    raise ValueError(f"Mapping file missing columns: {sorted(missing_cols)}")

if mapping["product"].duplicated().any():
    raise ValueError("Duplicate product codes in sector mapping.")

# Latest available PCI for each product, based on all retained countries.
latest_year = panel["year"].max()
latest_pci = (
    panel.loc[panel["year"] == latest_year]
    .groupby("product", as_index=False)["pci"]
    .median()
    .rename(columns={"pci": "pci_latest"})
)

products = mapping[
    ["product", "product_description", "hs_chapter", "sector_name"]
].rename(
    columns={
        "product_description": "name",
        "hs_chapter": "hs_section",
        "sector_name": "sector",
    }
)

products = products.merge(
    latest_pci,
    on="product",
    how="left",
    validate="one_to_one",
)

# Keep the entire panel's product universe, including XXXX.
# Missing metadata stays missing; never fabricate a classification.
products = pd.DataFrame({"product": sorted(products_in_panel)}).merge(
    products,
    on="product",
    how="left",
    validate="one_to_one",
)

products = products[
    ["product", "name", "hs_section", "sector", "pci_latest"]
]

products.to_parquet(PRODUCTS_PATH, index=False)

print(f"Saved: {PRODUCTS_PATH}")
print(f"Product rows: {len(products):,}")
print(f"Missing product names: {products['name'].isna().sum()}")
print(f"Missing sectors: {products['sector'].isna().sum()}")


# ------------------------------------------------------------
# 2. Build historical emergence labels and 2024 forecast rows
# ------------------------------------------------------------
print("\nPreparing label windows...")

# Use the configured historical windows. Only create a label when
# the target year exists in the panel.
historical_windows = [
    y for y in windows
    if y + horizon <= last_year
]

if not historical_windows:
    raise ValueError("No historical windows have a target year in the panel.")

# The execution plan's eligibility condition is strict:
# rca3 must be below rca_low in every year from first_year to as-of.
# A country-product becomes eligible only after it has remained
# below the threshold throughout its complete history to that year.
wide = panel.pivot(
    index=["country", "product"],
    columns="year",
    values="rca3",
)

available_years = sorted(int(y) for y in wide.columns)
expected_years = list(range(first_year, last_year + 1))

if available_years != expected_years:
    raise ValueError(
        "Panel does not contain every configured year. "
        f"Found {len(available_years)} years; expected {len(expected_years)}."
    )

label_frames = []

for asof in historical_windows:
    target = asof + horizon
    print(f"Preparing {asof} -> {target}...")

    history_years = list(range(first_year, asof + 1))
    history = wide[history_years]

    eligible = history.lt(low_threshold).all(axis=1)

    target_values = wide[target]
    eligible_index = wide.index[eligible]

    if len(eligible_index) == 0:
        print(f"  No eligible pairs for {asof}; skipping.")
        continue

    # Construct rows only for eligible country-product pairs.
    block = eligible_index.to_frame(index=False)
    block["asof_year"] = asof
    block["target_year"] = target
    block["label"] = (
        target_values.reindex(eligible_index).to_numpy() >= high_threshold
    ).astype("int8")

    block["split"] = np.select(
        [
            block["country"].eq(india),
            block["country"].isin(peers),
        ],
        [
            "test_india",
            "test_peer",
        ],
        default="train",
    )

    label_frames.append(
        block[
            ["asof_year", "target_year", "country", "product", "label", "split"]
        ]
    )

# The forecast snapshot uses the same strict eligibility rule through 2024.
forecast_asof = last_year
forecast_history = wide[list(range(first_year, forecast_asof + 1))]
forecast_eligible = forecast_history.lt(low_threshold).all(axis=1)
forecast_index = wide.index[forecast_eligible]

print(f"Preparing {forecast_asof} forecast snapshot...")

if len(forecast_index):
    forecast = forecast_index.to_frame(index=False)
    forecast["asof_year"] = forecast_asof
    forecast["target_year"] = pd.Series(
        pd.NA, index=forecast.index, dtype="Int64"
    )
    forecast["label"] = pd.Series(
        pd.NA, index=forecast.index, dtype="Int64"
    )
    forecast["split"] = "forecast"

    label_frames.append(
        forecast[
            ["asof_year", "target_year", "country", "product", "label", "split"]
        ]
    )

if not label_frames:
    raise ValueError("No historical labels or forecast candidates were generated.")

labels = pd.concat(label_frames, ignore_index=True)

# Preserve nullable target and label fields because forecast rows have no outcome.
labels["asof_year"] = labels["asof_year"].astype("int16")
labels["target_year"] = labels["target_year"].astype("Int64")
labels["label"] = labels["label"].astype("Int8")

if labels.duplicated(
    ["asof_year", "country", "product"]
).any():
    raise ValueError("Duplicate as-of/country/product label rows.")

# Validate historical eligibility directly against the source panel.
historical = labels[labels["split"] != "forecast"]

for asof in historical["asof_year"].unique():
    subset = historical[historical["asof_year"] == asof]
    hist = wide[list(range(first_year, int(asof) + 1))]
    idx = pd.MultiIndex.from_frame(subset[["country", "product"]])
    values = hist.reindex(idx)

    if not values.lt(low_threshold).all(axis=1).all():
        raise ValueError(
            f"Eligibility validation failed for as-of year {asof}."
        )

labels.to_parquet(LABELS_PATH, index=False)

print(f"Saved: {LABELS_PATH}")
print(f"Total label rows, including forecast: {len(labels):,}")


# ------------------------------------------------------------
# 3. Write data report
# ------------------------------------------------------------
print("\nWriting report...")

lines = [
    "# Trade RCA and Emergence Label Data Report",
    "",
    "## Configuration",
    "",
    f"- Data years: {first_year}–{last_year}",
    f"- RCA3 eligibility threshold: below {low_threshold}",
    f"- Emergence threshold: at least {high_threshold}",
    f"- Prediction horizon: {horizon} years",
    f"- Historical windows configured: {windows}",
    f"- India test country: {india}",
    f"- Peer test countries: {', '.join(sorted(peers))}",
    "",
    "## Product metadata",
    "",
    f"- Products: {len(products):,}",
    f"- Missing names: {products['name'].isna().sum():,}",
    f"- Missing sectors: {products['sector'].isna().sum():,}",
    "- Special code XXXX remains unmapped; it is not assigned an invented sector.",
    "",
    "## Label counts",
    "",
    "| As-of year | Target year | Split | Rows | Positive labels |",
    "|---:|---:|---|---:|---:|",
]

count_table = (
    labels.groupby(
        ["asof_year", "target_year", "split"],
        dropna=False,
        observed=True,
    )
    .agg(
        rows=("product", "size"),
        positives=("label", lambda s: int(s.eq(1).sum())),
    )
    .reset_index()
)

for _, row in count_table.iterrows():
    target = (
        "Forecast"
        if pd.isna(row["target_year"])
        else str(int(row["target_year"]))
    )
    lines.append(
        f"| {int(row['asof_year'])} | {target} | {row['split']} | "
        f"{int(row['rows']):,} | {int(row['positives']):,} |"
    )

lines.extend(
    [
        "",
        "## India positive labels by historical window",
        "",
        "| As-of year | Target year | Eligible India pairs | Positive labels |",
        "|---:|---:|---:|---:|",
    ]
)

india_labels = labels[labels["country"] == india]
for asof in historical_windows:
    subset = india_labels[india_labels["asof_year"] == asof]
    lines.append(
        f"| {asof} | {asof + horizon} | {len(subset):,} | "
        f"{int(subset['label'].eq(1).sum()):,} |"
    )

lines.extend(
    [
        "",
        "## Interpretation cautions",
        "",
        "- Labels represent transitions in the calculated RCA3 measure, not causal evidence.",
        "- Historical eligibility requires RCA3 to remain below the threshold in every year since the first data year.",
        "- Forecast rows have no observed target outcomes and must not be treated as labeled training examples.",
        "- A small number of India positives limits the reliability of India-only evaluation metrics.",
        "- The Atlas source RCA differs from the independently calculated Balassa RCA for some records; this remains an audit item.",
        "- Merchandise exports and RCA are not direct measures of GNP or domestic value added.",
        "",
    ]
)

REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

print(f"Saved: {REPORT_PATH}")
print("\nLABEL SUMMARY")
print(count_table.to_string(index=False))
print(f"\nElapsed seconds: {time.time() - start:.1f}")
