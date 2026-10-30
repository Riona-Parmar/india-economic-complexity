
from pathlib import Path
import time

import numpy as np
import pandas as pd
import yaml


ROOT = Path.cwd()
with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

PANEL_PATH = ROOT / "data/processed/trade_rca_v1.parquet"
OUTPUT_PATH = ROOT / "data/processed/recovery_labels_v1.parquet"
REPORT_PATH = ROOT / "outputs/reports/recovery_label_report.md"

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

first_year = int(cfg["first_year"])
last_year = int(cfg["last_year"])
horizon = int(cfg["horizon"])
windows = sorted(set(map(int, cfg["windows"])))
india = str(cfg["india"])
peers = set(map(str, cfg["peers"]))
recovery_threshold = float(cfg["rca_high"])

start = time.time()

print("Loading RCA3 panel...")
df = pd.read_parquet(
    PANEL_PATH,
    columns=["country", "product", "year", "rca3"],
)

df["country"] = df["country"].astype(str)
df["product"] = df["product"].astype(str)
df["year"] = df["year"].astype(int)

if df.duplicated(["country", "product", "year"]).any():
    raise ValueError("Duplicate country-product-year keys.")

# Wide format lets us evaluate past and future states without
# using any information from after the prediction date as a feature.
wide = df.pivot(
    index=["country", "product"],
    columns="year",
    values="rca3",
)

expected_years = list(range(first_year, last_year + 1))
if list(map(int, wide.columns)) != expected_years:
    raise ValueError("The panel does not contain the expected full year range.")

eligible_windows = [
    year for year in windows
    if year >= first_year and year + horizon <= last_year
]

frames = []

for asof in eligible_windows:
    target = asof + horizon
    print(f"Building recovery labels: {asof} -> {target}")

    # A product must have had comparative advantage BEFORE the
    # as-of year, and must be below the threshold at the as-of year.
    previous_years = list(range(first_year, asof))
    if not previous_years:
        print("  No prior years available; skipping.")
        continue

    previously_specialized = wide[previous_years].ge(
        recovery_threshold
    ).any(axis=1)

    currently_below_threshold = wide[asof].lt(
        recovery_threshold
    )

    eligible = previously_specialized & currently_below_threshold
    eligible_index = wide.index[eligible]

    if len(eligible_index) == 0:
        print("  No eligible pairs.")
        continue

    block = eligible_index.to_frame(index=False)
    block["asof_year"] = asof
    block["target_year"] = target

    target_rca3 = wide[target].reindex(eligible_index).to_numpy()
    block["label"] = (
        target_rca3 >= recovery_threshold
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

    frames.append(
        block[
            [
                "asof_year",
                "target_year",
                "country",
                "product",
                "label",
                "split",
            ]
        ]
    )

if not frames:
    raise ValueError("No recovery labels were generated.")

labels = pd.concat(frames, ignore_index=True)
labels["asof_year"] = labels["asof_year"].astype("int16")
labels["target_year"] = labels["target_year"].astype("int16")
labels["label"] = labels["label"].astype("int8")

if labels.duplicated(["asof_year", "country", "product"]).any():
    raise ValueError("Duplicate recovery label keys.")

# Eligibility audit: each eligible product had historical RCA3 >= 1
# before as-of and RCA3 < 1 at as-of.
for asof, subset in labels.groupby("asof_year"):
    idx = pd.MultiIndex.from_frame(subset[["country", "product"]])
    historical = wide.loc[idx, list(range(first_year, int(asof)))]
    prior_specialized = historical.ge(recovery_threshold).any(axis=1)
    current_values = wide.loc[idx, int(asof)]

    if not prior_specialized.all():
        raise ValueError(f"Historical specialization check failed for {asof}.")
    if not current_values.lt(recovery_threshold).all():
        raise ValueError(f"Current RCA3 check failed for {asof}.")

labels.to_parquet(OUTPUT_PATH, index=False)

summary = (
    labels.groupby(["asof_year", "target_year", "split"], observed=True)
    .agg(
        eligible=("product", "size"),
        recovered=("label", "sum"),
    )
    .reset_index()
)

india_summary = summary[summary["split"] == "test_india"].copy()
india_summary["recovery_rate_pct"] = (
    100 * india_summary["recovered"] / india_summary["eligible"]
)

lines = [
    "# Recovery Label Report",
    "",
    "## Definition",
    "",
    "- Previously specialized: RCA3 >= configured recovery threshold in at least one year before the as-of year.",
    "- Currently lost advantage: RCA3 < configured recovery threshold at the as-of year.",
    "- Recovered: RCA3 >= configured recovery threshold at the target year.",
    f"- Recovery threshold: {recovery_threshold}",
    f"- Prediction horizon: {horizon} years",
    "- This baseline does not require the loss to persist for two consecutive years.",
    "",
    "## Counts by window and split",
    "",
    "| As-of | Target | Split | Eligible | Recovered |",
    "|---:|---:|---|---:|---:|",
]

for _, row in summary.iterrows():
    lines.append(
        f"| {int(row['asof_year'])} | {int(row['target_year'])} | "
        f"{row['split']} | {int(row['eligible']):,} | "
        f"{int(row['recovered']):,} |"
    )

lines.extend([
    "",
    "## India recovery outcomes",
    "",
    "| As-of | Target | Eligible | Recovered | Rate (%) |",
    "|---:|---:|---:|---:|---:|",
])

for _, row in india_summary.iterrows():
    lines.append(
        f"| {int(row['asof_year'])} | {int(row['target_year'])} | "
        f"{int(row['eligible']):,} | {int(row['recovered']):,} | "
        f"{row['recovery_rate_pct']:.1f} |"
    )

lines.extend([
    "",
    "## Limitations",
    "",
    "- These labels describe observed RCA3 transitions; they do not establish causality.",
    "- Features must be restricted to information available by the as-of year.",
    "- Recovery labels must be modeled separately from new-emergence labels.",
    "- A sensitivity analysis should test a two-consecutive-year loss definition.",
    "",
])

REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

print("\nRECOVERY LABELS BUILT")
print(f"Output: {OUTPUT_PATH}")
print(f"Rows: {len(labels):,}")
print("\nSummary:")
print(summary.to_string(index=False))
print(f"\nReport: {REPORT_PATH}")
print(f"Elapsed seconds: {time.time() - start:.1f}")
