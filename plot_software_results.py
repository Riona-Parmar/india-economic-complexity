
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = "software_results"
OUTPUT_DIR = os.path.join(INPUT_DIR, "visuals")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read generated CSV files
summary = pd.read_csv(
    os.path.join(INPUT_DIR, "india_software_summary.csv")
)

rca = pd.read_csv(
    os.path.join(INPUT_DIR, "services_rca.csv")
)

proximity = pd.read_csv(
    os.path.join(INPUT_DIR, "services_proximity.csv")
)

density = pd.read_csv(
    os.path.join(INPUT_DIR, "services_density.csv")
)

exports = pd.read_csv(
    os.path.join(INPUT_DIR, "services_exports.csv")
)

# Clean and sort
summary = summary.sort_values("year")

for df in [summary, rca, proximity, density, exports]:
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")


# ============================================================
# VISUAL 1: INDIA COMPUTER SERVICES EXPORTS
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    summary["year"],
    summary["india_computer_services_exports_million_usd"],
    marker="o",
    label="Computer Services Exports"
)

plt.title("India's Computer Services Exports (2005–2025)")
plt.xlabel("Year")
plt.ylabel("Exports (Million USD)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "01_india_exports.png"),
    dpi=300
)
plt.close()


# ============================================================
# VISUAL 2: INDIA RCA AND GLOBAL EXPORT SHARE
# ============================================================

fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(
    summary["year"],
    summary["rca"],
    marker="o",
    label="RCA"
)

ax1.axhline(
    y=1,
    linestyle="--",
    label="RCA = 1 threshold"
)

ax1.set_xlabel("Year")
ax1.set_ylabel("RCA")
ax1.set_title(
    "India's Computer Services Export Competitiveness"
)
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()

ax2.plot(
    summary["year"],
    summary["india_global_export_share_pct"],
    marker="s",
    linestyle=":",
    label="Global Export Share"
)

ax2.set_ylabel("Global Export Share (%)")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="best"
)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "02_india_rca_share.png"),
    dpi=300
)
plt.close()


# ============================================================
# VISUAL 3: INTERNATIONAL RCA COMPARISON
# ============================================================

# Select latest year with SI2 data
si2 = rca[rca["indicator"] == "SI2"].copy()

latest_year = int(si2["year"].max())

latest = si2[si2["year"] == latest_year].copy()

# Top 10 countries by computer-services exports
top_exporters = (
    exports[
        (exports["indicator"] == "SI2") &
        (exports["year"] == latest_year)
    ]
    .nlargest(10, "value")["reporter_code"]
    .tolist()
)

# Always include India
if "IN" not in top_exporters:
    top_exporters.append("IN")

comparison = latest[
    latest["reporter_code"].isin(top_exporters)
].copy()

comparison = comparison.sort_values("rca", ascending=False)

plt.figure(figsize=(11, 6))

plt.bar(
    comparison["reporter_name"],
    comparison["rca"]
)

plt.axhline(
    y=1,
    linestyle="--",
    label="RCA = 1 threshold"
)

plt.title(
    f"Computer Services RCA: International Comparison ({latest_year})"
)

plt.xlabel("Country")
plt.ylabel("RCA")
plt.xticks(rotation=45, ha="right")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "03_global_rca_comparison.png"),
    dpi=300
)
plt.close()


# ============================================================
# VISUAL 4: GLOBAL SERVICES PROXIMITY HEATMAP
# ============================================================

# Prefer 2020 for the heatmap; otherwise use latest available year
available_years = sorted(proximity["year"].dropna().unique())

selected_year = (
    2020 if 2020 in available_years
    else int(max(available_years))
)

prox_year = proximity[
    proximity["year"] == selected_year
]

heatmap = prox_year.pivot(
    index="service_1",
    columns="service_2",
    values="proximity"
)

# Ensure consistent category ordering
service_order = [
    "SF", "SG", "SH", "SI1",
    "SI2", "SI3", "SJ", "SK"
]

heatmap = heatmap.reindex(
    index=service_order,
    columns=service_order
)

fig, ax = plt.subplots(figsize=(9, 7))

im = ax.imshow(
    heatmap.values,
    aspect="auto",
    vmin=0,
    vmax=1
)

ax.set_xticks(range(len(service_order)))
ax.set_yticks(range(len(service_order)))

ax.set_xticklabels(service_order)
ax.set_yticklabels(service_order)

ax.set_xlabel("Service Category")
ax.set_ylabel("Service Category")

ax.set_title(
    f"Global Service Proximity Matrix ({selected_year})"
)

plt.colorbar(
    im,
    ax=ax,
    label="Proximity (0–1)"
)

# Add numerical values inside cells
for i in range(len(service_order)):
    for j in range(len(service_order)):
        value = heatmap.iloc[i, j]

        if pd.notna(value):
            ax.text(
                j, i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8
            )

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "04_services_proximity.png"),
    dpi=300
)
plt.close()


# ============================================================
# VISUAL 5: INDIA COMPUTER SERVICES DENSITY
# ============================================================

india_density = density[
    (density["reporter_code"] == "IN") &
    (density["indicator"] == "SI2")
].sort_values("year")

plt.figure(figsize=(10, 5))

plt.plot(
    india_density["year"],
    india_density["density"],
    marker="o",
    label="Computer Services Density"
)

plt.title(
    "India's Computer Services Density Over Time"
)

plt.xlabel("Year")
plt.ylabel("Density")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "05_india_density.png"),
    dpi=300
)
plt.close()


# ============================================================
# COMPLETION
# ============================================================

print("\n======================================")
print("ALL VISUALIZATIONS GENERATED")
print("======================================")

for filename in sorted(os.listdir(OUTPUT_DIR)):
    print(filename)

print("\nSaved in:", OUTPUT_DIR)
