
import os
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

INPUT = "data/DDS_bulk_download.csv"
OUTPUT_DIR = "software_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Detailed service categories.
# DDS is excluded because it is an aggregate category.
SERVICE_CODES = [
    "SF", "SG", "SH", "SI1",
    "SI2", "SI3", "SJ", "SK"
]

# SI2 = Computer services
INDIA_CODE = "IN"


# ============================================================
# 1. READ DATA
# ============================================================

print("Reading dataset...")

df = pd.read_csv(INPUT, low_memory=False)

# Normalize column names
df.columns = df.columns.str.strip().str.lower()

# Check required columns
required_columns = [
    "flow", "reporter_code", "reporter_name",
    "partner", "year", "indicator",
    "indicator_name", "mode", "unit", "value"
]

missing_columns = set(required_columns) - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing columns in CSV: {missing_columns}")

# Clean columns
for col in [
    "flow", "reporter_code", "reporter_name",
    "partner", "indicator", "indicator_name",
    "mode", "unit"
]:
    df[col] = df[col].astype(str).str.strip()

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["value"] = pd.to_numeric(df["value"], errors="coerce")

df = df.dropna(subset=["year", "value"])
df["year"] = df["year"].astype(int)

print("Total rows:", len(df))


# ============================================================
# 2. FILTER EXPORT DATA
# ============================================================

# FLOW = X represents exports.
# PARTNER = World.
# MODE = Mode 1.
# Keep only detailed service categories.

df = df[
    (df["flow"] == "X") &
    (df["partner"].str.lower() == "world") &
    (df["mode"].str.lower() == "mode 1") &
    (df["indicator"].isin(SERVICE_CODES))
].copy()

if df.empty:
    raise ValueError(
        "No rows found after filtering. Check FLOW, PARTNER, MODE and INDICATOR values."
    )

print("Rows after filtering:", len(df))
print("Years:", sorted(df["year"].unique()))
print("Countries:", df["reporter_code"].nunique())
print("Service categories:", sorted(df["indicator"].unique()))


# ============================================================
# 3. AGGREGATE EXPORT VALUES
# ============================================================

# Aggregate duplicate observations, if any.

df = (
    df.groupby(
        [
            "reporter_code",
            "reporter_name",
            "year",
            "indicator",
            "indicator_name"
        ],
        as_index=False
    )["value"].sum()
)

# Export values are in Million US dollars, as indicated by the dataset.
df.to_csv(
    os.path.join(OUTPUT_DIR, "services_exports.csv"),
    index=False
)


# ============================================================
# 4. CALCULATE SERVICES RCA
# ============================================================

print("\nCalculating services RCA...")

# Country's total exports across selected service categories
country_totals = (
    df.groupby(["reporter_code", "year"])["value"]
    .sum()
    .rename("country_total")
    .reset_index()
)

# Total exports of each service category across all reporters
world_service_totals = (
    df.groupby(["indicator", "year"])["value"]
    .sum()
    .rename("world_service_total")
    .reset_index()
)

# Total exports across all selected service categories
world_totals = (
    df.groupby("year")["value"]
    .sum()
    .rename("world_total")
    .reset_index()
)

rca = df.merge(
    country_totals,
    on=["reporter_code", "year"]
)

rca = rca.merge(
    world_service_totals,
    on=["indicator", "year"]
)

rca = rca.merge(
    world_totals,
    on="year"
)

# RCA = (country service share) / (world service share)

rca["rca"] = (
    (rca["value"] / rca["country_total"]) /
    (rca["world_service_total"] / rca["world_total"])
)

rca["has_rca"] = (rca["rca"] >= 1).astype(int)

rca.to_csv(
    os.path.join(OUTPUT_DIR, "services_rca.csv"),
    index=False
)


# ============================================================
# 5. BUILD RCA BINARY MATRIX
# ============================================================

print("\nBuilding RCA matrix...")

# Each row = country-year
# Each column = service category
# 1 = RCA >= 1, 0 = RCA < 1

binary = rca.pivot_table(
    index=["reporter_code", "reporter_name", "year"],
    columns="indicator",
    values="has_rca",
    aggfunc="max",
    fill_value=0
)

# Ensure all eight service categories are present
binary = binary.reindex(
    columns=SERVICE_CODES,
    fill_value=0
)


# ============================================================
# 6. CALCULATE GLOBAL SERVICE PROXIMITY
# ============================================================

print("\nCalculating proximity...")

proximity_results = []

for year in sorted(binary.index.get_level_values("year").unique()):

    year_data = binary[
        binary.index.get_level_values("year") == year
    ]

    M = year_data.to_numpy(dtype=float)

    # Number of countries with RCA >= 1 in each service
    counts = M.sum(axis=0)

    # Co-occurrence counts between service pairs
    cooccurrence = M.T @ M

    # Conditional probabilities P(service i | service j)
    conditional = np.divide(
        cooccurrence,
        counts[np.newaxis, :],
        out=np.zeros_like(cooccurrence),
        where=counts[np.newaxis, :] != 0
    )

    # Symmetric proximity
    phi = np.minimum(conditional, conditional.T)

    for i, service_i in enumerate(SERVICE_CODES):
        for j, service_j in enumerate(SERVICE_CODES):

            proximity_results.append({
                "year": year,
                "service_1": service_i,
                "service_2": service_j,
                "proximity": phi[i, j]
            })

proximity_out = pd.DataFrame(proximity_results)

proximity_out.to_csv(
    os.path.join(OUTPUT_DIR, "services_proximity.csv"),
    index=False
)


# ============================================================
# 7. CALCULATE COUNTRY-SPECIFIC DENSITY
# ============================================================

print("\nCalculating density...")

density_results = []

for year in sorted(binary.index.get_level_values("year").unique()):

    year_data = binary[
        binary.index.get_level_values("year") == year
    ]

    M = year_data.to_numpy(dtype=float)

    # Reconstruct proximity matrix for this year
    counts = M.sum(axis=0)
    cooccurrence = M.T @ M

    conditional = np.divide(
        cooccurrence,
        counts[np.newaxis, :],
        out=np.zeros_like(cooccurrence),
        where=counts[np.newaxis, :] != 0
    )

    phi = np.minimum(conditional, conditional.T)

    # Density = (M @ Phi.T) / sum(Phi)
    denominator = phi.sum(axis=1)

    density = np.divide(
        M @ phi.T,
        denominator[np.newaxis, :],
        out=np.zeros((M.shape[0], len(SERVICE_CODES))),
        where=denominator[np.newaxis, :] != 0
    )

    for row_idx, (country_code, country_name, row_year) in enumerate(
        year_data.index
    ):

        for service_idx, service in enumerate(SERVICE_CODES):

            density_results.append({
                "reporter_code": country_code,
                "reporter_name": country_name,
                "year": row_year,
                "indicator": service,
                "rca": M[row_idx, service_idx],
                "density": density[row_idx, service_idx]
            })

density_out = pd.DataFrame(density_results)

density_out.to_csv(
    os.path.join(OUTPUT_DIR, "services_density.csv"),
    index=False
)


# ============================================================
# 8. INDIA COMPUTER SERVICES ANALYSIS (SI2)
# ============================================================

print("\nExtracting India's computer services results...")

india_computer = rca[
    (rca["reporter_code"] == INDIA_CODE) &
    (rca["indicator"] == "SI2")
].copy()

india_computer = india_computer.merge(
    density_out[
        (density_out["reporter_code"] == INDIA_CODE) &
        (density_out["indicator"] == "SI2")
    ][["reporter_code", "year", "density"]],
    on=["reporter_code", "year"],
    how="left"
)

india_computer = india_computer[
    [
        "reporter_code",
        "reporter_name",
        "year",
        "indicator",
        "indicator_name",
        "value",
        "rca",
        "has_rca",
        "density"
    ]
].sort_values("year")

india_computer.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "india_computer_services_rca.csv"
    ),
    index=False
)


# ============================================================
# 9. INDIA SOFTWARE SERVICES SUMMARY
# ============================================================

# Compare India's SI2 exports with global SI2 exports.
india_summary = india_computer.copy()

india_summary = india_summary.merge(
    world_service_totals[
        world_service_totals["indicator"] == "SI2"
    ],
    on=["indicator", "year"],
    how="left"
)

india_summary["india_global_export_share_pct"] = (
    india_summary["value"] /
    india_summary["world_service_total"]
) * 100

india_summary = india_summary.rename(
    columns={
        "value": "india_computer_services_exports_million_usd"
    }
)

india_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "india_software_summary.csv"
    ),
    index=False
)


# ============================================================
# 10. PRINT RESULTS
# ============================================================

print("\n======================================")
print("SOFTWARE SERVICES ANALYSIS COMPLETED")
print("======================================")

print("\nIndia Computer Services (SI2):")

if india_summary.empty:
    print("No India SI2 records found.")
else:
    print(
        india_summary[
            [
                "year",
                "india_computer_services_exports_million_usd",
                "rca",
                "density",
                "india_global_export_share_pct"
            ]
        ].to_string(index=False)
    )

print("\nFiles generated inside:", OUTPUT_DIR)

print("""
1. services_exports.csv
2. services_rca.csv
3. services_proximity.csv
4. services_density.csv
5. india_computer_services_rca.csv
6. india_software_summary.csv
""")
