
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = ROOT / "data/raw/hs92_country_product_year_4.csv"
DESCRIPTION_FILE = ROOT / "data/interim/hs92_hs4_descriptions.csv"
MAPPING_FILE = ROOT / "data/interim/hs92_hs4_sector_mapping_v2.csv"
OUTPUT_FILE = ROOT / "data/processed/india_product_baseline_2024.csv"


# --------------------------------------------------
# 2. Helper function
# --------------------------------------------------

def normalize_hs4(series):
    """Normalize numeric HS4 codes to four-character strings."""
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(4)
    )


# --------------------------------------------------
# 3. Load India's 2024 product records
# --------------------------------------------------

print("Loading India's 2024 product records...")

df = pd.read_csv(
    RAW_FILE,
    dtype={"product_hs92_code": str},
    low_memory=False
)

df = df[
    (df["country_iso3_code"] == "IND") &
    (df["year"] == 2024)
].copy()

if df.empty:
    raise ValueError("No India product records found for 2024.")

df["product_hs92_code"] = normalize_hs4(
    df["product_hs92_code"]
)

print("India product records loaded:", len(df))
print("Unique product codes:", df["product_hs92_code"].nunique())


# --------------------------------------------------
# 4. Load and validate HS92 product descriptions
# --------------------------------------------------

print("\nLoading HS92 product descriptions...")

descriptions = pd.read_csv(
    DESCRIPTION_FILE,
    dtype={"product_hs92_code": str}
)

required_description_columns = {
    "product_hs92_code",
    "product_description"
}

missing_columns = required_description_columns - set(descriptions.columns)

if missing_columns:
    raise ValueError(
        f"Description file is missing columns: {sorted(missing_columns)}"
    )

descriptions["product_hs92_code"] = normalize_hs4(
    descriptions["product_hs92_code"]
)

descriptions = descriptions.rename(
    columns={"product_description": "description"}
)

# Only one description row should exist per HS4 code.
duplicate_descriptions = descriptions[
    descriptions["product_hs92_code"].duplicated(keep=False)
]

if not duplicate_descriptions.empty:
    raise ValueError(
        "Duplicate HS4 codes found in the description file. "
        "Resolve them before merging."
    )

descriptions = descriptions[
    ["product_hs92_code", "description"]
]


# --------------------------------------------------
# 5. Load and validate sector mapping
# --------------------------------------------------

print("Loading HS92 sector mapping...")

mapping = pd.read_csv(
    MAPPING_FILE,
    dtype={"product_hs92_code": str}
)

required_mapping_columns = {
    "product_hs92_code",
    "sector_id",
    "sector_name"
}

missing_columns = required_mapping_columns - set(mapping.columns)

if missing_columns:
    raise ValueError(
        f"Sector mapping file is missing columns: {sorted(missing_columns)}"
    )

mapping["product_hs92_code"] = normalize_hs4(
    mapping["product_hs92_code"]
)

duplicate_mapping = mapping[
    mapping["product_hs92_code"].duplicated(keep=False)
]

if not duplicate_mapping.empty:
    raise ValueError(
        "Duplicate HS4 codes found in the sector mapping. "
        "Resolve them before merging."
    )

mapping = mapping[
    ["product_hs92_code", "sector_id", "sector_name"]
]


# --------------------------------------------------
# 6. Merge descriptions and sector assignments
# --------------------------------------------------

print("\nJoining product descriptions...")

df = df.merge(
    descriptions,
    on="product_hs92_code",
    how="left",
    validate="many_to_one",
    indicator="_description_match"
)

missing_description_count = (
    df["_description_match"] == "left_only"
).sum()

df = df.drop(columns=["_description_match"])

print("Products without descriptions:", missing_description_count)


print("Joining sector assignments...")

df = df.merge(
    mapping,
    on="product_hs92_code",
    how="left",
    validate="many_to_one",
    indicator="_sector_match"
)

missing_sector_count = (
    df["_sector_match"] == "left_only"
).sum()

df = df.drop(columns=["_sector_match"])

print("Products without sector assignments:", missing_sector_count)



# --------------------------------------------------
# 7. Create baseline indicators
# --------------------------------------------------

df["has_exports"] = df["export_value"] > 0

df["revealed_specialization"] = df["export_rca"] > 1

# Explicitly identify special and unmapped product codes.
df["mapping_status"] = "mapped_hs4"

df.loc[
    df["product_hs92_code"] == "XXXX",
    "mapping_status"
] = "unmapped_special_code"

df.loc[
    df["sector_id"].isna() & (df["product_hs92_code"] != "XXXX"),
    "mapping_status"
] = "unexpected_unmapped_code"

# Preserve missing sector IDs as nullable integers.
df["sector_id"] = df["sector_id"].astype("Int64")
print("\nMapping status check:")
print(
    df.loc[
        df["sector_id"].isna(),
        ["product_hs92_code", "export_value", "export_rca", "pci"]
    ].to_string(index=False)
)

# --------------------------------------------------
# 8. Sort and save
# --------------------------------------------------

df = df.sort_values(
    "export_value",
    ascending=False
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# 9. Report validation results
# --------------------------------------------------

print("\n" + "=" * 55)
print("INDIA PRODUCT BASELINE — 2024")
print("=" * 55)

print("Output file:", OUTPUT_FILE)
print("Product records:", len(df))
print("Unique HS4 codes:", df["product_hs92_code"].nunique())
print("Missing descriptions:", df["description"].isna().sum())
print("Missing sector assignments:", df["sector_id"].isna().sum())
print("Products with positive exports:", int(df["has_exports"].sum()))
print("Products with RCA > 1:", int(df["revealed_specialization"].sum()))
print("Total exports:", f"{df['export_value'].sum():,.2f}")

print("\nTop 15 products by export value:")

print(
    df[
        [
            "product_hs92_code",
            "description",
            "sector_name",
            "export_value",
            "export_rca",
            "pci"
        ]
    ]
    .head(15)
    .to_string(index=False)
)

print("\nBaseline generation completed.")
