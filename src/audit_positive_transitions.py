
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = ROOT / "data" / "raw" / "hs92_country_product_year_4.csv"
DESC_FILE = ROOT / "data" / "interim" / "hs92_hs4_descriptions.csv"
SECTOR_FILE = ROOT / "data" / "interim" / "hs92_hs4_sector_mapping_v2.csv"
OUTPUT_FILE = ROOT / "outputs" / "reports" / "positive_transition_audit.csv"

df = pd.read_csv(
    RAW_FILE,
    usecols=[
        "country_iso3_code",
        "product_hs92_code",
        "year",
        "export_value",
        "export_rca",
        "pci",
    ],
    dtype={
        "country_iso3_code": "string",
        "product_hs92_code": "string",
    },
    low_memory=False,
)

df = df.loc[df["country_iso3_code"] == "IND"].copy()

df["product_hs92_code"] = (
    df["product_hs92_code"]
    .str.strip()
    .str.replace(r"\.0$", "", regex=True)
    .str.zfill(4)
)

# Exclude special codes such as XXXX.
df = df.loc[
    df["product_hs92_code"].str.fullmatch(r"\d{4}")
].copy()

if df.duplicated(["product_hs92_code", "year"]).any():
    raise ValueError("Duplicate product-year records found.")

descriptions = pd.read_csv(
    DESC_FILE,
    dtype={"product_hs92_code": "string"},
)
descriptions["product_hs92_code"] = (
    descriptions["product_hs92_code"].str.strip().str.zfill(4)
)
descriptions = descriptions[
    ["product_hs92_code", "product_description"]
].drop_duplicates("product_hs92_code")

sectors = pd.read_csv(
    SECTOR_FILE,
    dtype={"product_hs92_code": "string"},
)
sectors["product_hs92_code"] = (
    sectors["product_hs92_code"].str.strip().str.zfill(4)
)
sectors = sectors[
    ["product_hs92_code", "sector_name"]
].drop_duplicates("product_hs92_code")

records = df.set_index(["product_hs92_code", "year"])

audit_rows = []

for start_year in [2015, 2016, 2017, 2018, 2019]:
    target_year = start_year + 5

    start = df.loc[
        df["year"] == start_year,
        ["product_hs92_code", "export_value", "export_rca"],
    ].rename(columns={
        "export_value": "start_exports",
        "export_rca": "start_rca",
    }).set_index("product_hs92_code")

    target = df.loc[
        df["year"] == target_year,
        ["product_hs92_code", "export_value", "export_rca", "pci"],
    ].rename(columns={
        "export_value": "target_exports",
        "export_rca": "target_rca",
        "pci": "target_pci",
    }).set_index("product_hs92_code")

    joined = start.join(target, how="inner")

    positives = joined.loc[
        (joined["start_rca"] < 0.25)
        & (joined["target_rca"] >= 1.0)
    ].copy()

    positives["start_year"] = start_year
    positives["target_year"] = target_year

    positives["export_growth_pct"] = (
        (positives["target_exports"] / positives["start_exports"] - 1)
        * 100
    ).where(positives["start_exports"] > 0)

    audit_rows.append(positives.reset_index())

audit = pd.concat(audit_rows, ignore_index=True)
audit = audit.merge(descriptions, on="product_hs92_code", how="left")
audit = audit.merge(sectors, on="product_hs92_code", how="left")

audit = audit[
    [
        "product_hs92_code",
        "product_description",
        "sector_name",
        "start_year",
        "target_year",
        "start_exports",
        "target_exports",
        "export_growth_pct",
        "start_rca",
        "target_rca",
        "target_pci",
    ]
].sort_values(["start_year", "product_hs92_code"])

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
audit.to_csv(OUTPUT_FILE, index=False)

print("POSITIVE TRANSITION AUDIT")
print("=" * 80)
print(f"Transition records: {len(audit)}")
print(f"Unique products across windows: {audit['product_hs92_code'].nunique()}")
print("\nAudit results:")
print(audit.round(3).to_string(index=False))
print(f"\nSaved report to: {OUTPUT_FILE}")
