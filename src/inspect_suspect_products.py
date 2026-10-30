
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "hs92_country_product_year_4.csv"

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

df["product_hs92_code"] = (
    df["product_hs92_code"]
    .str.strip()
    .str.replace(r"\.0$", "", regex=True)
    .str.zfill(4)
)

df = df.loc[
    (df["country_iso3_code"] == "IND")
    & (df["product_hs92_code"].isin(["7206", "8525", "8411"]))
    & df["year"].between(2015, 2024)
].copy()

df = df.sort_values(["product_hs92_code", "year"])

for code, group in df.groupby("product_hs92_code"):
    print(f"\nPRODUCT {code}")
    print("=" * 65)
    print(
        group[
            ["year", "export_value", "export_rca", "pci"]
        ].to_string(index=False)
    )

OUTPUT_FILE = ROOT / "outputs" / "reports" / "suspect_product_annual_series.csv"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved annual series to: {OUTPUT_FILE}")
