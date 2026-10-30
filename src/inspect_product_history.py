
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

# Inspect one known product: motor vehicle parts.
product_code = "8708"

history = df.loc[
    (df["country_iso3_code"] == "IND")
    & (df["product_hs92_code"] == product_code)
].sort_values("year")

print(f"INDIA — PRODUCT {product_code}")
print("=" * 65)
print(
    history[
        ["year", "export_value", "export_rca", "pci"]
    ].to_string(index=False)
)

print("\nCheck proposed five-year transition:")
for start_year in [2015, 2016, 2017, 2018, 2019]:
    start = history.loc[history["year"] == start_year]
    target = history.loc[history["year"] == start_year + 5]

    if start.empty or target.empty:
        print(f"{start_year} -> {start_year + 5}: missing record")
        continue

    start_rca = start["export_rca"].iloc[0]
    target_rca = target["export_rca"].iloc[0]

    emerged = start_rca < 0.25 and target_rca >= 1.0

    print(
        f"{start_year} -> {start_year + 5}: "
        f"RCA {start_rca:.3f} -> {target_rca:.3f}; "
        f"emerged={emerged}"
    )
