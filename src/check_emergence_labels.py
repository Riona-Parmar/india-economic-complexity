
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
        "export_rca",
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

# Special codes are not valid HS4 products for this analysis.
df = df.loc[df["product_hs92_code"].str.fullmatch(r"\d{4}")].copy()

# Check for duplicate product-year records before creating labels.
duplicates = df.duplicated(
    subset=["product_hs92_code", "year"]
).sum()

if duplicates:
    raise ValueError(f"Found {duplicates} duplicate product-year records.")

rca = df.pivot(
    index="product_hs92_code",
    columns="year",
    values="export_rca",
)

print("FIVE-YEAR EXPORT-SPECIALIZATION TRANSITIONS")
print("=" * 65)
print("Definition: starting RCA < 0.25 and target RCA >= 1.0")
print("Only products with records in both years are eligible.\n")


print("POSITIVE TRANSITIONS BY PRODUCT")
print("=" * 75)

for start_year in [2015, 2016, 2017, 2018, 2019]:
    target_year = start_year + 5

    start = rca[start_year]
    target = rca[target_year]

    eligible = start.notna() & target.notna()
    positive = eligible & (start < 0.25) & (target >= 1.0)

    results = pd.DataFrame({
        "product_hs92_code": rca.index,
        "start_rca": start.values,
        "target_rca": target.values,
    })

    results = results.loc[positive.values].copy()
    results["start_year"] = start_year
    results["target_year"] = target_year

    print(f"\n{start_year} -> {target_year}: {len(results)} products")

    if results.empty:
        print("No positive transitions.")
    else:
        print(
            results[
                [
                    "product_hs92_code",
                    "start_year",
                    "target_year",
                    "start_rca",
                    "target_rca",
                ]
            ].to_string(index=False)
        )
