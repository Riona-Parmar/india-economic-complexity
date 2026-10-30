
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

df = df.loc[
    df["product_hs92_code"].str.fullmatch(r"\d{4}")
].copy()

if df.duplicated(["product_hs92_code", "year"]).any():
    raise ValueError("Duplicate product-year records found.")

rca = df.pivot(
    index="product_hs92_code",
    columns="year",
    values="export_rca",
)

print("SUSTAINED SPECIALIZATION AUDIT")
print("=" * 70)
print("Strict emergence: start RCA < 0.25, target RCA >= 1.0")
print("Persistence: RCA >= 1.0 in target year and the next two years.")
print("Windows without three post-target years are marked unavailable.\n")

for start_year in [2015, 2016, 2017, 2018, 2019]:
    target_year = start_year + 5

    start = rca[start_year]
    target = rca[target_year]

    eligible = start.notna() & target.notna()
    emerged = eligible & (start < 0.25) & (target >= 1.0)

    # We need target year plus two subsequent years.
    future_years = [target_year, target_year + 1, target_year + 2]

    if any(year not in rca.columns for year in future_years):
        print(
            f"{start_year} -> {target_year}: "
            f"strict emergence={int(emerged.sum())}; "
            "persistence unavailable"
        )
        continue

    future = rca[future_years]
    sustained = future.ge(1.0).all(axis=1) & emerged

    print(
        f"{start_year} -> {target_year}: "
        f"strict emergence={int(emerged.sum())}; "
        f"sustained for 3 years={int(sustained.sum())}"
    )

    if emerged.any():
        print("  Emerged products and persistence:")
        for code in emerged[emerged].index:
            status = "sustained" if sustained.get(code, False) else "not sustained"
            print(f"    {code}: {status}")
