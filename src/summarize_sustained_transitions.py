
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "hs92_country_product_year_4.csv"
OUTPUT_FILE = ROOT / "outputs" / "reports" / "sustained_transition_summary.csv"

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

df = df.loc[
    df["product_hs92_code"].str.fullmatch(r"\d{4}")
].copy()

if df.duplicated(["product_hs92_code", "year"]).any():
    raise ValueError("Duplicate product-year records found.")

indexed = df.set_index(["product_hs92_code", "year"])

rows = []

for start_year in [2015, 2016, 2017]:
    target_year = start_year + 5

    for code, product in df.groupby("product_hs92_code"):
        by_year = product.set_index("year")

        needed_years = [start_year, target_year, target_year + 1, target_year + 2]
        if not all(year in by_year.index for year in needed_years):
            continue

        start_rca = by_year.loc[start_year, "export_rca"]
        target_rca = by_year.loc[target_year, "export_rca"]

        if not (start_rca < 0.25 and target_rca >= 1.0):
            continue

        future_rca = by_year.loc[
            [target_year, target_year + 1, target_year + 2],
            "export_rca",
        ]

        if (future_rca >= 1.0).all():
            rows.append({
                "product_hs92_code": code,
                "start_year": start_year,
                "target_year": target_year,
                "start_rca": start_rca,
                "target_rca": target_rca,
                "target_exports": by_year.loc[target_year, "export_value"],
                "exports_2024": (
                    by_year.loc[2024, "export_value"]
                    if 2024 in by_year.index else pd.NA
                ),
                "target_pci": by_year.loc[target_year, "pci"],
            })

result = pd.DataFrame(rows)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT_FILE, index=False)

print("SUSTAINED TRANSITION SUMMARY")
print("=" * 65)
print(f"Sustained product-window records: {len(result)}")

if not result.empty:
    print(f"Unique sustained products: {result['product_hs92_code'].nunique()}")
    print("\nResults:")
    print(result.sort_values(
        ["start_year", "target_exports"],
        ascending=[True, False],
    ).to_string(index=False))

print(f"\nSaved to: {OUTPUT_FILE}")
