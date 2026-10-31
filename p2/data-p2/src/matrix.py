
import pandas as pd
import numpy as np

# Load P1 dataset
df = pd.read_parquetdf = pd.read_parquet(
    "../india-economic-complexity/data/processed/trade_rca_v1.parquet"
)

# Years for which we need matrices
asof_years = [2000, 2005, 2010, 2015]

for asof_year in asof_years:

    print(f"\nProcessing year: {asof_year}")

    # Use only data available up to this year
    temp = df[df["year"] <= asof_year]

    # Get the latest observation for each country-product pair
    latest = (
        temp.sort_values("year")
        .groupby(["country", "product"], as_index=False)
        .tail(1)
    )

    # Convert into country-product matrix
    m = latest.pivot(
        index="country",
        columns="product",
        values="m"
    )

    # Treat missing country-product combinations as absence
    m = m.fillna(0)

    # Save matrix
    filename = f"matrix_A{asof_year}.parquet"
    m.to_parquet(filename)

    print("Matrix shape:", m.shape)
    print("Missing values:", m.isna().sum().sum())
    print("Saved:", filename)

print("\nAll matrices generated successfully!")
