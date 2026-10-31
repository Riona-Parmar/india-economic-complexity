import pandas as pd

df = pd.read_parquet(
    "../india-economic-complexity/data/processed/trade_rca_v1.parquet"
)

# 1. Display first 5 rows
print(df.head())

# 2. Display columns
print("\nColumns:")
print(df.columns.tolist())

# 3. Dataset dimensions
print("\nShape:", df.shape)

# 4. Check data types and missing values
print("\nDataset information:")
print(df.info())

# 5. Check countries, products and years
print("\nCountries:", df["country"].nunique())
print("Products:", df["product"].nunique())
print("First year:", df["year"].min())
print("Last year:", df["year"].max())

# 6. Check m values
print("\nm value counts:")
print(df["m"].value_counts())

# 7. Check missing values
print("\nMissing values:")
print(df.isnull().sum())