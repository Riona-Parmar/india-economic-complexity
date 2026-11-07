
import pandas as pd
from pathlib import Path

BASE = Path("data/processed")
P2 = Path("p2/data-p2")

files = {
    "Trade RCA": BASE / "trade_rca_v1.parquet",
    "Labels": BASE / "labels_v1.parquet",
    "Products": BASE / "products_v1.parquet",
    "Existing Features": BASE / "model_features_v1.parquet",
    "Density A2000": P2 / "density/density_A2000.parquet",
    "Density A2005": P2 / "density/density_A2005.parquet",
}

for name, path in files.items():
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    if not path.exists():
        print("FILE NOT FOUND:", path)
        continue

    df = pd.read_parquet(path)

    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())
