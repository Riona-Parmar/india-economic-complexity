
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

path = (
    ROOT / "p2" / "data-p2" /
    "proximity" / "proximity_A2015.parquet"
)

df = pd.read_parquet(path)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)
