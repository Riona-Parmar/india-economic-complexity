
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Load existing related-products results
related = pd.read_csv(
    ROOT / "outputs" / "india_nearest_related_products.csv",
    dtype={
        "candidate_product": str,
        "related_product": str
    }
)

# Load product metadata
products = pd.read_parquet(
    ROOT / "data" / "processed" / "products_v1.parquet"
)

# Keep product codes as four-digit strings
products["product"] = products["product"].astype(str).str.zfill(4)

# Create product-code-to-name mapping
product_names = products[["product", "name"]].drop_duplicates(
    subset="product"
)

# Add candidate and related product names
related = related.merge(
    product_names.rename(columns={
        "product": "candidate_product",
        "name": "candidate_name"
    }),
    on="candidate_product",
    how="left"
)

related = related.merge(
    product_names.rename(columns={
        "product": "related_product",
        "name": "related_name"
    }),
    on="related_product",
    how="left"
)

# Arrange columns
related = related[[
    "asof_year",
    "candidate_product",
    "candidate_name",
    "candidate_probability",
    "related_product",
    "related_name",
    "proximity"
]]

# Save
output = ROOT / "outputs" / "india_nearest_related_products_named.csv"
related.to_csv(output, index=False)

print("Saved:", output)
print("\nRESULTS:")
print(related.to_string(index=False))
print("\nMissing candidate names:", related["candidate_name"].isna().sum())
print("Missing related names:", related["related_name"].isna().sum())
