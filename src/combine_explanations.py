
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Load SHAP explanations
shap = pd.read_csv(
    ROOT / "outputs" / "india_shap_explanations.csv",
    dtype={"product": str}
)

# Load nearest related products
related = pd.read_csv(
    ROOT / "outputs" / "india_nearest_related_products_named.csv",
    dtype={
        "candidate_product": str,
        "related_product": str
    }
)

# Standardise product codes
shap["product"] = shap["product"].str.zfill(4)
related["candidate_product"] = related["candidate_product"].str.zfill(4)

# Keep only the top 10 candidates per year
shap = shap.sort_values(
    ["asof_year", "probability"],
    ascending=[True, False]
).groupby("asof_year").head(10)

# Rename columns for merging
shap = shap.rename(columns={
    "product": "candidate_product",
    "probability": "candidate_probability"
})

# Merge SHAP explanations with related products
final = shap.merge(
    related,
    on=["asof_year", "candidate_product", "candidate_probability"],
    how="left"
)

# Select final columns
final = final[[
    "asof_year",
    "candidate_product",
    "candidate_name",
    "candidate_probability",
    "label",
    "top_3_reasons",
    "related_product",
    "related_name",
    "proximity"
]]

# Save final dataset
output = ROOT / "outputs" / "india_final_explanations.csv"
final.to_csv(output, index=False)

print("Saved:", output)
print("\nFinal dataset shape:", final.shape)
print("\nMissing product names:", final["related_name"].isna().sum())
print("\nPreview:")
print(final.head(10).to_string(index=False))
