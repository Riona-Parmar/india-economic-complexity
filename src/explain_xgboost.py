
import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
import shap

ROOT = Path(__file__).resolve().parents[1]

# Load dataset
df = pd.read_parquet(
    ROOT / "data" / "processed" / "p3_dataset_v2.parquet"
)

FEATURES = [
    "density",
    "pci_current",
    "rca3_current",
    "rca3_change_3y",
    "eci_current",
    "f_ubiquity",
    "f_world_growth"
]


# Generate SHAP explanations for India

all_explanations = []

for year in [2010, 2015]:

    print(f"\nGenerating SHAP values for {year}")

    india_data = df[
        (df["country"] == "IND") &
        (df["asof_year"] == year) &
        (df["split"] == "test_india")
    ].copy()

    # Load trained v1 model
    model = XGBClassifier()
    model.load_model(
        ROOT / "outputs" / f"xgboost_model_{year}.json"
    )

    X_india = india_data[FEATURES]

    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_india)

    print("SHAP shape:", shap_values.shape)

    # Get model predictions
    probabilities = model.predict_proba(X_india)[:, 1]

    # Find top 3 influential features per product
    for i in range(len(india_data)):

        feature_contributions = pd.DataFrame({
            "feature": FEATURES,
            "shap_value": shap_values[i]
        })

        feature_contributions["absolute_shap"] = (
            feature_contributions["shap_value"].abs()
        )

        top3 = feature_contributions.sort_values(
            "absolute_shap",
            ascending=False
        ).head(3)

        explanations = []

        for _, row in top3.iterrows():

            direction = (
                "increases"
                if row["shap_value"] > 0
                else "decreases"
            )

            explanations.append(
                f'{row["feature"]} {direction}'
            )

        all_explanations.append({
            "asof_year": year,
            "country": "IND",
            "product": india_data.iloc[i]["product"],
            "probability": probabilities[i],
            "label": india_data.iloc[i]["label"],
            "top_3_reasons": "; ".join(explanations)
        })

# Save explanations
explanation_df = pd.DataFrame(all_explanations)

output_path = (
    ROOT / "outputs" / "india_shap_explanations.csv"
)

explanation_df.to_csv(output_path, index=False)

print("\nSHAP explanations saved:", output_path)

# Display India's top 10 predicted products
for year in [2010, 2015]:

    print(f"\nTOP 10 INDIA PRODUCTS - {year}")

    top_products = explanation_df[
        explanation_df["asof_year"] == year
    ].sort_values(
        "probability", ascending=False
    ).head(10)

    print(
        top_products[
            ["product", "probability", "label", "top_3_reasons"]
        ].to_string(index=False)
    )
