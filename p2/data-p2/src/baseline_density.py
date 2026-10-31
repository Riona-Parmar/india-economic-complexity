
import pandas as pd

# Load P1 labels
labels = pd.read_parquet("labels_v1.parquet")

# Only evaluate rows with known outcomes
test_labels = labels[
    labels["split"].isin(["test_india", "test_peer"])
].copy()

predictions = []

# Generate density predictions for each as-of year
for year in sorted(test_labels["asof_year"].unique()):

    print(f"Processing {year}")

    # Load density calculated using information up to this year
    density = pd.read_parquet(f"density_A{year}.parquet")

    # Keep only the country-product pairs eligible for evaluation
    eligible = test_labels[
        test_labels["asof_year"] == year
    ][["asof_year", "country", "product"]]

    # Match density scores with eligible pairs
    year_predictions = eligible.merge(
        density[["country", "product", "density"]],
        on=["country", "product"],
        how="left",
        validate="one_to_one"
    )

    # Check that every eligible pair has a density score
    if year_predictions["density"].isna().any():
        raise ValueError(f"Missing density scores for year {year}")

    # Rename density to score so the evaluation function can use it
    year_predictions = year_predictions.rename(
        columns={"density": "score"}
    )

    predictions.append(year_predictions)

# Combine all years
predictions = pd.concat(predictions, ignore_index=True)

# Save predictions
predictions.to_parquet(
    "predictions_density.parquet",
    index=False
)

print("\nPredictions shape:", predictions.shape)
print(predictions.head())

print("\nSaved predictions_density.parquet")

# Evaluate using the same harness
from evaluation import evaluate

metrics = evaluate(predictions, labels)

metrics.to_csv("metrics_density.csv", index=False)

print("\nDensity Baseline Results:")
print(metrics.to_string(index=False))
