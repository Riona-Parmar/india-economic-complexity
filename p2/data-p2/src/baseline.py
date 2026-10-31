
import pandas as pd

# Load original trade data
df = pd.read_parquetdf = pd.read_parquet(
    "../india-economic-complexity/data/processed/trade_rca_v1.parquet"
)

# Load P1 labels
labels = pd.read_parquet("labels_v1.parquet")

# Only rows that have actual labels
test_labels = labels[
    labels["split"].isin(["test_india", "test_peer"])
].copy()

predictions = []

# Generate predictions for each as-of year
for year in sorted(test_labels["asof_year"].unique()):

    print(f"Processing {year}")

    # Use ONLY historical data up to the as-of year
    historical = df[df["year"] <= year]

    # Get the latest RCA3 value for each country-product pair
    latest = (
        historical.sort_values("year")
        .groupby(["country", "product"], as_index=False)
        .tail(1)
    )

    # Keep only eligible test pairs
    eligible = test_labels[
        test_labels["asof_year"] == year
    ][["asof_year", "country", "product"]]

    # Match current RCA3 with eligible pairs
    year_predictions = eligible.merge(
        latest[["country", "product", "rca3"]],
        on=["country", "product"],
        how="left"
    )

    if year_predictions["rca3"].isna().any():
        raise ValueError(f"Missing RCA3 values for year {year}")

    year_predictions = year_predictions.rename(
        columns={"rca3": "score"}
    )

    predictions.append(year_predictions)

# Combine all years
predictions = pd.concat(predictions, ignore_index=True)

# Save predictions
predictions.to_parquet("predictions_rca.parquet", index=False)

print("\nPredictions shape:", predictions.shape)
print(predictions.head())
print("\nSaved predictions_rca.parquet")

# Evaluate
from evaluation import evaluate

metrics = evaluate(predictions, labels)

metrics.to_csv("metrics_rca.csv", index=False)

print("\nRCA Persistence Results:")
print(metrics.to_string(index=False))
