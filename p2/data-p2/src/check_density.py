import pandas as pd

labels = pd.read_parquet("labels_v1.parquet")
predictions = pd.read_parquet("predictions_density.parquet")

data = labels[
    labels["split"].isin(["test_india", "test_peer"])
].merge(
    predictions,
    on=["asof_year", "country", "product"],
    how="inner"
)

print("Density-label correlation by year:\n")

for year, group in data.groupby("asof_year"):
    correlation = group["score"].corr(group["label"].astype(float))
    print(year, ":", correlation)

print("\nOverall correlation:")
print(data["score"].corr(data["label"].astype(float)))