
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score


def evaluate(predictions, labels):

    # Columns required
    required_pred = ["asof_year", "country", "product", "score"]
    required_label = [
        "asof_year", "country", "product", "label", "split"
    ]

    if not all(col in predictions.columns for col in required_pred):
        raise ValueError("Predictions have missing columns")

    if not all(col in labels.columns for col in required_label):
        raise ValueError("Labels have missing columns")

    keys = ["asof_year", "country", "product"]

    # Check duplicate predictions
    if predictions.duplicated(keys).any():
        raise ValueError("Duplicate country-product predictions found")

    # Only evaluate rows with known labels in test sets
    test_labels = labels[
        labels["split"].isin(["test_india", "test_peer"])
    ].copy()

    test_labels = test_labels.dropna(subset=["label"])

    # Attach prediction scores to actual labels
    data = test_labels.merge(
        predictions[keys + ["score"]],
        on=keys,
        how="left",
        validate="one_to_one"
    )

    # Every test row must have a valid prediction
    if data["score"].isna().any():
        raise ValueError("Missing prediction scores for some test rows")

    if not np.isfinite(data["score"]).all():
        raise ValueError("Predictions contain invalid scores")

    results = []

    # Evaluate each as-of year
    for year in sorted(data["asof_year"].unique()):

        year_data = data[data["asof_year"] == year]

        subsets = {
            "india": year_data[year_data["split"] == "test_india"],
            "peers": year_data[year_data["split"] == "test_peer"],
            "all": year_data
        }

        for subset_name, subset in subsets.items():

            if len(subset) == 0:
                continue

            y_true = subset["label"].astype(int)
            y_score = subset["score"].astype(float)

            # Overall fraction of successful products
            base_rate = y_true.mean()

            # PR-AUC (average precision), pooled across test rows
            if y_true.nunique() == 2:
                pr_auc = average_precision_score(y_true, y_score)
            else:
                pr_auc = np.nan

            # Rank products separately within each country
            country_p10 = []
            country_p20 = []
            country_r20 = []

            for country, group in subset.groupby("country"):

                group = group.sort_values(
                    "score", ascending=False
                )

                actual = group["label"].astype(int).to_numpy()

                top10 = actual[:10]
                top20 = actual[:20]

                # Precision@10 and Precision@20
                p10 = top10.sum() / len(top10)
                p20 = top20.sum() / len(top20)

                country_p10.append(p10)
                country_p20.append(p20)

                # Recall@20
                total_positive = actual.sum()

                if total_positive > 0:
                    r20 = top20.sum() / total_positive
                    country_r20.append(r20)

            precision10 = np.mean(country_p10)
            precision20 = np.mean(country_p20)

            recall20 = (
                np.mean(country_r20)
                if country_r20 else np.nan
            )

            lift = (
                precision20 / base_rate
                if base_rate > 0 else np.nan
            )

            results.append({
                "asof_year": year,
                "subset": subset_name,
                "Precision@10": precision10,
                "Precision@20": precision20,
                "Recall@20": recall20,
                "PR-AUC": pr_auc,
                "base_rate": base_rate,
                "lift": lift
            })

    return pd.DataFrame(results)


# Load P1 labels
labels = pd.read_parquet("labels_v1.parquet")

print("Labels loaded:", labels.shape)
print("Test rows:", labels["split"].isin(
    ["test_india", "test_peer"]
).sum())

print("\nEvaluation function ready!")
