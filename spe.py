
import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = "software_results"
OUTPUT_DIR = os.path.join(INPUT_DIR, "evaluation")

os.makedirs(OUTPUT_DIR, exist_ok=True)

HORIZON = 5
K = 3
INDIA = "IN"

# ============================================================
# 1. LOAD DATA
# ============================================================

rca = pd.read_csv(
    os.path.join(INPUT_DIR, "services_rca.csv")
)

density = pd.read_csv(
    os.path.join(INPUT_DIR, "services_density.csv")
)

proximity = pd.read_csv(
    os.path.join(INPUT_DIR, "services_proximity.csv")
)

for df in [rca, density, proximity]:
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

rca["rca"] = pd.to_numeric(rca["rca"], errors="coerce")
density["density"] = pd.to_numeric(
    density["density"], errors="coerce"
)
proximity["proximity"] = pd.to_numeric(
    proximity["proximity"], errors="coerce"
)

rca = rca.dropna(subset=["year", "rca"])
density = density.dropna(subset=["year", "density"])

rca["year"] = rca["year"].astype(int)
density["year"] = density["year"].astype(int)

# ============================================================
# 2. PREPARE CURRENT AND FUTURE RCA LABELS
# ============================================================

# Current RCA for each country-service-year
current = rca[
    [
        "reporter_code",
        "reporter_name",
        "year",
        "indicator",
        "indicator_name",
        "rca"
    ]
].copy()

# Future RCA, shifted backwards by five years
future = rca[
    [
        "reporter_code",
        "year",
        "indicator",
        "rca"
    ]
].copy()

future = future.rename(
    columns={
        "year": "target_year",
        "rca": "future_rca"
    }
)

# Merge density with current RCA
data = current.merge(
    density[
        [
            "reporter_code",
            "year",
            "indicator",
            "density"
        ]
    ],
    on=["reporter_code", "year", "indicator"],
    how="inner"
)

# Match each current observation to its five-year future RCA
data["target_year"] = data["year"] + HORIZON

data = data.merge(
    future,
    on=["reporter_code", "indicator", "target_year"],
    how="left",
    indicator=True
)

# Do not treat missing future observations as failures
data = data[data["_merge"] == "both"].copy()
data.drop(columns="_merge", inplace=True)

# Candidate must not already have RCA >= 1
data = data[data["rca"] < 1].copy()

# Actual success label
data["label"] = (
    data["future_rca"] >= 1
).astype(int)

# ============================================================
# 3. GENERATE BASELINE SCORES
# ============================================================

# Density score: higher density = stronger connection
# to the country's existing competitive services.
data["density_score"] = data["density"]

# RCA persistence baseline:
# among currently non-competitive services,
# rank those closest to RCA = 1 higher.
data["rca_persistence_score"] = data["rca"]

# ============================================================
# 4. EVALUATION FUNCTION
# ============================================================

def evaluate_model(df, score_column, k=3):

    precisions = []
    recalls = []
    hits = []

    for country, group in df.groupby("reporter_code"):

        group = group.sort_values(
            score_column,
            ascending=False
        )

        top = group.head(k)

        actual_positives = int(group["label"].sum())
        found_positives = int(top["label"].sum())

        precision = (
            found_positives / len(top)
            if len(top) > 0 else 0
        )

        recall = (
            found_positives / actual_positives
            if actual_positives > 0 else np.nan
        )

        precisions.append(precision)

        if actual_positives > 0:
            recalls.append(recall)

        hits.append(
            int(found_positives > 0)
        )

    pr_auc = (
        average_precision_score(
            df["label"],
            df[score_column]
        )
        if df["label"].nunique() > 1
        else np.nan
    )

    return {
        f"Precision@{k}": np.mean(precisions),
        f"Recall@{k}": (
            np.mean(recalls) if recalls else np.nan
        ),
        f"Hit@{k}": np.mean(hits),
        "PR-AUC": pr_auc,
        "Base_rate": df["label"].mean(),
        "Countries_evaluated": df["reporter_code"].nunique(),
        "Candidate_services": len(df),
        "Actual_successes": int(df["label"].sum())
    }


# ============================================================
# 5. EVALUATE EACH AS-OF YEAR
# ============================================================

metrics = []

for year, group in data.groupby("year"):

    density_metrics = evaluate_model(
        group,
        "density_score",
        K
    )

    rca_metrics = evaluate_model(
        group,
        "rca_persistence_score",
        K
    )

    metrics.append({
        "asof_year": year,
        "target_year": year + HORIZON,
        "model": "Density",
        **density_metrics
    })

    metrics.append({
        "asof_year": year,
        "target_year": year + HORIZON,
        "model": "RCA Persistence",
        **rca_metrics
    })

metrics_df = pd.DataFrame(metrics)

metrics_df.to_csv(
    os.path.join(OUTPUT_DIR, "services_evaluation_metrics.csv"),
    index=False
)

# ============================================================
# 6. SAVE ALL HISTORICAL PREDICTIONS
# ============================================================

predictions = []

for (year, country), group in data.groupby(
    ["year", "reporter_code"]
):

    for model, score_column in [
        ("Density", "density_score"),
        ("RCA Persistence", "rca_persistence_score")
    ]:

        ranked = group.sort_values(
            score_column,
            ascending=False
        ).copy()

        ranked["rank"] = range(1, len(ranked) + 1)
        ranked["model"] = model

        predictions.append(
            ranked[
                [
                    "reporter_code",
                    "reporter_name",
                    "year",
                    "target_year",
                    "indicator",
                    "indicator_name",
                    "rca",
                    "density",
                    "future_rca",
                    "label",
                    "model",
                    score_column,
                    "rank"
                ]
            ].rename(
                columns={score_column: "prediction_score"}
            )
        )

predictions_df = pd.concat(
    predictions,
    ignore_index=True
)

predictions_df.to_csv(
    os.path.join(OUTPUT_DIR, "services_historical_predictions.csv"),
    index=False
)

# ============================================================
# 7. LATEST INDIA RECOMMENDATIONS
# ============================================================

latest_year = int(rca["year"].max())

latest_india = current[
    (current["reporter_code"] == INDIA) &
    (current["year"] == latest_year)
].copy()

latest_india = latest_india.merge(
    density[
        (density["reporter_code"] == INDIA) &
        (density["year"] == latest_year)
    ][
        ["reporter_code", "year", "indicator", "density"]
    ],
    on=["reporter_code", "year", "indicator"],
    how="left"
)

# Only recommend services where India does not yet have RCA >= 1
latest_india = latest_india[
    latest_india["rca"] < 1
].copy()

latest_india = latest_india.sort_values(
    "density",
    ascending=False
)

latest_india["density_rank"] = range(
    1, len(latest_india) + 1
)

latest_india.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "india_latest_service_recommendations.csv"
    ),
    index=False
)

# ============================================================
# 8. INDIA HISTORICAL BACKTEST
# ============================================================

india_backtest = predictions_df[
    (predictions_df["reporter_code"] == INDIA) &
    (predictions_df["model"] == "Density")
].copy()

india_backtest.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "india_historical_service_predictions.csv"
    ),
    index=False
)

# ============================================================
# 9. PRINT RESULTS
# ============================================================

print("\n========================================")
print("SERVICE PREDICTION EVALUATION COMPLETE")
print("========================================")

print("\nEvaluation metrics:")
print(metrics_df.to_string(index=False))

print("\nLatest India recommendations:")
print(
    latest_india[
        [
            "indicator",
            "indicator_name",
            "rca",
            "density",
            "density_rank"
        ]
    ].to_string(index=False)
)

print("\nLatest year:", latest_year)
print("Prediction horizon:", HORIZON, "years")
print("Top K:", K)

print("\nFiles saved in:", OUTPUT_DIR)
