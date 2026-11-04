
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]

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

# Best parameters from the tuning experiments
BEST_PARAMS = {
    2010: {
        "max_depth": 3,
        "learning_rate": 0.05,
        "n_estimators": 180,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "min_child_weight": 7
    },
    2015: {
        "max_depth": 3,
        "learning_rate": 0.15,
        "n_estimators": 72,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "min_child_weight": 1
    }
}

all_predictions = []

for test_year in [2010, 2015]:

    print(f"\nTraining tuned model for {test_year}")

    train_years = [
        y for y in [2000, 2005, 2010, 2015]
        if y < test_year
    ]

    train_data = df[
        (df["split"] == "train") &
        (df["asof_year"].isin(train_years))
    ].copy()

    test_data = df[
        (df["split"].isin(["test_india", "test_peer"])) &
        (df["asof_year"] == test_year)
    ].copy()

    X_train = train_data[FEATURES]
    y_train = train_data["label"].astype(int)
    X_test = test_data[FEATURES]

    negatives = (y_train == 0).sum()
    positives = (y_train == 1).sum()
    scale = negatives / positives

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        scale_pos_weight=scale,
        random_state=42,
        n_jobs=-1,
        **BEST_PARAMS[test_year]
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = test_data[
        ["asof_year", "country", "product"]
    ].copy()

    predictions["score"] = probabilities
    all_predictions.append(predictions)

    model_path = (
        ROOT / "outputs" /
        f"xgboost_tuned_{test_year}.json"
    )
    model.save_model(model_path)

    print("Model saved:", model_path)

# Combine predictions
predictions_df = pd.concat(
    all_predictions, ignore_index=True
)

predictions_path = (
    ROOT / "outputs" /
    "predictions_xgboost_tuned.parquet"
)

predictions_df.to_parquet(
    predictions_path, index=False
)

print("\nPredictions saved:", predictions_path)

# Evaluate using the shared P2 evaluator
sys.path.insert(
    0, str(ROOT / "p2" / "data-p2" / "src")
)

from evaluation import evaluate

labels = pd.read_parquet(
    ROOT / "data" / "processed" / "labels_v1.parquet"
)

labels = labels[
    labels["asof_year"].isin([2010, 2015])
].copy()

metrics = evaluate(predictions_df, labels)

print("\nTUNED XGBOOST RESULTS:")
print(metrics.to_string(index=False))

metrics.to_csv(
    ROOT / "outputs" /
    "metrics_xgboost_tuned.csv",
    index=False
)

print("\nMetrics saved successfully.")
