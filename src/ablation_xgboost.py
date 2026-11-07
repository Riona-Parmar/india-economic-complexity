
import sys
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_parquet(
    ROOT / "data" / "processed" / "p3_dataset_v2.parquet"
)

# Full model: all seven features
FULL_FEATURES = [
    "density",
    "pci_current",
    "rca3_current",
    "rca3_change_3y",
    "eci_current",
    "f_ubiquity",
    "f_world_growth"
]

# Ablation model: remove density
ABLATION_FEATURES = [
    "pci_current",
    "rca3_current",
    "rca3_change_3y",
    "eci_current",
    "f_ubiquity",
    "f_world_growth"
]

# Use the same v1 parameters for both models
MODEL_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "random_state": 42,
    "n_jobs": -1
}

all_predictions = []

for test_year in [2010, 2015]:

    print(f"\n{'=' * 45}")
    print("ABLATION STUDY:", test_year)

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

    y_train = train_data["label"].astype(int)

    scale = (
        (y_train == 0).sum() /
        (y_train == 1).sum()
    )

    for model_name, features in [
        ("full", FULL_FEATURES),
        ("without_density", ABLATION_FEATURES)
    ]:

        print(f"\nTraining {model_name} model...")

        model = XGBClassifier(
            **MODEL_PARAMS,
            scale_pos_weight=scale
        )

        model.fit(
            train_data[features],
            y_train
        )

        probabilities = model.predict_proba(
            test_data[features]
        )[:, 1]

        predictions = test_data[
            ["asof_year", "country", "product"]
        ].copy()

        predictions["score"] = probabilities
        predictions["model"] = model_name

        all_predictions.append(predictions)

        print("Completed:", model_name)

# Combine predictions
predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)

# Evaluate using the same P2 evaluator
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

for model_name in ["full", "without_density"]:

    model_predictions = predictions_df[
        predictions_df["model"] == model_name
    ].drop(columns="model")

    metrics = evaluate(model_predictions, labels)

    metrics["model"] = model_name

    all_predictions_path = (
        ROOT / "outputs" /
        f"ablation_predictions_{model_name}.parquet"
    )

    model_predictions.to_parquet(
        all_predictions_path,
        index=False
    )

    if model_name == "full":
        all_metrics = metrics
    else:
        all_metrics = pd.concat(
            [all_metrics, metrics],
            ignore_index=True
        )

print("\nABLATION RESULTS:")
print(all_metrics.to_string(index=False))

all_metrics.to_csv(
    ROOT / "outputs" / "ablation_metrics.csv",
    index=False
)

print("\nSaved ablation_metrics.csv")
