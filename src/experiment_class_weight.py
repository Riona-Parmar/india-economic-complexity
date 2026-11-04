
import sys
import pandas as pd
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

# Multipliers applied to the original imbalance weight
WEIGHT_MULTIPLIERS = [0.25, 0.5, 1.0, 2.0]

p2_src = ROOT / "p2" / "data-p2" / "src"
sys.path.insert(0, str(p2_src))

from evaluation import evaluate

labels = pd.read_parquet(
    ROOT / "data" / "processed" / "labels_v1.parquet"
)
labels = labels[labels["asof_year"].isin([2010, 2015])].copy()

all_metrics = []
all_predictions = []

for multiplier in WEIGHT_MULTIPLIERS:

    print(f"\n========== WEIGHT MULTIPLIER: {multiplier} ==========")

    predictions_for_weight = []

    for test_year in [2010, 2015]:

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

        original_weight = (
            (y_train == 0).sum() / (y_train == 1).sum()
        )

        adjusted_weight = original_weight * multiplier

        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=adjusted_weight,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)[:, 1]

        predictions = test_data[
            ["asof_year", "country", "product"]
        ].copy()

        predictions["score"] = probabilities
        predictions["weight_multiplier"] = multiplier

        predictions_for_weight.append(predictions)

        print(
            f"Year {test_year}: "
            f"original weight={original_weight:.2f}, "
            f"adjusted weight={adjusted_weight:.2f}"
        )

    predictions_df = pd.concat(
        predictions_for_weight,
        ignore_index=True
    )

    metrics = evaluate(predictions_df, labels)
    metrics["weight_multiplier"] = multiplier

    all_metrics.append(metrics)
    all_predictions.append(predictions_df)

    print("\nMetrics:")
    print(metrics.to_string(index=False))

# Combine results from all weight settings
final_metrics = pd.concat(all_metrics, ignore_index=True)
final_predictions = pd.concat(all_predictions, ignore_index=True)

metrics_path = ROOT / "outputs" / "class_weight_experiment_metrics.csv"
predictions_path = ROOT / "outputs" / "class_weight_experiment_predictions.parquet"

final_metrics.to_csv(metrics_path, index=False)
final_predictions.to_parquet(predictions_path, index=False)

print("\nSaved metrics:", metrics_path)
print("Saved predictions:", predictions_path)

print("\n========== ALL EXPERIMENT RESULTS ==========")
print(final_metrics.to_string(index=False))
