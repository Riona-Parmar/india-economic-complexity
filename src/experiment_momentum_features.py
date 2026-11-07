
import sys
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_parquet(
    ROOT / "data" / "processed" / "p3_dataset_v2.parquet"
)

BASE_FEATURES = [
    "density",
    "pci_current",
    "rca3_current",
    "rca3_change_3y",
    "eci_current",
    "f_ubiquity",
    "f_world_growth"
]

ENHANCED_FEATURES = BASE_FEATURES + [
    "log_export_change_1y",
    "log_export_change_3y",
    "rca3_lag1",
    "rca3_lag3"
]

p2_src = ROOT / "p2" / "data-p2" / "src"
sys.path.insert(0, str(p2_src))
from evaluation import evaluate

labels = pd.read_parquet(
    ROOT / "data" / "processed" / "labels_v1.parquet"
)
labels = labels[labels["asof_year"].isin([2010, 2015])].copy()

results = []

for feature_set_name, FEATURES in [
    ("baseline", BASE_FEATURES),
    ("momentum_features", ENHANCED_FEATURES)
]:

    print(f"\n========== {feature_set_name} ==========")

    all_predictions = []

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

        scale = (y_train == 0).sum() / (y_train == 1).sum()

        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=scale,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        predictions = test_data[
            ["asof_year", "country", "product"]
        ].copy()

        predictions["score"] = model.predict_proba(X_test)[:, 1]
        all_predictions.append(predictions)

    predictions_df = pd.concat(all_predictions, ignore_index=True)

    metrics = evaluate(predictions_df, labels)
    metrics["feature_set"] = feature_set_name

    results.append(metrics)

    print(metrics.to_string(index=False))

final_results = pd.concat(results, ignore_index=True)

output_path = ROOT / "outputs" / "momentum_feature_experiment.csv"
final_results.to_csv(output_path, index=False)

print("\n========== FINAL COMPARISON ==========")
print(final_results.to_string(index=False))
print("\nSaved:", output_path)
