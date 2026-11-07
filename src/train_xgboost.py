
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

all_predictions = []

for test_year in [2010, 2015]:

    train_years = [y for y in [2000, 2005, 2010, 2015] if y < test_year]

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
    model.save_model(
    ROOT / "outputs" / f"xgboost_model_{test_year}.json"
)

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = test_data[
        ["asof_year", "country", "product"]
    ].copy()

    predictions["score"] = probabilities
    all_predictions.append(predictions)

# Combine both evaluation years
predictions_df = pd.concat(all_predictions, ignore_index=True)

# Save predictions
output_path = ROOT / "outputs" / "predictions_xgboost_v1.parquet"
predictions_df.to_parquet(output_path, index=False)

print("Predictions saved:", output_path)
print("Prediction rows:", len(predictions_df))

# Use P2's shared evaluation function
p2_src = ROOT / "p2" / "data-p2" / "src"
sys.path.insert(0, str(p2_src))

from evaluation import evaluate

labels = pd.read_parquet(
    ROOT / "data" / "processed" / "labels_v1.parquet"
)

labels = labels[labels["asof_year"].isin([2010, 2015])].copy()

metrics = evaluate(predictions_df, labels)

print("\nXGBoost metrics using P2 evaluation:")
print(metrics.to_string(index=False))

metrics.to_csv(
    ROOT / "outputs" / "metrics_xgboost_v1.csv",
    index=False
)

print("\nMetrics saved successfully.")
