
import pandas as pd
import numpy as np
from pathlib import Path

from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_parquet(
    ROOT / "data" / "processed" / "p3_dataset_v2.parquet"
)

# Best feature set from our previous ablation experiment
FEATURES = [
    "density",
    "pci_current",
    "rca3_current",
    "rca3_change_3y",
    "eci_current",
    "f_ubiquity",
    "f_world_growth",
    "log_export_change_1y",
    "log_export_change_3y"
]

folds = [
    (2005, [2000]),
    (2010, [2000, 2005]),
    (2015, [2000, 2005, 2010])
]

# Small, controlled hyperparameter experiment
CONFIGS = {
    "current_default": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.3,
        "min_child_weight": 1,
        "subsample": 1.0,
        "colsample_bytree": 1.0
    },

    "shallow_2": {
        "n_estimators": 400,
        "max_depth": 2,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    },

    "depth_3": {
        "n_estimators": 400,
        "max_depth": 3,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    },

    "depth_4": {
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    },

    "slow_learning": {
        "n_estimators": 700,
        "max_depth": 3,
        "learning_rate": 0.03,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    },

    "strong_regularization": {
        "n_estimators": 400,
        "max_depth": 3,
        "learning_rate": 0.05,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    }
}


def evaluate(predictions, validation_data):

    keys = ["asof_year", "country", "product"]

    data = validation_data[
        keys + ["label"]
    ].merge(
        predictions[keys + ["score"]],
        on=keys,
        how="left",
        validate="one_to_one"
    )

    y_true = data["label"].astype(int)
    y_score = data["score"].astype(float)

    pr_auc = average_precision_score(y_true, y_score)

    precision10 = []
    precision20 = []
    recall20 = []

    for country, group in data.groupby("country"):

        group = group.sort_values(
            "score", ascending=False
        )

        actual = group["label"].astype(int).to_numpy()

        top10 = actual[:10]
        top20 = actual[:20]

        precision10.append(top10.mean())
        precision20.append(top20.mean())

        total_positive = actual.sum()

        if total_positive > 0:
            recall20.append(
                top20.sum() / total_positive
            )

    return {
        "PR-AUC": pr_auc,
        "Precision@10": np.mean(precision10),
        "Precision@20": np.mean(precision20),
        "Recall@20": np.mean(recall20) if recall20 else np.nan
    }


results = []

for config_name, params in CONFIGS.items():

    print(f"\n========== CONFIG: {config_name} ==========")

    for validation_year, train_years in folds:

        train_data = df[
            (df["split"] == "train") &
            (df["asof_year"].isin(train_years))
        ].copy()

        validation_data = df[
            (df["split"] == "train") &
            (df["asof_year"] == validation_year)
        ].copy()

        X_train = train_data[FEATURES]
        y_train = train_data["label"].astype(int)

        X_val = validation_data[FEATURES]

        scale_weight = (
            (y_train == 0).sum() /
            (y_train == 1).sum()
        )

        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=scale_weight,
            random_state=42,
            n_jobs=-1,
            **params
        )

        model.fit(X_train, y_train)

        predictions = validation_data[
            ["asof_year", "country", "product"]
        ].copy()

        predictions["score"] = model.predict_proba(X_val)[:, 1]

        metrics = evaluate(predictions, validation_data)

        metrics["config"] = config_name
        metrics["validation_year"] = validation_year

        results.append(metrics)

        print(pd.DataFrame([metrics]).to_string(index=False))


results_df = pd.DataFrame(results)

summary = (
    results_df.groupby("config")
    [["PR-AUC", "Precision@10", "Precision@20", "Recall@20"]]
    .mean()
    .sort_values("Precision@20", ascending=False)
)

output_path = ROOT / "outputs" / "xgb_temporal_tuning_results.csv"

results_df.to_csv(output_path, index=False)

print("\n========== AVERAGE RESULTS ACROSS VALIDATION YEARS ==========")
print(summary.to_string())

print("\nSaved results to:", output_path)
