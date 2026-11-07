import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "p3_dataset_v2.parquet"
OUTPUT_PATH = ROOT / "outputs" / "xgb_finetuning_results.csv"

FEATURES = [
    "density",
    "pci_current",
    "rca3_current",
    "rca3_change_3y",
    "eci_current",
    "f_ubiquity",
    "f_world_growth",
    "log_export_change_1y",
    "log_export_change_3y",
]

FOLDS = [
    (2005, [2000]),
    (2010, [2000, 2005]),
    (2015, [2000, 2005, 2010]),
]

# Balanced 12-configuration design spanning each requested parameter value.
CONFIGURATIONS = [
    (2, 3, 0.8, 0.8),
    (2, 3, 1.0, 1.0),
    (2, 5, 0.8, 1.0),
    (2, 5, 1.0, 0.8),
    (2, 10, 0.8, 0.8),
    (2, 10, 1.0, 1.0),
    (3, 3, 0.8, 0.8),
    (3, 3, 1.0, 1.0),
    (3, 5, 0.8, 1.0),
    (3, 5, 1.0, 0.8),
    (3, 10, 0.8, 1.0),
    (3, 10, 1.0, 0.8),
]

PARAMETER_COLUMNS = [
    "max_depth",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "learning_rate",
    "n_estimators",
]
METRIC_COLUMNS = [
    "PR-AUC",
    "Precision@10",
    "Precision@20",
    "Recall@20",
    "Lift@20",
    "base_rate",
]


def evaluate_validation(predictions, validation_data):
    keys = ["asof_year", "country", "product"]

    data = validation_data[keys + ["label"]].merge(
        predictions[keys + ["score"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )

    if data["score"].isna().any():
        raise ValueError("Missing validation prediction scores")

    y_true = data["label"].astype(int)
    y_score = data["score"].astype(float)
    pr_auc = (
        average_precision_score(y_true, y_score)
        if y_true.nunique() == 2
        else np.nan
    )

    precision10_by_country = []
    precision20_by_country = []
    recall20_by_country = []

    for _, group in data.groupby("country"):
        group = group.sort_values("score", ascending=False)
        actual = group["label"].astype(int).to_numpy()
        top10 = actual[:10]
        top20 = actual[:20]

        precision10_by_country.append(top10.sum() / len(top10))
        precision20_by_country.append(top20.sum() / len(top20))

        total_positive = actual.sum()
        if total_positive > 0:
            recall20_by_country.append(top20.sum() / total_positive)

    base_rate = y_true.mean()
    precision20 = np.mean(precision20_by_country)

    return {
        "PR-AUC": pr_auc,
        "Precision@10": np.mean(precision10_by_country),
        "Precision@20": precision20,
        "Recall@20": (
            np.mean(recall20_by_country)
            if recall20_by_country
            else np.nan
        ),
        "Lift@20": precision20 / base_rate if base_rate > 0 else np.nan,
        "base_rate": base_rate,
        "positive_count": int(y_true.sum()),
        "validation_rows": len(data),
    }


if OUTPUT_PATH.exists():
    raise FileExistsError(
        f"Refusing to overwrite existing experiment results: {OUTPUT_PATH}"
    )

df = pd.read_parquet(DATA_PATH)
required_columns = set(FEATURES + ["split", "asof_year", "country", "product", "label"])
missing_columns = sorted(required_columns.difference(df.columns))
if missing_columns:
    raise ValueError(f"Dataset is missing required columns: {missing_columns}")

# Restrict both training and validation partitions to the train split.
train_split_data = df.loc[df["split"].eq("train")].copy()
results = []

for max_depth, min_child_weight, subsample, colsample_bytree in CONFIGURATIONS:
    config_name = (
        f"depth_{max_depth}_child_{min_child_weight}"
        f"_subsample_{subsample}_colsample_{colsample_bytree}"
    )
    parameters = {
        "max_depth": max_depth,
        "min_child_weight": min_child_weight,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "learning_rate": 0.05,
        "n_estimators": 400,
    }

    print(f"\n========== CONFIGURATION: {config_name} ==========")

    for validation_year, training_years in FOLDS:
        training_data = train_split_data.loc[
            train_split_data["asof_year"].isin(training_years)
        ].copy()
        validation_data = train_split_data.loc[
            train_split_data["asof_year"].eq(validation_year)
        ].copy()

        if training_data.empty:
            raise ValueError(f"No training rows for years {training_years}")
        if validation_data.empty:
            raise ValueError(f"No validation rows for year {validation_year}")

        X_train = training_data[FEATURES]
        y_train = training_data["label"].astype(int)
        if y_train.nunique() != 2:
            raise ValueError(
                f"Training years {training_years} must contain both label classes"
            )

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            **parameters,
        )
        model.fit(X_train, y_train)

        predictions = validation_data[
            ["asof_year", "country", "product"]
        ].copy()
        predictions["score"] = model.predict_proba(
            validation_data[FEATURES]
        )[:, 1]

        metrics = evaluate_validation(predictions, validation_data)
        results.append(
            {
                "result_level": "fold",
                "config": config_name,
                **parameters,
                "scale_pos_weight": scale_pos_weight,
                "validation_year": validation_year,
                "training_years": str(training_years),
                "training_rows": len(training_data),
                "training_positive_count": int(y_train.sum()),
                **metrics,
            }
        )
        print(
            pd.DataFrame([metrics]).to_string(index=False)
        )

fold_results = pd.DataFrame(results)
summary = (
    fold_results.groupby(
        ["config"] + PARAMETER_COLUMNS,
        as_index=False,
    )[METRIC_COLUMNS]
    .mean()
)
summary.insert(0, "result_level", "summary")

output_columns = [
    "result_level",
    "config",
    *PARAMETER_COLUMNS,
    "scale_pos_weight",
    "validation_year",
    "training_years",
    "training_rows",
    "training_positive_count",
    *METRIC_COLUMNS,
    "positive_count",
    "validation_rows",
]
combined_results = pd.concat(
    [fold_results, summary],
    ignore_index=True,
).reindex(columns=output_columns)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
combined_results.to_csv(OUTPUT_PATH, index=False)

ranking_columns = ["config", *PARAMETER_COLUMNS, *METRIC_COLUMNS]
print("\n========== CONFIGURATION LEADERBOARDS (NOT A FINAL MODEL SELECTION) ==========")
for metric in ["PR-AUC", "Precision@20"]:
    ranked = summary.sort_values(metric, ascending=False).head(5)
    print(f"\nTop configurations by mean {metric} across validation folds:")
    print(ranked[ranking_columns].to_string(index=False))

print(f"\nSaved fold-level and summary results to: {OUTPUT_PATH}")
