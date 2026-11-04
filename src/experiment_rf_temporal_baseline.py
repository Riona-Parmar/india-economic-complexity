import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "p3_dataset_v2.parquet"
OUTPUT_PATH = ROOT / "outputs" / "rf_temporal_baseline_results.csv"

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
    "log_export_value",
    "rca_current",
    "atlas_distance_current",
    "atlas_cog_current",
]

FOLDS = [
    (2005, [2000]),
    (2010, [2000, 2005]),
    (2015, [2000, 2005, 2010]),
]

MODEL_PARAMETERS = {
    "n_estimators": 400,
    "max_features": "sqrt",
    "min_samples_leaf": 5,
    "class_weight": "balanced_subsample",
    "random_state": 42,
    "n_jobs": -1,
}

METRIC_COLUMNS = [
    "PR-AUC",
    "ROC-AUC",
    "Precision@10",
    "Precision@20",
    "Recall@20",
    "Lift@20",
    "base_rate",
    "train_PR-AUC",
    "train_ROC-AUC",
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
        "PR-AUC": average_precision_score(y_true, y_score),
        "ROC-AUC": roc_auc_score(y_true, y_score),
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
required_columns = set(
    FEATURES + ["split", "asof_year", "country", "product", "label"]
)
missing_columns = sorted(required_columns.difference(df.columns))
if missing_columns:
    raise ValueError(f"Dataset is missing required columns: {missing_columns}")

# Exclude country holdouts before constructing any fold.
train_split_data = df.loc[df["split"].eq("train")].copy()
results = []

for validation_year, training_years in FOLDS:
    training_data = train_split_data.loc[
        train_split_data["asof_year"].isin(training_years)
    ].copy()
    validation_data = train_split_data.loc[
        train_split_data["asof_year"].eq(validation_year)
    ].copy()
    if training_data.empty or validation_data.empty:
        raise ValueError(
            f"Missing data for fold with validation year {validation_year}"
        )

    X_train = training_data[FEATURES]
    y_train = training_data["label"].astype(int)
    X_validation = validation_data[FEATURES]
    if y_train.nunique() != 2:
        raise ValueError(
            f"Training years {training_years} must contain both label classes"
        )
    if X_train.isna().any().any() or X_validation.isna().any().any():
        raise ValueError("Missing values in Random Forest features")

    model = RandomForestClassifier(**MODEL_PARAMETERS)
    model.fit(X_train, y_train)

    predictions = validation_data[
        ["asof_year", "country", "product"]
    ].copy()
    predictions["score"] = model.predict_proba(X_validation)[:, 1]
    metrics = evaluate_validation(predictions, validation_data)

    train_scores = model.predict_proba(X_train)[:, 1]
    metrics["train_PR-AUC"] = average_precision_score(y_train, train_scores)
    metrics["train_ROC-AUC"] = roc_auc_score(y_train, train_scores)
    results.append(
        {
            "result_level": "fold",
            "model": "random_forest_baseline",
            "validation_year": validation_year,
            "training_years": str(training_years),
            "training_rows": len(training_data),
            "training_positive_count": int(y_train.sum()),
            **MODEL_PARAMETERS,
            **metrics,
        }
    )
    print(pd.DataFrame([metrics]).to_string(index=False))

fold_results = pd.DataFrame(results)
summary = fold_results.groupby(
    ["model"] + list(MODEL_PARAMETERS),
    as_index=False,
)[METRIC_COLUMNS].mean()
summary.insert(0, "result_level", "summary")
summary["validation_year"] = np.nan
summary["training_years"] = ""
summary["training_rows"] = np.nan
summary["training_positive_count"] = np.nan

output = pd.concat([fold_results, summary], ignore_index=True)
output.to_csv(OUTPUT_PATH, index=False)

print("\n========== RANDOM FOREST BASELINE ==========")
print(summary[["model"] + METRIC_COLUMNS].to_string(index=False))
print(f"\nSaved fold and summary results to: {OUTPUT_PATH}")
