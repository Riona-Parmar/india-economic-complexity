import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "p3_dataset_v2.parquet"
RESULTS_PATH = ROOT / "outputs" / "xgb_signal_feature_results.csv"
IMPORTANCE_PATH = ROOT / "outputs" / "xgb_signal_feature_importance.csv"

BASE_FEATURES = [
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

# Test distinct, as-of-only signals not included in the prior momentum trials.
FEATURE_SETS = {
    "export_growth_control": BASE_FEATURES,
    "plus_export_scale": BASE_FEATURES + ["log_export_value"],
    "plus_current_rca": BASE_FEATURES + ["rca_current"],
    "plus_atlas_relations": BASE_FEATURES + [
        "atlas_distance_current",
        "atlas_cog_current",
    ],
    "all_new_signals": BASE_FEATURES + [
        "log_export_value",
        "rca_current",
        "atlas_distance_current",
        "atlas_cog_current",
    ],
}

FOLDS = [
    (2005, [2000]),
    (2010, [2000, 2005]),
    (2015, [2000, 2005, 2010]),
]

# Fixed to the existing fine-tuning PR-AUC leader to isolate feature effects.
MODEL_PARAMETERS = {
    "max_depth": 3,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "learning_rate": 0.05,
    "n_estimators": 400,
}

BOOTSTRAP_ITERATIONS = 2000
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
    metrics = {
        "PR-AUC": (
            average_precision_score(y_true, y_score)
            if y_true.nunique() == 2
            else np.nan
        ),
        "ROC-AUC": (
            roc_auc_score(y_true, y_score)
            if y_true.nunique() == 2
            else np.nan
        ),
    }

    precision10_by_country = []
    precision20_by_country = []
    recall20_by_country = []
    precision20_by_country_name = {}

    for country, group in data.groupby("country"):
        group = group.sort_values("score", ascending=False)
        actual = group["label"].astype(int).to_numpy()
        top10 = actual[:10]
        top20 = actual[:20]

        precision10_by_country.append(top10.sum() / len(top10))
        country_precision20 = top20.sum() / len(top20)
        precision20_by_country.append(country_precision20)
        precision20_by_country_name[country] = country_precision20

        total_positive = actual.sum()
        if total_positive > 0:
            recall20_by_country.append(top20.sum() / total_positive)

    base_rate = y_true.mean()
    precision20 = np.mean(precision20_by_country)
    metrics.update(
        {
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
    )
    return metrics, precision20_by_country_name


def paired_country_bootstrap_delta(
    baseline_by_country,
    candidate_by_country,
    seed,
):
    countries = sorted(set(baseline_by_country) & set(candidate_by_country))
    if not countries:
        raise ValueError("No matching countries for paired bootstrap")

    baseline = np.array([baseline_by_country[c] for c in countries])
    candidate = np.array([candidate_by_country[c] for c in countries])
    deltas = candidate - baseline
    rng = np.random.default_rng(seed)
    samples = rng.integers(
        0,
        len(countries),
        size=(BOOTSTRAP_ITERATIONS, len(countries)),
    )
    bootstrap_means = deltas[samples].mean(axis=1)
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
    return float(deltas.mean()), float(lower), float(upper)


for output_path in [RESULTS_PATH, IMPORTANCE_PATH]:
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing experiment results: {output_path}"
        )

df = pd.read_parquet(DATA_PATH)
all_features = sorted(
    {feature for features in FEATURE_SETS.values() for feature in features}
)
required_columns = set(
    all_features + ["split", "asof_year", "country", "product", "label"]
)
missing_columns = sorted(required_columns.difference(df.columns))
if missing_columns:
    raise ValueError(f"Dataset is missing required columns: {missing_columns}")

# Exclude India and peer holdouts before making any fold partitions.
train_split_data = df.loc[df["split"].eq("train")].copy()
fold_results = []
importance_results = []
country_precision20 = {}

for feature_set_name, features in FEATURE_SETS.items():
    print(f"\n========== FEATURE SET: {feature_set_name} ==========")

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

        X_train = training_data[features]
        y_train = training_data["label"].astype(int)
        if y_train.nunique() != 2:
            raise ValueError(
                f"Training years {training_years} must contain both label classes"
            )
        if X_train.isna().any().any() or validation_data[features].isna().any().any():
            raise ValueError(f"Missing values in feature set {feature_set_name}")

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            **MODEL_PARAMETERS,
        )
        model.fit(X_train, y_train)

        validation_predictions = validation_data[
            ["asof_year", "country", "product"]
        ].copy()
        validation_predictions["score"] = model.predict_proba(
            validation_data[features]
        )[:, 1]
        metrics, country_p20 = evaluate_validation(
            validation_predictions,
            validation_data,
        )
        country_precision20[(feature_set_name, validation_year)] = country_p20

        train_scores = model.predict_proba(X_train)[:, 1]
        metrics["train_PR-AUC"] = average_precision_score(y_train, train_scores)
        metrics["train_ROC-AUC"] = roc_auc_score(y_train, train_scores)
        fold_results.append(
            {
                "result_level": "fold",
                "feature_set": feature_set_name,
                "feature_count": len(features),
                "features": "|".join(features),
                "validation_year": validation_year,
                "training_years": str(training_years),
                "training_rows": len(training_data),
                "training_positive_count": int(y_train.sum()),
                "scale_pos_weight": scale_pos_weight,
                **MODEL_PARAMETERS,
                **metrics,
                "Precision@20_delta_vs_control": np.nan,
                "Precision@20_delta_ci95_low": np.nan,
                "Precision@20_delta_ci95_high": np.nan,
            }
        )

        gain_by_feature = model.get_booster().get_score(importance_type="gain")
        for feature in features:
            importance_results.append(
                {
                    "result_level": "fold",
                    "feature_set": feature_set_name,
                    "validation_year": validation_year,
                    "feature": feature,
                    "gain": float(gain_by_feature.get(feature, 0.0)),
                }
            )
        print(pd.DataFrame([metrics]).to_string(index=False))

baseline_name = "export_growth_control"
for row in fold_results:
    if row["feature_set"] == baseline_name:
        row["Precision@20_delta_vs_control"] = 0.0
        continue

    year = row["validation_year"]
    delta, lower, upper = paired_country_bootstrap_delta(
        country_precision20[(baseline_name, year)],
        country_precision20[(row["feature_set"], year)],
        seed=42 + year,
    )
    row["Precision@20_delta_vs_control"] = delta
    row["Precision@20_delta_ci95_low"] = lower
    row["Precision@20_delta_ci95_high"] = upper

fold_results_df = pd.DataFrame(fold_results)
summary = (
    fold_results_df.groupby(
        ["feature_set", "feature_count", "features"],
        as_index=False,
    )[METRIC_COLUMNS + ["Precision@20_delta_vs_control"]]
    .mean()
)
summary.insert(0, "result_level", "summary")
summary["validation_year"] = np.nan
summary["training_years"] = ""
summary["training_rows"] = np.nan
summary["training_positive_count"] = np.nan
summary["scale_pos_weight"] = np.nan
summary["Precision@20_delta_ci95_low"] = np.nan
summary["Precision@20_delta_ci95_high"] = np.nan
for parameter, value in MODEL_PARAMETERS.items():
    summary[parameter] = value

results = pd.concat([fold_results_df, summary], ignore_index=True)
results.to_csv(RESULTS_PATH, index=False)

importance_df = pd.DataFrame(importance_results)
importance_summary = (
    importance_df.groupby(
        ["feature_set", "feature"],
        as_index=False,
    )["gain"]
    .mean()
)
importance_summary.insert(0, "result_level", "summary")
importance_summary["validation_year"] = np.nan
importance_df = pd.concat(
    [importance_df, importance_summary],
    ignore_index=True,
)
importance_df.to_csv(IMPORTANCE_PATH, index=False)

summary_columns = [
    "feature_set",
    *METRIC_COLUMNS,
    "Precision@20_delta_vs_control",
]
print("\n========== MEAN RESULTS ACROSS TEMPORAL FOLDS ==========")
print(
    summary.sort_values("PR-AUC", ascending=False)[summary_columns].to_string(
        index=False
    )
)
print(
    "\nPaired country-bootstrap intervals are reported per fold for "
    "Precision@20 only; they quantify country sampling uncertainty, "
    "not temporal-fold uncertainty."
)
print(f"\nSaved fold and summary results to: {RESULTS_PATH}")
print(f"Saved gain importance results to: {IMPORTANCE_PATH}")
