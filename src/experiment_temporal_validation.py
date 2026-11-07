
import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score

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

MOMENTUM_FEATURES = BASE_FEATURES + [
    "log_export_change_1y",
    "log_export_change_3y",
    "rca3_lag1",
    "rca3_lag3"
]

# Train only on earlier years; validate on the next year
folds = [
    (2005, [2000]),
    (2010, [2000, 2005]),
    (2015, [2000, 2005, 2010])
]

feature_sets = [
    ("baseline", BASE_FEATURES),
    ("momentum", MOMENTUM_FEATURES)
]


def evaluate_validation(predictions, validation_data):

    keys = ["asof_year", "country", "product"]

    data = validation_data[
        keys + ["label"]
    ].merge(
        predictions[keys + ["score"]],
        on=keys,
        how="left",
        validate="one_to_one"
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

    p10_by_country = []
    p20_by_country = []
    recall_by_country = []

    for country, group in data.groupby("country"):

        group = group.sort_values("score", ascending=False)
        actual = group["label"].astype(int).to_numpy()

        top10 = actual[:10]
        top20 = actual[:20]

        p10_by_country.append(top10.sum() / len(top10))
        p20_by_country.append(top20.sum() / len(top20))

        total_positive = actual.sum()

        if total_positive > 0:
            recall_by_country.append(
                top20.sum() / total_positive
            )

    base_rate = y_true.mean()
    precision20 = np.mean(p20_by_country)

    return {
        "PR-AUC": pr_auc,
        "Precision@10": np.mean(p10_by_country),
        "Precision@20": precision20,
        "Recall@20": (
            np.mean(recall_by_country)
            if recall_by_country else np.nan
        ),
        "base_rate": base_rate,
        "lift": precision20 / base_rate if base_rate > 0 else np.nan,
        "positive_count": int(y_true.sum()),
        "validation_rows": len(data)
    }


results = []

for feature_name, FEATURES in feature_sets:

    print(f"\n========== FEATURE SET: {feature_name} ==========")

    for validation_year, train_years in folds:

        print(
            f"\nValidation year: {validation_year}"
            f" | Training years: {train_years}"
        )

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

        weight = (
            (y_train == 0).sum() / (y_train == 1).sum()
        )

        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=weight,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        predictions = validation_data[
            ["asof_year", "country", "product"]
        ].copy()

        predictions["score"] = model.predict_proba(X_val)[:, 1]

        metrics = evaluate_validation(
            predictions,
            validation_data
        )

        metrics["feature_set"] = feature_name
        metrics["validation_year"] = validation_year
        metrics["training_years"] = str(train_years)

        results.append(metrics)

        print(pd.DataFrame([metrics]).to_string(index=False))

final_results = pd.DataFrame(results)

output_path = ROOT / "outputs" / "temporal_validation_metrics.csv"
final_results.to_csv(output_path, index=False)

print("\n========== ALL TEMPORAL VALIDATION RESULTS ==========")
print(final_results.to_string(index=False))

print("\nSaved:", output_path)
