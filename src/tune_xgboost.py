
import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold, ParameterSampler
from sklearn.metrics import average_precision_score

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

PARAM_GRID = {
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15],
    "n_estimators": [300, 500, 800, 1000],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7, 10]
}

for test_year in [2010, 2015]:

    print(f"\n{'=' * 50}")
    print(f"TUNING XGBOOST FOR {test_year}")
    print(f"{'=' * 50}")

    train_years = [
        y for y in [2000, 2005, 2010, 2015]
        if y < test_year
    ]

    train_data = df[
        (df["split"] == "train") &
        (df["asof_year"].isin(train_years))
    ].copy()

    X = train_data[FEATURES]
    y = train_data["label"].astype(int)
    groups = train_data["country"]

    cv = GroupKFold(n_splits=3)

    # Generate 20 parameter combinations
    candidates = list(ParameterSampler(
        PARAM_GRID,
        n_iter=20,
        random_state=42
    ))

    trial_results = []

    for trial, params in enumerate(candidates, start=1):

        fold_scores = []
        best_iterations = []

        print(f"\nTrial {trial}/20")
        print("Parameters:", params)

        for fold, (train_idx, val_idx) in enumerate(
            cv.split(X, y, groups), start=1
        ):

            X_fold_train = X.iloc[train_idx]
            y_fold_train = y.iloc[train_idx]

            X_fold_val = X.iloc[val_idx]
            y_fold_val = y.iloc[val_idx]

            # Calculate class weight using fold training data only
            negatives = (y_fold_train == 0).sum()
            positives = (y_fold_train == 1).sum()
            scale = negatives / positives

            model = XGBClassifier(
                objective="binary:logistic",
                eval_metric="aucpr",
                early_stopping_rounds=30,
                tree_method="hist",
                scale_pos_weight=scale,
                random_state=42,
                n_jobs=-1,
                **params
            )

            model.fit(
                X_fold_train,
                y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                verbose=False
            )

            val_probabilities = model.predict_proba(
                X_fold_val
            )[:, 1]

            score = average_precision_score(
                y_fold_val,
                val_probabilities
            )

            fold_scores.append(score)
            best_iterations.append(model.best_iteration)

            print(
                f"  Fold {fold}: "
                f"PR-AUC={score:.5f}, "
                f"Best iteration={model.best_iteration}"
            )

        mean_score = np.mean(fold_scores)

        trial_results.append({
            **params,
            "mean_pr_auc": mean_score,
            "fold1_pr_auc": fold_scores[0],
            "fold2_pr_auc": fold_scores[1],
            "fold3_pr_auc": fold_scores[2],
            "mean_best_iteration": np.mean(best_iterations)
        })

        print("Mean PR-AUC:", round(mean_score, 5))

    # Save results for this evaluation year
    results = pd.DataFrame(trial_results)
    results = results.sort_values(
        "mean_pr_auc", ascending=False
    )

    output_path = (
        ROOT / "outputs" /
        f"xgboost_tuning_{test_year}.csv"
    )

    results.to_csv(output_path, index=False)

    print(f"\nBEST PARAMETERS FOR {test_year}:")
    print(results.iloc[0].to_string())

    print("\nSaved:", output_path)
