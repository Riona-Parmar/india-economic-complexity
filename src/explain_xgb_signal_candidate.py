import numpy as np
import pandas as pd
from pathlib import Path
import shap

from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "p3_dataset_v2.parquet"
SUMMARY_PATH = ROOT / "outputs" / "xgb_signal_shap_summary.csv"
EXAMPLES_PATH = ROOT / "outputs" / "xgb_signal_shap_examples_2015.csv"

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

TRAINING_YEARS = [2000, 2005, 2010]
VALIDATION_YEAR = 2015
EXPLANATION_SAMPLE_SIZE = 10000
TOP_EXAMPLES = 25
SEED = 42

MODEL_PARAMETERS = {
    "max_depth": 3,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "learning_rate": 0.05,
    "n_estimators": 400,
}

for output_path in [SUMMARY_PATH, EXAMPLES_PATH]:
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing explanations: {output_path}"
        )

df = pd.read_parquet(DATA_PATH)
required_columns = set(
    FEATURES + ["split", "asof_year", "country", "product", "label"]
)
missing_columns = sorted(required_columns.difference(df.columns))
if missing_columns:
    raise ValueError(f"Dataset is missing required columns: {missing_columns}")

# Keep India and peer holdouts out of both fitting and explanations.
train_split_data = df.loc[df["split"].eq("train")]
training_data = train_split_data.loc[
    train_split_data["asof_year"].isin(TRAINING_YEARS)
].copy()
validation_data = train_split_data.loc[
    train_split_data["asof_year"].eq(VALIDATION_YEAR)
].copy()
if training_data.empty or validation_data.empty:
    raise ValueError("Training or 2015 train-split validation data is empty")
if training_data["label"].nunique() != 2:
    raise ValueError("Training data must contain both label classes")
if training_data[FEATURES].isna().any().any():
    raise ValueError("Missing values in training features")
if validation_data[FEATURES].isna().any().any():
    raise ValueError("Missing values in validation features")

X_train = training_data[FEATURES]
y_train = training_data["label"].astype(int)
X_validation = validation_data[FEATURES]
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",
    scale_pos_weight=scale_pos_weight,
    random_state=SEED,
    n_jobs=-1,
    **MODEL_PARAMETERS,
)
model.fit(X_train, y_train)

validation_scores = model.predict_proba(X_validation)[:, 1]
validation_ranked = validation_data[
    ["asof_year", "country", "product"]
].copy()
validation_ranked["score"] = validation_scores
top_rows = validation_ranked.sort_values(
    "score",
    ascending=False,
).head(TOP_EXAMPLES)
top_indices = top_rows.index

sample_size = min(EXPLANATION_SAMPLE_SIZE, len(X_validation))
explanation_sample = X_validation.sample(
    n=sample_size,
    random_state=SEED,
)

# TreeSHAP raw-output contributions are additive in the model's log-odds.
explainer = shap.TreeExplainer(model)
sample_shap_values = np.asarray(
    explainer.shap_values(explanation_sample)
)
if sample_shap_values.ndim != 2:
    raise ValueError(
        f"Unexpected SHAP value shape: {sample_shap_values.shape}"
    )

summary = pd.DataFrame(
    {
        "feature": FEATURES,
        "mean_abs_shap_log_odds": np.abs(sample_shap_values).mean(axis=0),
        "mean_shap_log_odds": sample_shap_values.mean(axis=0),
        "explained_rows": sample_size,
        "validation_year": VALIDATION_YEAR,
        "training_years": str(TRAINING_YEARS),
        "feature_set": "all_new_signals",
    }
).sort_values("mean_abs_shap_log_odds", ascending=False)
summary.to_csv(SUMMARY_PATH, index=False)

top_shap_values = np.asarray(explainer.shap_values(X_validation.loc[top_indices]))
if top_shap_values.ndim != 2:
    raise ValueError(f"Unexpected top-row SHAP value shape: {top_shap_values.shape}")

examples = top_rows.reset_index(drop=True).copy()
for index, feature in enumerate(FEATURES):
    examples[f"{feature}__value"] = X_validation.loc[
        top_indices, feature
    ].to_numpy()
    examples[f"{feature}__shap_log_odds"] = top_shap_values[:, index]
examples["training_years"] = str(TRAINING_YEARS)
examples["feature_set"] = "all_new_signals"
examples.to_csv(EXAMPLES_PATH, index=False)

print("Mean absolute SHAP contribution (log-odds), 2015 train-split fold:")
print(summary[["feature", "mean_abs_shap_log_odds", "mean_shap_log_odds"]].to_string(index=False))
print(f"\nTop-ranked internal validation examples saved to: {EXAMPLES_PATH}")
print(f"SHAP summary saved to: {SUMMARY_PATH}")
