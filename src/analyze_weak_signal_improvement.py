"""Summarize saved train-only weak-signal experiment results."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs/p3_weak_signal_feature_results.csv"
SUMMARY = ROOT / "outputs/p3_weak_signal_improvement_summary.csv"
FOLDS = ROOT / "outputs/p3_weak_signal_fold_and_subgroup_metrics.csv"

if SUMMARY.exists() or FOLDS.exists():
    raise FileExistsError("Refusing to overwrite an existing analysis output")
r = pd.read_csv(RESULTS)
means = r.loc[r.result_level.eq("mean_over_3_folds")].copy()
columns = ["feature_set", "PR-AUC", "ROC-AUC", "Brier", "Precision@10", "Precision@20",
           "Recall@20", "Lift@20", "weak_positive_recall20", "high_pci_positive_recall20",
           "low_density_positive_recall20", "PR_AUC_train_minus_validation",
           "Precision@20_delta_vs_baseline", "Precision@20_delta_ci95_low",
           "Precision@20_delta_ci95_high"]
means[columns].to_csv(SUMMARY, index=False)
folds = r.loc[r.result_level.eq("fold"), ["feature_set", "validation_year", "training_years",
    "PR-AUC", "ROC-AUC", "Brier", "Precision@10", "Precision@20", "Recall@20", "Lift@20",
    "weak_positive_count", "weak_positive_recall20", "high_pci_positive_recall20",
    "low_density_positive_recall20", "PR_AUC_train_minus_validation",
    "Precision@20_delta_vs_baseline", "Precision@20_delta_ci95_low",
    "Precision@20_delta_ci95_high"]]
folds.to_csv(FOLDS, index=False)
print(f"Wrote {SUMMARY}")
print(f"Wrote {FOLDS}")
