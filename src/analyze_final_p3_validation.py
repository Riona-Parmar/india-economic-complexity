"""Summarize the final P3 ranking comparison and tie sensitivity."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PRED = OUT / "p3_final_model_validation_predictions.parquet"
FOLDS = OUT / "p3_final_model_validation_fold_metrics.csv"
SUMMARY = OUT / "p3_final_model_validation_summary.csv"
BOOTSTRAP = OUT / "p3_final_model_validation_bootstrap.csv"
REPORT = OUT / "reports" / "p3_final_model_validation_report.md"
TIES = OUT / "p3_final_model_validation_tie_sensitivity.csv"
KS = [5, 10, 20, 50, 100]


def fractional_topk_precision(group, k):
    """Expected precision under uniform selection among score ties at the K boundary."""
    group = group.sort_values("score", ascending=False, kind="mergesort")
    if len(group) <= k:
        return float(group.label.mean())
    cutoff = group.score.iloc[k - 1]
    above = group.loc[group.score.gt(cutoff)]
    tied = group.loc[group.score.eq(cutoff)]
    slots = k - len(above)
    return float((above.label.sum() + slots * tied.label.mean()) / k)


def main():
    if REPORT.exists() or TIES.exists():
        raise FileExistsError("Refusing to overwrite final P3 report or tie analysis")
    pred = pd.read_parquet(PRED)
    tie_rows = []
    for (model, year), fold in pred.groupby(["model", "asof_year"], sort=True):
        country_groups = list(fold.groupby("country", sort=True))
        for k in KS:
            stable = []
            fractional = []
            boundary_ties = 0
            for _, group in country_groups:
                ordered = group.sort_values(["score", "product"], ascending=[False, True], kind="mergesort")
                stable.append(float(ordered.label.iloc[:k].mean()))
                fractional.append(fractional_topk_precision(group, k))
                if len(group) > k:
                    boundary = ordered.score.iloc[k - 1]
                    boundary_ties += int(group.score.eq(boundary).sum() > 1)
            tie_rows.append({"model": model, "validation_year": year, "K": k,
                             "Precision_country_macro_product_tiebreak": float(np.mean(stable)),
                             "Precision_country_macro_fractional_ties": float(np.mean(fractional)),
                             "countries_with_boundary_tie": boundary_ties})
    tie_df = pd.DataFrame(tie_rows)
    tie_df.to_csv(TIES, index=False)

    folds = pd.read_csv(FOLDS)
    summary = pd.read_csv(SUMMARY)
    boot = pd.read_csv(BOOTSTRAP)
    feature_rows = ["Precision@5_country_macro", "Precision@10_country_macro",
                    "Precision@20_country_macro", "Precision@50_country_macro",
                    "Precision@100_country_macro", "Recall@20_country_macro",
                    "PR_AUC_global", "ROC_AUC_global", "Lift@20_macro",
                    "Weak_positive_recall@20_country_macro", "High_PCI_recall@20_country_macro",
                    "Low_density_recall@20_country_macro"]
    wide = summary[summary.metric.isin(feature_rows)].pivot(index="model", columns="metric", values="mean_over_folds")
    display_names = {
        "XGB-A-baseline": "A: baseline XGBoost",
        "XGB-F-existing": "F: expanded-feature XGBoost",
        "XGB-A-weight_0.25": "A + 0.25 class-weight multiplier",
        "RankEnsemble_equal_A_F_w025": "Equal rank blend (exploratory)",
    }
    cols = ["Precision@5_country_macro", "Precision@10_country_macro", "Precision@20_country_macro",
            "Precision@50_country_macro", "Precision@100_country_macro", "Recall@20_country_macro",
            "PR_AUC_global", "ROC_AUC_global", "Lift@20_macro"]
    table = ["| Model | P@5 | P@10 | P@20 | P@50 | P@100 | R@20 | PR-AUC | ROC-AUC | Lift@20 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model in display_names:
        vals = [wide.loc[model, c] for c in cols]
        formatted = [f"{v:.2%}" for v in vals[:6]] + [f"{v:.4f}" for v in vals[6:8]] + [f"{vals[8]:.2f}"]
        table.append("| " + display_names[model] + " | " + " | ".join(formatted) + " |")

    fold_cols = ["model", "validation_year", "Precision@5_country_macro", "Precision@10_country_macro",
                 "Precision@20_country_macro", "Precision@50_country_macro", "Precision@100_country_macro",
                 "Recall@20_country_macro", "PR_AUC_global"]
    fold_table = ["| Model | Fold | P@5 | P@10 | P@20 | P@50 | P@100 | R@20 | PR-AUC |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, row in folds[fold_cols].iterrows():
        fold_table.append("| " + display_names[row.model] + " | " + str(int(row.validation_year)) + " | " +
                          " | ".join([f"{row[c]:.2%}" for c in fold_cols[2:8]] + [f"{row.PR_AUC_global:.4f}"]) + " |")

    selected_boot = boot[(boot.model.isin(["XGB-A-weight_0.25", "XGB-F-existing", "RankEnsemble_equal_A_F_w025"])) &
                         (boot.baseline.eq("XGB-A-baseline")) & (boot.K.eq(20)) & (boot.metric.eq("precision"))]
    boot_table = ["| Candidate vs A | Fold | Delta P@20 | 95% country-bootstrap interval |",
                  "|---|---:|---:|---:|"]
    for _, row in selected_boot.iterrows():
        boot_table.append(f"| {display_names[row.model]} | {int(row.validation_year)} | {row.delta:+.4f} | [{row.ci95_low:+.4f}, {row.ci95_high:+.4f}] |")

    tie_ens = tie_df[tie_df.model.eq("RankEnsemble_equal_A_F_w025")]
    ties_lines = ["| Fold | K | P@K product tie-break | P@K fractional ties | Countries tied at boundary |",
                  "|---:|---:|---:|---:|---:|"]
    for _, row in tie_ens.iterrows():
        ties_lines.append(f"| {int(row.validation_year)} | {int(row.K)} | {row.Precision_country_macro_product_tiebreak:.2%} | "
                          f"{row.Precision_country_macro_fractional_ties:.2%} | {int(row.countries_with_boundary_tie)} |")

    text = f"""# Final P3 model validation and code cleanup

## Executive summary

This final run compared the unchanged baseline (A), existing expanded-feature candidate (F), prior precision-focused XGBoost (A with a 0.25 class-weight multiplier), and a fixed equal rank blend of those three saved score sets. All outcomes are from the existing train-only temporal validation folds. The blend is exploratory: it raises macro P@20 only marginally over A while sharply lowering pooled PR-AUC and ROC-AUC. The class-weight candidate is the strongest overall precision-oriented option across P@5–P@20 and has the best mean PR-AUC. Candidate F has the strongest weak-signal positive recall and recall at 20. Evidence does not establish a universally superior model.

## What was investigated

Repository inspection covered the P3 dataset, baseline and feature experiments, precision optimization, Random Forest comparison, historical fold construction, error analysis, explainability outputs, saved predictions, and prior experiment reports. Previous work had already tested feature extensions, class weights, shallow trees, and a Random Forest comparator; those were not repeated. The remaining useful, low-cost question was whether a fixed rank blend of the three already evaluated candidates could improve country-level shortlisting without fitting anything to validation labels.

The blend averages within-country normalized rank percentiles from A, F, and the 0.25 weight candidate equally. It uses no labels to fit blend weights and introduces no new input features. The experiment consumes the existing saved train-only predictions, preserving the established split and forecast target.

## Evaluation protocol

Training and validation are the project’s historical folds: train 2000, validate 2005; train 2000+2005, validate 2010; train 2000+2005+2010, validate 2015. India and peer holdout rows were excluded. Labels, eligibility, and the five-year horizon were unchanged. Scores are ranked separately within each country, with product code ascending as the deterministic tie break. Precision@K and Recall@K reported in the main tables are macro averages across countries; pooled selected precision and global PR-AUC/ROC-AUC use validation rows pooled across countries. Lift@20 is macro Precision@20 divided by the pooled fold prevalence, consistent with the existing analyzer. Precision@K naturally rises at smaller K because it describes a shorter top-ranked shortlist; comparisons below use matched K.

## Mean results across folds

""" + "\n".join(table) + """

Precision@K is country-macro; recall is country-macro at K=20; PR-AUC and ROC-AUC are global pooled metrics. The five precision columns provide a direct shortlist-size comparison. The 0.25 candidate is best on average from P@5 through P@20; the rank blend is marginally best at P@50/P@100. These latter differences are tiny and the blend has materially worse global ordering.

## Per-fold precision and ranking performance

""" + "\n".join(fold_table) + """

The weight-0.25 candidate’s P@20 exceeds baseline in each fold (+0.0024, +0.0012, +0.0012). Candidate F exceeds baseline in 2010 and 2015 but is slightly below in 2005. The blend has a small P@20 gain in each fold (+0.0003, +0.0015, +0.0024), but PR-AUC is below baseline in every fold. None of the models is stable enough across the full set of objectives to call it universally best.

## Country-paired uncertainty at P@20

Country bootstrap uses 2,000 paired resamples within each validation fold; it estimates variability across countries within that fold, not temporal-fold uncertainty.

""" + "\n".join(boot_table) + """

Most intervals include zero. The blend’s P@20 interval is above zero only in 2015; a three-fold result with just one such fold is not convincing evidence of a general improvement. The reused folds are also development data, not an untouched final test.

## Weak-signal and difficult-product results

At K=20, mean country-macro weak-positive recall is 0.0231 for A, 0.0356 for F, 0.0231 for the 0.25 candidate, and 0.0250 for the blend. F best supports the project’s weak-signal objective. Mean low-density positive recall is 0.0709, 0.0805, 0.0762, and 0.0915 respectively; the blend’s increase is promising but comes with lower global ranking quality and many score ties. High-PCI recall is 0.0830 for A, 0.0760 for F, 0.0751 for the weighted candidate, and 0.0767 for the blend. No approach resolves the high-complexity challenge. These subgroup figures are descriptive associations, not causal effects.

## Rank overlap and tie sensitivity

The blend does select different opportunities from baseline. At K=20 it shares 207, 153, and 120 positive hits with A in folds 2005, 2010, and 2015; it adds 18, 13, and 12 hits while dropping 17, 8, and 4 baseline hits. The overlap is substantial, so this is a rearrangement of much of the existing shortlist rather than a new discovery system.

Equal rank averaging creates many tied scores. Deterministic product-code ordering is needed for reproducibility but can affect the shortlist at a tie boundary. The sensitivity table recomputes macro Precision@K by fractional expected inclusion among all tied products at the cutoff. The blend’s small gains do not justify treating its product-code tie-break as meaningful ranking information.

""" + "\n".join(ties_lines) + f"""

The table reports the number of countries with a tie at the cutoff (not the number of tied rows). The fractional measure changes only slightly in these folds, but shows the uncertainty in which tied products enter the shortlist. Full CSV: `{TIES.relative_to(ROOT).as_posix()}`.

## Research interpretation and recommendation

The controlled experiment did not demonstrate a robust, broad improvement from ensembling. The class-weight candidate is the precision-focused choice: best mean P@5, P@10, P@20, and PR-AUC among the three XGBoost candidates, with P@20 above A in all three folds. F remains the better choice when weak-signal discovery and recall are prioritized. The equal rank blend should remain exploratory, not the recommended final model, because its global PR-AUC and ROC-AUC fall substantially and boundary ties complicate fine ranking.

For a single project presentation, retain A as the immutable benchmark and present the 0.25 candidate as the modest precision-focused finalist, with F as the weak-signal/recall trade-off. Report all three rather than claiming the 0.25 model dominates. The mean P@20 gain from A is about 0.16 percentage points (5.17% to 5.34%); the absolute value remains low, and country-bootstrap intervals mostly overlap zero.

## Majority-class accuracy context

The validation positive rate is roughly 1% or lower, so an always-negative rule can achieve accuracy near 99% while finding no opportunities. Accuracy is therefore not a useful selection metric here. This final comparison retained the project’s ranking metrics and did not choose models by accuracy. Calibration was not analyzed: these candidates are evaluated as within-country rankings, and no calibrated probabilities are required for the current decision task.

## P3 code cleanup and new files

Cleanup was limited to `src/experiment_p3_precision_optimization.py` and `src/experiment_weak_signal_features.py`: removed an unused import/list and redundant narration; removed a temporary export-presence assignment that was overwritten before use; clarified a comment about the train-only related-product reconstruction. No intended model behavior changed. New P3 files are:

- `src/experiment_final_p3_rank_ensemble.py` — fixed rank-blend evaluation using saved predictions.
- `src/analyze_final_p3_validation.py` — summary, country-bootstrap sensitivity, and tie-aware precision analysis.
- `outputs/p3_final_model_validation_predictions.parquet` and four CSVs for fold metrics, summary, overlap, and bootstrap.
- `outputs/p3_final_model_validation_tie_sensitivity.csv` — fractional inclusion sensitivity for rank ties.
- `outputs/reports/p3_final_model_validation_report.md` — this report.

## Leakage, integrity, and limitations

- The saved prediction rows contain only the three train-only temporal validation folds; the six India/peer holdout countries are absent.
- The ensemble operates only on contemporaneous candidate scores and does not fit on validation labels. Labels are used only for evaluation.
- No target-year predictors, new transformations, or future aggregates were introduced. Existing P1/P2 feature-provenance caveats remain: several product-space and complexity signals are cross-country as-of-year aggregates, and country-isolated construction was not established here. This run does not rebuild or modify them.
- Existing labels, RCA thresholds, eligibility rules, and five-year forecast horizon were preserved.
- Three temporal folds are limited evidence and were already used in prior development. Bootstrap intervals resample countries within folds, not time periods.
- Positives are rare; macro country metrics and subgroup recalls can be noisy. Absolute precision remains low.
- Temporal drift is visible as PR-AUC declines in later folds. No model addresses that uncertainty.
- This comparison is not a new untouched test; no production-readiness or causal claims are supported.

## Presentation-ready explanation

**Problem:** identify products that may become competitive exports about five years later. **Existing method:** XGBoost ranks country-product candidates from economic-complexity, product-space, and trade-history signals. **Gap:** previous error analysis showed that products with weak current exports, low density, and low RCA are often missed. **Final experiment:** compare the unchanged benchmark, existing expanded-feature model, a previously tested class-weight adjustment, and a fixed rank blend using historical years only. **Result:** the class-weight model gives a small precision improvement across folds, while the expanded-feature model finds more weak-signal positives. The blend is not better overall. **Conclusion:** the project has ranking signal but precision is still low; evidence supports presenting the trade-off rather than claiming a definitive winning model.

## Reproducibility

The experiment uses seed 42 for country bootstrap, 2,000 resamples per fold, the saved prediction artifact, and no new dependencies. Run from the repository root with the existing environment: `python src/experiment_final_p3_rank_ensemble.py`, then `python src/analyze_final_p3_validation.py`. Both scripts refuse to overwrite existing outputs.
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")
    print(f"Wrote {REPORT}")
    print(f"Wrote {TIES}")


if __name__ == "__main__":
    main()
