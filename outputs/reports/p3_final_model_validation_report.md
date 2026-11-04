# Final P3 model validation and code cleanup

## Executive summary

This final run compared the unchanged baseline (A), existing expanded-feature candidate (F), prior precision-focused XGBoost (A with a 0.25 class-weight multiplier), and a fixed equal rank blend of those three saved score sets. All outcomes are from the existing train-only temporal validation folds. The blend is exploratory: it raises macro P@20 only marginally over A while sharply lowering pooled PR-AUC and ROC-AUC. The class-weight candidate is the strongest overall precision-oriented option across P@5–P@20 and has the best mean PR-AUC. Candidate F has the strongest weak-signal positive recall and recall at 20. Evidence does not establish a universally superior model.

## What was investigated

Repository inspection covered the P3 dataset, baseline and feature experiments, precision optimization, Random Forest comparison, historical fold construction, error analysis, explainability outputs, saved predictions, and prior experiment reports. Previous work had already tested feature extensions, class weights, shallow trees, and a Random Forest comparator; those were not repeated. The remaining useful, low-cost question was whether a fixed rank blend of the three already evaluated candidates could improve country-level shortlisting without fitting anything to validation labels.

The blend averages within-country normalized rank percentiles from A, F, and the 0.25 weight candidate equally. It uses no labels to fit blend weights and introduces no new input features. The experiment consumes the existing saved train-only predictions, preserving the established split and forecast target.

## Evaluation protocol

Training and validation are the project’s historical folds: train 2000, validate 2005; train 2000+2005, validate 2010; train 2000+2005+2010, validate 2015. India and peer holdout rows were excluded. Labels, eligibility, and the five-year horizon were unchanged. Scores are ranked separately within each country, with product code ascending as the deterministic tie break. Precision@K and Recall@K reported in the main tables are macro averages across countries; pooled selected precision and global PR-AUC/ROC-AUC use validation rows pooled across countries. Lift@20 is macro Precision@20 divided by the pooled fold prevalence, consistent with the existing analyzer. Precision@K naturally rises at smaller K because it describes a shorter top-ranked shortlist; comparisons below use matched K.

## Mean results across folds

| Model | P@5 | P@10 | P@20 | P@50 | P@100 | R@20 | PR-AUC | ROC-AUC | Lift@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A: baseline XGBoost | 7.85% | 6.57% | 5.17% | 3.71% | 2.92% | 15.96% | 0.0383 | 0.7279 | 4.22 |
| F: expanded-feature XGBoost | 7.60% | 6.40% | 5.30% | 3.77% | 2.93% | 16.61% | 0.0385 | 0.7317 | 4.38 |
| A + 0.25 class-weight multiplier | 8.13% | 6.77% | 5.34% | 3.74% | 2.95% | 16.33% | 0.0392 | 0.7302 | 4.35 |
| Equal rank blend (exploratory) | 8.13% | 6.50% | 5.32% | 3.80% | 2.96% | 16.53% | 0.0316 | 0.6733 | 4.37 |

Precision@K is country-macro; recall is country-macro at K=20; PR-AUC and ROC-AUC are global pooled metrics. The five precision columns provide a direct shortlist-size comparison. The 0.25 candidate is best on average from P@5 through P@20; the rank blend is marginally best at P@50/P@100. These latter differences are tiny and the blend has materially worse global ordering.

## Per-fold precision and ranking performance

| Model | Fold | P@5 | P@10 | P@20 | P@50 | P@100 | R@20 | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A: baseline XGBoost | 2005 | 8.41% | 8.48% | 6.83% | 4.95% | 4.06% | 13.21% | 0.0493 |
| A: baseline XGBoost | 2010 | 8.78% | 6.22% | 4.91% | 3.44% | 2.65% | 16.59% | 0.0360 |
| A: baseline XGBoost | 2015 | 6.34% | 5.00% | 3.78% | 2.73% | 2.04% | 18.08% | 0.0297 |
| F: expanded-feature XGBoost | 2005 | 8.41% | 7.68% | 6.80% | 5.11% | 4.14% | 13.40% | 0.0514 |
| F: expanded-feature XGBoost | 2010 | 7.68% | 6.16% | 4.94% | 3.45% | 2.68% | 16.71% | 0.0355 |
| F: expanded-feature XGBoost | 2015 | 6.71% | 5.37% | 4.18% | 2.76% | 1.98% | 19.73% | 0.0288 |
| A + 0.25 class-weight multiplier | 2005 | 9.15% | 8.54% | 7.07% | 5.00% | 4.20% | 14.05% | 0.0508 |
| A + 0.25 class-weight multiplier | 2010 | 8.90% | 6.40% | 5.03% | 3.49% | 2.62% | 16.76% | 0.0370 |
| A + 0.25 class-weight multiplier | 2015 | 6.34% | 5.37% | 3.90% | 2.72% | 2.03% | 18.18% | 0.0297 |
| Equal rank blend (exploratory) | 2005 | 9.51% | 8.35% | 6.86% | 5.18% | 4.14% | 13.53% | 0.0393 |
| Equal rank blend (exploratory) | 2010 | 8.41% | 5.91% | 5.06% | 3.44% | 2.65% | 16.94% | 0.0305 |
| Equal rank blend (exploratory) | 2015 | 6.46% | 5.24% | 4.02% | 2.78% | 2.08% | 19.12% | 0.0251 |

The weight-0.25 candidate’s P@20 exceeds baseline in each fold (+0.0024, +0.0012, +0.0012). Candidate F exceeds baseline in 2010 and 2015 but is slightly below in 2005. The blend has a small P@20 gain in each fold (+0.0003, +0.0015, +0.0024), but PR-AUC is below baseline in every fold. None of the models is stable enough across the full set of objectives to call it universally best.

## Country-paired uncertainty at P@20

Country bootstrap uses 2,000 paired resamples within each validation fold; it estimates variability across countries within that fold, not temporal-fold uncertainty.

| Candidate vs A | Fold | Delta P@20 | 95% country-bootstrap interval |
|---|---:|---:|---:|
| Equal rank blend (exploratory) | 2005 | +0.0003 | [-0.0027, +0.0034] |
| F: expanded-feature XGBoost | 2005 | -0.0003 | [-0.0058, +0.0055] |
| A + 0.25 class-weight multiplier | 2005 | +0.0024 | [-0.0018, +0.0067] |
| Equal rank blend (exploratory) | 2010 | +0.0015 | [-0.0012, +0.0043] |
| F: expanded-feature XGBoost | 2010 | +0.0003 | [-0.0034, +0.0040] |
| A + 0.25 class-weight multiplier | 2010 | +0.0012 | [-0.0018, +0.0046] |
| Equal rank blend (exploratory) | 2015 | +0.0024 | [+0.0006, +0.0046] |
| F: expanded-feature XGBoost | 2015 | +0.0040 | [+0.0006, +0.0073] |
| A + 0.25 class-weight multiplier | 2015 | +0.0012 | [-0.0009, +0.0034] |

Most intervals include zero. The blend’s P@20 interval is above zero only in 2015; a three-fold result with just one such fold is not convincing evidence of a general improvement. The reused folds are also development data, not an untouched final test.

## Weak-signal and difficult-product results

At K=20, mean country-macro weak-positive recall is 0.0231 for A, 0.0356 for F, 0.0231 for the 0.25 candidate, and 0.0250 for the blend. F best supports the project’s weak-signal objective. Mean low-density positive recall is 0.0709, 0.0805, 0.0762, and 0.0915 respectively; the blend’s increase is promising but comes with lower global ranking quality and many score ties. High-PCI recall is 0.0830 for A, 0.0760 for F, 0.0751 for the weighted candidate, and 0.0767 for the blend. No approach resolves the high-complexity challenge. These subgroup figures are descriptive associations, not causal effects.

## Rank overlap and tie sensitivity

The blend does select different opportunities from baseline. At K=20 it shares 207, 153, and 120 positive hits with A in folds 2005, 2010, and 2015; it adds 18, 13, and 12 hits while dropping 17, 8, and 4 baseline hits. The overlap is substantial, so this is a rearrangement of much of the existing shortlist rather than a new discovery system.

Equal rank averaging creates many tied scores. Deterministic product-code ordering is needed for reproducibility but can affect the shortlist at a tie boundary. The sensitivity table recomputes macro Precision@K by fractional expected inclusion among all tied products at the cutoff. The blend’s small gains do not justify treating its product-code tie-break as meaningful ranking information.

| Fold | K | P@K product tie-break | P@K fractional ties | Countries tied at boundary |
|---:|---:|---:|---:|---:|
| 2005 | 5 | 9.51% | 9.53% | 28 |
| 2005 | 10 | 8.35% | 8.43% | 34 |
| 2005 | 20 | 6.86% | 6.86% | 31 |
| 2005 | 50 | 5.18% | 5.18% | 40 |
| 2005 | 100 | 4.14% | 4.14% | 25 |
| 2010 | 5 | 8.41% | 8.35% | 32 |
| 2010 | 10 | 5.91% | 5.95% | 34 |
| 2010 | 20 | 5.06% | 5.06% | 34 |
| 2010 | 50 | 3.44% | 3.44% | 30 |
| 2010 | 100 | 2.65% | 2.65% | 28 |
| 2015 | 5 | 6.46% | 6.40% | 24 |
| 2015 | 10 | 5.24% | 5.21% | 27 |
| 2015 | 20 | 4.02% | 4.04% | 29 |
| 2015 | 50 | 2.78% | 2.78% | 34 |
| 2015 | 100 | 2.08% | 2.07% | 38 |

The table reports the number of countries with a tie at the cutoff (not the number of tied rows). The fractional measure changes only slightly in these folds, but shows the uncertainty in which tied products enter the shortlist. Full CSV: `outputs/p3_final_model_validation_tie_sensitivity.csv`.

## Research interpretation and recommendation

The controlled experiment did not demonstrate a robust, broad improvement from ensembling. The class-weight candidate is the precision-focused choice: best mean P@5, P@10, P@20, and PR-AUC among the three XGBoost candidates, with P@20 above A in all three folds. F remains the better choice when weak-signal discovery and recall are prioritized. The equal rank blend should remain exploratory, not the recommended final model, because its global PR-AUC and ROC-AUC fall substantially and boundary ties complicate fine ranking.

For a single project presentation, retain A as the immutable benchmark and present the 0.25 candidate as the modest precision-focused finalist, with F as the weak-signal/recall trade-off. Report all three rather than claiming the 0.25 model dominates. The mean P@20 gain from A is about 0.16 percentage points (5.17% to 5.34%); the absolute value remains low, and country-bootstrap intervals mostly overlap zero.

## Majority-class accuracy context

The validation positive rate is roughly 1% or lower, so an always-negative rule can achieve accuracy near 99% while finding no opportunities. Accuracy is therefore not a useful selection metric here. This final comparison retained the project’s ranking metrics and did not choose models by accuracy. Calibration was not analyzed: these candidates are evaluated as within-country rankings, and no calibrated probabilities are required for the current decision task.

## P3 code cleanup and new files

Cleanup was limited to [src/experiment_p3_precision_optimization.py](../../src/experiment_p3_precision_optimization.py) and [src/experiment_weak_signal_features.py](../../src/experiment_weak_signal_features.py): removed an unused import/list and redundant narration; removed a temporary export-presence assignment that was overwritten before use; clarified a comment about the train-only related-product reconstruction. No intended model behavior changed. New P3 files are:

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
