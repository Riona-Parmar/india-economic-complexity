# P3 Model Improvement and Precision Optimization Report

**Repository:** `D:\export el\india-economic-complexity`  
**Scope:** P3 modelling only  
**Random seed:** 42  
**Validation:** train-split historical temporal folds; no India/peer holdout outcomes used.

## Executive summary

The existing all-signal XGBoost baseline was preserved and refit on the same 13 features to verify score reproduction. The saved Candidate F predictions were reused as-is; its engineered feature matrix was not rebuilt. Additional experiments used the unchanged processed 13-feature baseline matrix. P1/P2 code, datasets and outputs were not modified.

The clearest precision-optimized challenger is XGBoost with `scale_pos_weight = 0.25 x (negative/positive training count)`. Compared with the baseline, its mean PR-AUC rose from 0.03831 to 0.03916, mean Precision@20 from 0.05173 to 0.05335 (+3.14%), and mean Recall@20 from 0.15962 to 0.16332. Its Precision@20 was above baseline in all three folds. The ranking gains are small in absolute terms; the three folds are reused validation data and do not establish a statistically certain or external improvement.

Candidate F is the best balanced option under a transparent descriptive mean-rank across PR-AUC, Precision@20, Recall@20, weak-positive, high-PCI, and low-density recall. It has the highest mean Recall@20 except the shallow model, better weak-positive and low-density ranking than the baseline, and slightly lower Precision@20 than the weight-0.25 model. **Recommendation:** keep XGB-A as the benchmark; present Candidate F as the balanced research candidate and XGB-A-weight-0.25 as the precision-focused challenger. Do not present either as production-ready.

## Problem and research gap

The model ranks eligible country-product pairs that may reach the existing RCA3 >= 1 emergence label five years after the as-of year. Positives are rare, so overall accuracy can look high even if the model finds no emerging products. Earlier analysis found weak-signal and low-density positives difficult to rank. The previous P3 phase added export history and related-product capability; its full Candidate F modestly improved top-20 ranking and weak-signal recall, but the gain was uncertain and high-PCI recall remained difficult.

This phase asks whether class-weight changes, modest regularization, a linear benchmark, or a ranking objective further improve top-K precision without sacrificing recall. This is a project-specific predictive comparison; it does not claim a literature-wide novelty. Product-space proximity and export relatedness are established ideas (see the references in `p3_weak_signal_experiment_report.md`).

## Data, split, and protection record

At task start, `git status --short` showed the pre-existing untracked P3 scripts and weak-signal report; it showed no P1/P2 modifications. The P3 dataset has 466,856 rows, including 454,863 rows marked `train`; the six held-out countries are India, Bangladesh, Indonesia, Mexico, Turkey, and Vietnam. All fitting, fold-specific thresholds, and validation experimentation were restricted to `split == "train"`.

| Validation | Training years | Validation rows | Positives | Pooled base rate |
|---:|---|---:|---:|---:|
| 2005 | 2000 | 117,713 | 2,205 | 1.873% |
| 2010 | 2000, 2005 | 106,471 | 1,136 | 1.067% |
| 2015 | 2000, 2005, 2010 | 99,437 | 850 | 0.855% |

Labels, thresholds, five-year horizon, and P1/P2 features remain unchanged. Candidate F reuses the prior saved train-only validation scores. Its feature matrix and past train-only feature-construction process were not rerun here. All newly fitted models use the existing 13 baseline features: density, PCI, RCA3, three-year RCA3 change, ECI, ubiquity, world growth, one- and three-year export growth, export scale, current RCA, Atlas distance, and Atlas complexity-related signal. Each is an as-of or earlier input according to the existing data build.

Existing density, PCI/ECI, Atlas measures, ubiquity, and world-growth aggregates have inherited cross-country provenance and can include holdout-country contributions. They are contemporaneous as-of aggregates rather than future-year inputs, but full country-isolated provenance remains unverified. No P1/P2 aggregate was rebuilt.

## Experiments and methodology

The exact folds are train 2000 -> validate 2005; train 2000/2005 -> validate 2010; train 2000/2005/2010 -> validate 2015. No random split, target-year input, or holdout outcome was used. All experiments use seed 42.

| Candidate | Model / settings | Features |
|---|---|---|
| XGB-A | Existing configuration, fold ratio class weight | Existing 13-feature baseline; refit to verify saved score reproduction |
| XGB-F | Prior Candidate F predictions | Saved results reused without rebuilding feature matrix |
| XGB-A weight 0.25/0.50/2.00 | Existing XGB parameters; scale positive weight multiplied by 0.25/0.50/2.00 | Same 13 baseline features |
| XGB-A depth 2 | Same parameters except max depth 2 | Same 13 baseline features |
| Logistic-A | StandardScaler and balanced Logistic Regression, C=1 | Same 13 features; scaler fit on each training fold only |
| XGBRanker-A | `rank:pairwise`, 200 trees, depth 3; groups are country x as-of year | Same 13 features |

The four weight multipliers mirror the pre-existing experiment's predeclared grid; the old class-weight results themselves used India/peer test rows and were not consulted for model selection. No LightGBM or CatBoost package is installed, and neither was added. Random Forest was not rerun: the existing train-only temporal baseline has lower mean PR-AUC (0.02975) and Precision@20 (0.03801), alongside a severe train-validation PR gap (about 0.955). Earlier train-only hyperparameter experiments found no universal winner; this phase tests just one shallow-depth regularization check.

PR-AUC and ROC-AUC are global over validation rows. Precision@10/20 and Recall@10/20 are computed within each country and macro-averaged; recall excludes countries with zero positives. Lift@K preserves the prior definition: macro Precision@K divided by pooled base rate. Stable descending score sort is used for country ranks; exact ties preserve input row order. No fractional tie allocation is used.

Classification metrics use one fixed, untuned threshold of 0.5 on classifier scores. Ranking metrics do not use this threshold. For XGBRanker, predictions are ranking scores rather than probabilities, so threshold-based classification metrics are reported as not applicable. The majority baseline always predicts negative. Subgroup cutoffs use training-fold quartiles only: weak positives are positive rows in the bottom quartile of both export scale and density; high PCI uses the training-fold 75th percentile; low density uses the 25th percentile.

## Mean model comparison

All values below are means over the three folds; P@20 mean +/- fold standard deviation is included to show temporal variation. The CSV includes all metrics, fold standard deviations, absolute and relative baseline deltas, threshold metrics, and model parameters.

| Model | PR-AUC mean +/- SD | P@20 mean +/- SD | P@10 | R@10 | R@20 | Lift@20 | Weak R@20 | High-PCI R@20 | Low-density R@20 |
|---|---|---|---|---|---|---|---|---|---|
| XGB-A-baseline | 0.0383 ± 0.0100 | 0.0517 ± 0.0154 | 0.0657 | 0.1061 | 0.1596 | 4.2230 | 0.0231 | 0.0830 | 0.0709 |
| XGB-F-existing | 0.0385 ± 0.0116 | 0.0530 ± 0.0135 | 0.0640 | 0.1057 | 0.1661 | 4.3816 | 0.0356 | 0.0760 | 0.0805 |
| XGB-A-weight_0.25 | 0.0392 ± 0.0107 | 0.0534 ± 0.0161 | 0.0677 | 0.1051 | 0.1633 | 4.3520 | 0.0231 | 0.0751 | 0.0762 |
| XGB-A-weight_0.50 | 0.0392 ± 0.0107 | 0.0509 ± 0.0134 | 0.0665 | 0.1027 | 0.1596 | 4.1948 | 0.0239 | 0.0813 | 0.0728 |
| XGB-A-weight_2.00 | 0.0378 ± 0.0094 | 0.0498 ± 0.0137 | 0.0620 | 0.0989 | 0.1555 | 4.0918 | 0.0231 | 0.0786 | 0.0698 |
| XGB-A-depth2 | 0.0378 ± 0.0115 | 0.0523 ± 0.0151 | 0.0608 | 0.0974 | 0.1680 | 4.2867 | 0.0408 | 0.0732 | 0.0952 |
| Logistic-A-balanced | 0.0310 ± 0.0100 | 0.0460 ± 0.0130 | 0.0553 | 0.0802 | 0.1258 | 3.7754 | 0.0416 | 0.0500 | 0.0833 |
| XGBRanker-A-pairwise | 0.0346 ± 0.0085 | 0.0496 ± 0.0109 | 0.0581 | 0.0980 | 0.1600 | 4.1314 | 0.0271 | 0.0892 | 0.0918 |

Mean improvements versus XGB-A:

| Model | Delta PR-AUC | Delta P@20 | Relative P@20 change (%) | Delta R@20 |
|---|---|---|---|---|
| XGB-F-existing | 0.0002 | 0.0013 | 2.5540 | 0.0065 |
| XGB-A-weight_0.25 | 0.0008 | 0.0016 | 3.1434 | 0.0037 |
| XGB-A-weight_0.50 | 0.0009 | -0.0008 | -1.5717 | -0.0000 |
| XGB-A-weight_2.00 | -0.0005 | -0.0019 | -3.7328 | -0.0041 |
| XGB-A-depth2 | -0.0006 | 0.0006 | 1.1788 | 0.0084 |
| Logistic-A-balanced | -0.0073 | -0.0057 | -11.0020 | -0.0338 |
| XGBRanker-A-pairwise | -0.0037 | -0.0021 | -4.1257 | 0.0004 |

Best PR-AUC: **XGB-A-weight_0.50**. Best Precision@20: **XGB-A-weight_0.25**. Best Recall@20: **XGB-A-depth2**. Best balanced candidate by equal mean rank across the six reported ranking/subgroup metrics: **XGB-F-existing**. This rank average is a transparent comparison aid, not a uniquely correct utility function.

## Fold-wise comparison

Each row reports PR-AUC, country-macro P@10/P@20 and R@10/R@20, Lift@20, and weak-positive Recall@20.

| Model | Fold | PR-AUC | P@10 | P@20 | R@10 | R@20 | Lift@20 | Weak R@20 |
|---|---|---|---|---|---|---|---|---|
| XGB-A-baseline | 2005 | 0.0493 | 0.0848 | 0.0683 | 0.0865 | 0.1321 | 3.6458 | 0.0155 |
| XGB-A-baseline | 2010 | 0.0360 | 0.0622 | 0.0491 | 0.1119 | 0.1659 | 4.6005 | 0.0241 |
| XGB-A-baseline | 2015 | 0.0297 | 0.0500 | 0.0378 | 0.1198 | 0.1808 | 4.4226 | 0.0298 |
| XGB-F-existing | 2005 | 0.0514 | 0.0768 | 0.0680 | 0.0782 | 0.1340 | 3.6295 | 0.0119 |
| XGB-F-existing | 2010 | 0.0355 | 0.0616 | 0.0494 | 0.1162 | 0.1671 | 4.6291 | 0.0471 |
| XGB-F-existing | 2015 | 0.0288 | 0.0537 | 0.0418 | 0.1228 | 0.1973 | 4.8863 | 0.0476 |
| XGB-A-weight_0.25 | 2005 | 0.0508 | 0.0854 | 0.0707 | 0.0822 | 0.1405 | 3.7760 | 0.0155 |
| XGB-A-weight_0.25 | 2010 | 0.0370 | 0.0640 | 0.0503 | 0.1131 | 0.1676 | 4.7148 | 0.0241 |
| XGB-A-weight_0.25 | 2015 | 0.0297 | 0.0537 | 0.0390 | 0.1199 | 0.1818 | 4.5653 | 0.0298 |
| XGB-A-weight_0.50 | 2005 | 0.0508 | 0.0823 | 0.0652 | 0.0857 | 0.1326 | 3.4830 | 0.0119 |
| XGB-A-weight_0.50 | 2010 | 0.0369 | 0.0665 | 0.0488 | 0.1135 | 0.1591 | 4.5719 | 0.0299 |
| XGB-A-weight_0.50 | 2015 | 0.0298 | 0.0506 | 0.0387 | 0.1089 | 0.1871 | 4.5296 | 0.0298 |
| XGB-A-weight_2.00 | 2005 | 0.0480 | 0.0787 | 0.0649 | 0.0825 | 0.1339 | 3.4667 | 0.0155 |
| XGB-A-weight_2.00 | 2010 | 0.0357 | 0.0585 | 0.0460 | 0.1089 | 0.1601 | 4.3148 | 0.0241 |
| XGB-A-weight_2.00 | 2015 | 0.0295 | 0.0488 | 0.0384 | 0.1052 | 0.1725 | 4.4939 | 0.0298 |
| XGB-A-depth2 | 2005 | 0.0506 | 0.0780 | 0.0692 | 0.0794 | 0.1393 | 3.6946 | 0.0155 |
| XGB-A-depth2 | 2010 | 0.0340 | 0.0610 | 0.0476 | 0.1087 | 0.1674 | 4.4576 | 0.0356 |
| XGB-A-depth2 | 2015 | 0.0286 | 0.0433 | 0.0402 | 0.1040 | 0.1974 | 4.7079 | 0.0714 |
| Logistic-A-balanced | 2005 | 0.0416 | 0.0640 | 0.0595 | 0.0646 | 0.1063 | 3.1738 | 0.0077 |
| Logistic-A-balanced | 2010 | 0.0296 | 0.0573 | 0.0451 | 0.0897 | 0.1362 | 4.2290 | 0.0514 |
| Logistic-A-balanced | 2015 | 0.0218 | 0.0445 | 0.0335 | 0.0862 | 0.1349 | 3.9233 | 0.0655 |
| XGBRanker-A-pairwise | 2005 | 0.0436 | 0.0659 | 0.0613 | 0.0680 | 0.1204 | 3.2714 | 0.0017 |
| XGBRanker-A-pairwise | 2010 | 0.0333 | 0.0573 | 0.0479 | 0.0978 | 0.1726 | 4.4862 | 0.0500 |
| XGBRanker-A-pairwise | 2015 | 0.0268 | 0.0512 | 0.0396 | 0.1282 | 0.1869 | 4.6366 | 0.0298 |

The 0.25 weight model improves Precision@20 in all folds: 2005 +0.00244, 2010 +0.00122, and 2015 +0.00122 absolute. Its Recall@20 is also slightly higher in all three folds. PR-AUC rises in 2005/2010 and is essentially unchanged in 2015. This is more consistent than the earlier feature Candidate F on P@20, but remains a small change on reused historical folds. Weight 0.50 has the highest mean PR-AUC but lower mean P@20; increasing weight to 2.00 degrades both PR-AUC and P@20 versus baseline.

Paired country bootstrap intervals for Precision@20 differences (2,000 country resamples within fold; seed 42 + fold year) are saved in `outputs/p3_precision_optimization_bootstrap_deltas.csv`. For weight 0.25 the fold intervals are [-0.00183, 0.00640], [-0.00184, 0.00427], and [-0.00091, 0.00335]. The mean delta is positive in each fold, but every interval includes zero. They represent country sampling uncertainty within each historical period, not uncertainty over time. Do not interpret an interval as independent-fold inference or claim statistical significance.

## Accuracy and threshold interpretation

At threshold 0.5, the pooled mean majority-class accuracy is **98.74%** because only about 1.26% of all validation rows are positive. Every trained classifier has lower mean accuracy than this all-negative baseline, while finding some positives. XGB-A has mean threshold precision 2.78%, recall 54.92%, F1 0.0527, specificity 75.62%, and balanced accuracy 0.6527. The weight-0.25 candidate has threshold accuracy 97.57%, below majority accuracy, but threshold precision 7.81%, recall 8.40%, F1 0.0807.

| Model | Fold | Acc@0.5 | Majority acc | Prec@0.5 | Rec@0.5 | F1@0.5 | Specificity | Balanced acc | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| XGB-A-baseline | 2005 | 0.7435 | 0.9813 | 0.0398 | 0.5488 | 0.0742 | 0.7472 | 0.6480 | 86308.0000 | 29200.0000 | 995.0000 | 1210.0000 |
| XGB-A-baseline | 2010 | 0.7384 | 0.9893 | 0.0238 | 0.5871 | 0.0457 | 0.7400 | 0.6636 | 77950.0000 | 27385.0000 | 469.0000 | 667.0000 |
| XGB-A-baseline | 2015 | 0.7790 | 0.9915 | 0.0198 | 0.5118 | 0.0381 | 0.7813 | 0.6465 | 77029.0000 | 21558.0000 | 415.0000 | 435.0000 |
| XGB-F-existing | 2005 | 0.7518 | 0.9813 | 0.0417 | 0.5578 | 0.0777 | 0.7555 | 0.6567 | 87266.0000 | 28242.0000 | 975.0000 | 1230.0000 |
| XGB-F-existing | 2010 | 0.7359 | 0.9893 | 0.0236 | 0.5871 | 0.0453 | 0.7375 | 0.6623 | 77681.0000 | 27654.0000 | 469.0000 | 667.0000 |
| XGB-F-existing | 2015 | 0.7960 | 0.9915 | 0.0212 | 0.5059 | 0.0407 | 0.7985 | 0.6522 | 78717.0000 | 19870.0000 | 420.0000 | 430.0000 |
| XGB-A-weight_0.25 | 2005 | 0.9636 | 0.9813 | 0.0858 | 0.0980 | 0.0915 | 0.9801 | 0.5390 | 113207.0000 | 2301.0000 | 1989.0000 | 216.0000 |
| XGB-A-weight_0.25 | 2010 | 0.9789 | 0.9893 | 0.0764 | 0.0880 | 0.0818 | 0.9885 | 0.5383 | 104126.0000 | 1209.0000 | 1036.0000 | 100.0000 |
| XGB-A-weight_0.25 | 2015 | 0.9848 | 0.9915 | 0.0721 | 0.0659 | 0.0688 | 0.9927 | 0.5293 | 97866.0000 | 721.0000 | 794.0000 | 56.0000 |
| XGB-A-depth2 | 2005 | 0.7115 | 0.9813 | 0.0390 | 0.6091 | 0.0733 | 0.7135 | 0.6613 | 82411.0000 | 33097.0000 | 862.0000 | 1343.0000 |
| XGB-A-depth2 | 2010 | 0.7224 | 0.9893 | 0.0230 | 0.6030 | 0.0443 | 0.7237 | 0.6633 | 76226.0000 | 29109.0000 | 451.0000 | 685.0000 |
| XGB-A-depth2 | 2015 | 0.7663 | 0.9915 | 0.0198 | 0.5424 | 0.0382 | 0.7683 | 0.6553 | 75742.0000 | 22845.0000 | 389.0000 | 461.0000 |
| Logistic-A-balanced | 2005 | 0.7916 | 0.9813 | 0.0395 | 0.4345 | 0.0725 | 0.7985 | 0.6165 | 92229.0000 | 23279.0000 | 1247.0000 | 958.0000 |
| Logistic-A-balanced | 2010 | 0.7255 | 0.9893 | 0.0211 | 0.5449 | 0.0406 | 0.7275 | 0.6362 | 76627.0000 | 28708.0000 | 517.0000 | 619.0000 |
| Logistic-A-balanced | 2015 | 0.7842 | 0.9915 | 0.0170 | 0.4271 | 0.0327 | 0.7873 | 0.6072 | 77615.0000 | 20972.0000 | 487.0000 | 363.0000 |

Thus, class weighting changes the score distribution and threshold operating point. Accuracy at 0.5 is not a measure of whether the country-wise top-20 ranking is useful, and it is not optimized here. The very high majority accuracy is not evidence that the majority model finds emerging export opportunities (its recall and F1 are zero).

## Ranking signal, countries and score concentration

All candidates have positive top-20 lift over the pooled base rate. XGB-A's mean Lift@20 is 4.22; weight 0.25 is 4.35; F is 4.38. This is better than random selection under the inherited lift definition, while absolute precision remains around 5%.

Scores are not constant or collapsed: mean per-fold score standard deviation is 0.208 for XGB-A and 0.117 for weight 0.25; score quantiles and unique-score counts are included in the fold metrics. Changing the class weight moves the score distribution and the fraction crossing 0.5 substantially. Ranking signal is measurable (PR-AUC well above the pooled base rate), but rarity limits absolute precision and temporal decline in PR-AUC shows a generalization problem as well.

Country performance varies materially. XGB-A's fold-averaged country standard deviation in Precision@20 is 0.064; its observed fold country minimum/maximum average P@20 are 0.000/0.300. Candidate F has a similar spread (0.066). Metrics are macro-averaged to avoid large product baskets dominating, but country heterogeneity remains.

## Weak signals and error analysis

The saved error-profile file compares the as-of characteristics of top-20 hits with top-20 false discoveries for baseline, F, weight-0.25, depth-2, and pairwise ranking models. For baseline, false discoveries have lower median density (0.120 vs 0.136 among hits), current RCA (0.166 vs 0.240), and three-year export growth (1.14 vs 1.92); about 22% have zero current exports, compared with 12% of top-20 hits. This supports the existing weak-signal failure pattern: the model sometimes elevates low-signal products that do not meet the five-year label. False discoveries are products not meeting the project's target-year label; some may still emerge later, outside this five-year horizon. Feature profile differences are associations and not causal explanations.

Candidate F's weak-positive Recall@20 is 0.0356, compared with 0.0231 for baseline. Weight 0.25 does not improve that subgroup (0.0231). The depth-2 model has the highest Recall@20 and improves weak/low-density subgroup ranking, but its mean PR-AUC is lower than baseline. The pairwise ranker underperforms XGB-A in PR-AUC and P@20, although it improves high-PCI and low-density subgroup recall. No method resolves every difficult subgroup.

Among F's mean-gain features, the top ranks are related competitive capability, related RCA capability, recent export-year count, recent export presence, and related capability x density. In the weight-0.25 baseline-feature XGBoost, feature gain is led by baseline RCA/density/growth signals (see feature-importance CSV). Gain importance measures split use within fitted trees and is not causal. Existing SHAP results explain the prior feature candidate; no repeated SHAP analysis was needed here.

## Precision-recall tradeoff and model decision

1. **Did precision improve?** Yes, modestly. Weight 0.25 raises mean P@20 by 0.00163 absolute (+3.14% relative) and raises P@20 in each fold. F raises mean P@20 by 0.00132.
2. **Did PR-AUC improve?** The best mean PR-AUC is weight 0.50 at 0.03918, +0.00087 versus baseline. Weight 0.25 has 0.03916. These are small differences.
3. **Did top-K recall collapse?** No: weight 0.25's mean Recall@20 increases by 0.00370; however, its threshold-0.5 classification recall drops sharply. Keep those tasks separate.
4. **Was improvement consistent?** P@20 and R@20 for weight 0.25 rise in each fold; PR-AUC is effectively flat in 2015. F is more mixed across folds.
5. **Is it meaningful?** The direction is encouraging and may matter for a shortlist, but absolute P@20 rises by only about 0.16 percentage points. Three historical folds and country resampling do not support strong claims.
6. **Better than random?** Yes by Lift@20 (>1); baseline and candidates all rank positives above the pooled-rate benchmark on average. Absolute precision remains low.
7. **P3 benchmark:** retain the current all-signal XGB-A, unchanged.
8. **Presentation candidate:** Candidate F as the balanced research candidate; show weight-0.25 XGB-A as the best precision-focused challenger. This keeps weak-signal relevance visible while acknowledging that no single model wins every metric.
9. **Further work:** only after this report/recommendation, a strict train-only rebuild of inherited global aggregates, a genuinely untouched future test period, and pre-registered subgroup metrics are justified. Do not tune on the India/peer holdout.

## Limitations and leakage/reproducibility audit

- All models use train split rows for fold construction. No `test_india` or `test_peer` rows are used for training, model selection, or error analysis.
- The six held-out countries did not influence the new experiment. Historical old class-weight outputs were excluded because they evaluated on those holdouts; only their pre-existing multiplier grid was used as a candidate design.
- Label definitions, target horizon, source dataset, P1/P2 source code/data, and baseline feature definitions were not changed. No target-year columns or label values are used as predictors.
- The XGB-A fitted scores are checked against saved baseline predictions with maximum absolute score tolerance 1e-7. Candidate F validation scores are read from its prior experiment output and compared on the same train-only folds.
- Training-derived scale weights, subgroup cutoffs, and logistic scaling are computed inside each fold. No threshold is tuned on validation labels.
- XGBRanker uses ranking scores, not calibrated probabilities; its 0.5 threshold metrics are marked not applicable.
- Existing P2 aggregate feature provenance is cross-country and may include held-out countries as as-of contributors. These values are used as supplied, without modifying or rebuilding P1/P2. Consequently, evaluation is train-row-only but not fully country-isolated at the aggregate-feature level.
- Existing Random Forest comparator overfits severely. Logistic regression and pairwise ranking did not outperform the tree candidate overall. LightGBM/CatBoost were not installed; nothing was installed.
- The validation folds are reused, temporally dependent and only three in number. Bootstrap intervals resample countries within a fold, not periods. No statistical significance claim is made.
- Runtime environment: project virtual environment; seed 42; Python/pandas/NumPy/scikit-learn/XGBoost versions are recorded in the companion weak-signal report. P1/P2 directories had no status changes at task start and were rechecked after the experiment.

## Files and presentation explanation

Scripts: `src/experiment_p3_precision_optimization.py` and `src/analyze_p3_precision_optimization.py`. Results: `outputs/p3_precision_optimization_results.csv`, `outputs/p3_precision_optimization_fold_metrics.csv`, `outputs/p3_precision_optimization_predictions.parquet`, `outputs/p3_precision_optimization_feature_importance.csv`, plus paired bootstrap deltas and top-20 error profiles.

**Simple explanation:** “Our model ranks products that could become new competitive exports over five years. We kept our labels and historical splits and compared models only on training-country validation rows. Lowering the positive-class weight slightly improved top-20 precision in all three folds, while an earlier feature-expanded model did best on a balanced set of overall and weak-signal ranking measures. The gain is modest, top-20 precision is still near five percent, and accuracy is misleading because most rows are negative. We keep the old model as the benchmark and present the expanded model as a research candidate, with the weight-adjusted model as a precision-focused alternative.”
