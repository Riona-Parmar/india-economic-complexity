# P3 Weak-Signal Feature Experiment Report

**Project:** *What Should India Export Next? Predicting Emerging Export Opportunities Using Economic Complexity and Machine Learning*  
**Scope:** P3 predictive modelling and feature experiments  
**Data:** `p3_dataset_v2.parquet` (466,856 rows; 454,863 training rows; four as-of years)  
**Experiment seed:** 42  
**Status:** Completed on the existing historical train-only temporal folds; no India/peer holdout score was used.

## 1. Executive summary

The project predicts which country-product pairs that have not yet established comparative advantage will reach RCA3 >= 1 five years later. The best existing XGBoost candidate uses 13 as-of features and is evaluated by temporal ranking folds. It obtains mean PR-AUC 0.0383, Precision@20 0.0517, Recall@20 0.1596 and Lift@20 4.22. Precision is low in absolute terms, but top-20 ranking is better than random selection at the observed base rate.

This experiment tested export-presence history, export recency, proximity-weighted capability and a small set of motivated interactions. The most complete feasible candidate (F) moved mean Precision@20 from 0.05173 to 0.05305, mean Recall@20 from 0.15962 to 0.16614, and Lift@20 from 4.223 to 4.382. The average fold-specific country-bootstrap interval for the Precision@20 change was -0.00295 to 0.00539, spanning zero; it is not a pooled time-series interval. Mean PR-AUC changed only from 0.03831 to 0.03853; F improved PR-AUC in 2005, but not in 2010 or 2015.

For a training-fold-defined weak-signal positive subgroup (positive rows in the bottom quartile of both current export scale and density), mean country-averaged Recall@20 increased from 0.0231 to 0.0356 with F. Low-density positive Recall@20 rose from 0.0709 to 0.0805, while high-PCI positive Recall@20 fell from 0.0830 to 0.0760. This is a promising, mixed result, not proof of a robust improvement. The three time folds are limited evidence, and existing baseline aggregates still contain cross-country information that includes the holdout countries.

**Recommendation:** retain the existing all-signal candidate as the reference model. Keep F as a research challenger for further validation because its top-20 and weak-signal ranking are directionally better, but do not call it the winning or deployment model on these folds alone.

## 2. Project and label definition

The task is a five-year new-emergence classification/ranking problem. The existing label builder takes an eligible country-product pair when RCA3 is below the configured low threshold in every year from the panel's first year through the as-of year. It assigns a positive label when target-year RCA3 reaches the configured high threshold. The project's current thresholds are RCA3 < 0.25 for continued pre-emergence eligibility and target RCA3 >= 1; the forecast horizon is five years. The labels in the processed dataset were used unchanged.

These thresholds are project-specific operational choices. RCA >= 1 is a conventional revealed-comparative-advantage benchmark; the specific combination of a 0.25 pre-threshold, a sustained-history rule, and five years is not asserted to be universal. The label builder's configuration and source were inspected; this work did not rewrite labels. Sensitivity analysis of thresholds remains future work.

## 3. Existing P3 methodology and baseline

The existing candidate uses XGBoost with a fixed configuration selected in previous P3 work: `max_depth=3`, `min_child_weight=5`, `subsample=0.8`, `colsample_bytree=1.0`, `learning_rate=0.05`, `n_estimators=400`, `random_state=42`, and fold-specific `scale_pos_weight` computed from training labels. No new hyperparameter search was performed.

The 13 baseline features are density, PCI, current RCA3, three-year RCA3 change, ECI, product ubiquity, world export growth, one- and three-year export growth, export scale, current RCA, Atlas distance, and Atlas complexity-related signal. All were intended to describe the as-of year or earlier. The exact historical folds are:

| Validation as-of year | Training as-of years | Training rows | Validation rows | Validation positives |
|---:|---|---:|---:|---:|
| 2005 | 2000 | 131,242 | 117,713 | 2,205 |
| 2010 | 2000, 2005 | 248,955 | 106,471 | 1,136 |
| 2015 | 2000, 2005, 2010 | 355,426 | 99,437 | 850 |

The current code filters to `split == "train"` before forming folds. India and peer holdout rows are not used in these experiments. The validation years are historical reused folds, not an untouched final test set.

## 4. Previous error-analysis evidence and research gap

The existing error-analysis script reproduces the stored all-signal candidate's fold metrics exactly. It uses training rows only. Its analyses found that positives with low scale, low density, low current RCA, weak growth or zero exports were often missed; low-density and some high-PCI products were difficult to rank. These are predictive error patterns, not causal findings. Potential sector patterns had limited sample sizes.

Product-space research has long studied product proximity and countries' tendency to diversify toward related goods. Research also examines relatedness at the extensive margin of bilateral trade, and more recent work applies machine learning to export-product forecasting. Thus, this report does not claim that adding proximity or export-presence variables is novel to the literature. The evidence-based gap for this project is narrower: **the current all-signal model has identifiable weak-signal and low-density misses, and the repository had not tested whether export-history and train-reference-set related-capability signals improve their ranking while retaining overall ranking quality.**

The defensible project contribution is an implementation and evaluation of those feature groups under the existing strict emergence labels and temporal folds, with an explicit weak-signal subgroup analysis and a new related-product reference universe that excludes the designated country holdout set. Its value is empirical for this project, not a claim of first-in-literature novelty.

Relevant literature:

- Hausmann and Klinger, *The Structure of the Product Space and the Evolution of Comparative Advantage*, Harvard CID Working Paper 146 (2007), [paper page](https://www.hks.harvard.edu/centers/cid/publications/faculty-working-papers/structure-product-space-and-evolution-comparative-advantage).
- Kharel, *Product relatedness in the extensive margin of bilateral trade*, *International Economics* 159 (2019), [article page](https://www.sciencedirect.com/science/article/pii/S2110701718301835).
- *Product progression: a machine learning approach to forecasting industrial upgrading*, *Scientific Reports* (2023), [article](https://www.nature.com/articles/s41598-023-28179-x).
- Hidalgo and Hausmann, *The Building Blocks of Economic Complexity*, Harvard CID Working Paper 186 (2009), [paper](https://www.hks.harvard.edu/sites/default/files/centers/cid/files/publications/faculty-working-papers/186.pdf). Its example uses a different low-RCA threshold and horizon, reinforcing that operational label rules vary by study.

## 5. Feature feasibility and definitions

The annual `trade_rca_v1.parquet` panel covers 1995–2024. Extensive-margin features were derived only from years at or before the row's as-of year. The historical absence duration is left-censored before 1995 and capped at 31 years to represent no observed positive export in the available panel. The annual export records are used as provided; missing country-product-year observations are treated as no observed export.

For related capability, the existing historical proximity files exist for all four as-of years, but their source matrices use the global country set, including India/peers. To avoid importing those holdout contributions into the *new* related-product features, the experiment reconstructed RCA and product proximity from export values using only countries assigned to `train`. At each as-of year, training-country RCA is calculated relative to the training-country export universe; `M[c,p]=1` when this reconstructed RCA is at least one. Proximity uses the same symmetric co-competitive-country/max-ubiquity estimator used by the repository's P2 code. For target product `p`, related RCA capability is `sum_k phi[p,k] * RCA_train[c,k] / sum_k phi[p,k]`; related competitive capability substitutes `M[c,k]` for RCA.

| Feature | Definition / source | Economic intuition | Availability and caveat | Status |
|---|---|---|---|---|
| `export_present_current` | 1 if country-product export value > 0 at t | Current extensive-margin status | As-of annual export panel | Implemented |
| `export_present_3y_window` | 1 if any positive export in t-2..t | Recent, possibly intermittent presence | As-of and prior two years | Implemented |
| `export_present_5y_window` | 1 if any positive export in t-4..t | Longer recency window | As-of and prior four years | Implemented |
| `years_since_positive_export` | t minus most recent positive-export year; cap 31 | Distinguishes recent exit from long absence | Left-censored by 1995 panel start | Implemented |
| `positive_export_years_5y` | Count of positive export years in t-4..t | Persistence versus sporadic exports | Five-year as-of window | Implemented |
| `related_rca_capability` | Proximity-weighted mean train-reference RCA | Depth of capability in related goods | RCA and proximity reconstructed from training countries only | Implemented |
| `related_competitive_capability` | Proximity-weighted share of related goods with train-reference RCA >= 1 | Breadth of already-competitive nearby capabilities | Same train-only reconstruction | Implemented |
| `density_x_export_growth` | density x three-year export growth | Growth may have different value in dense product neighborhoods | Inputs as-of; inherited density is global | Implemented |
| `pci_x_density` | PCI x density | Complexity may interact with product-space distance | PCI/density are inherited global aggregates | Implemented |
| `pci_x_export_growth` | PCI x three-year export growth | Growth in complex goods may carry distinct signal | PCI is inherited global aggregate | Implemented |
| `related_capability_x_pci` | related competitive capability x PCI | Nearby capability may matter differently by target complexity | PCI is inherited global aggregate | Implemented |
| `related_capability_x_density` | related competitive capability x density | Separates related-country capability from density context | Density is inherited global aggregate | Implemented |
| `export_scale_x_growth` | log export scale x three-year export growth | Scale conditions interpretation of growth | Both country-product as-of measures | Implemented |
| Related capability weighted by PCI | Would weight competitive neighbors by product complexity | Could test complex capability spillovers | Not added: the PCI source is a global product measure and a country-isolated historical provenance could not be verified | Not implemented |

No annual export-absence values from target years are used. No label, target-year RCA, target export, or future proximity information is a model feature.

## 6. Experiments and evaluation

Six ablations were trained with identical folds, feature preprocessing (none; inputs were complete), model configuration and ranking procedure:

| Candidate | Added feature group |
|---|---|
| A | Existing 13-feature all-signal baseline |
| B | A + five extensive-margin features |
| C | A + two related-capability features |
| D | A + extensive margin + related capability |
| E | A + six economically motivated interactions |
| F | A + all feasible new features |

PR-AUC and ROC-AUC are calculated over all validation country-product rows. Precision@10, Precision@20, and Recall@20 are first calculated within each country and then macro-averaged across countries. Countries with no positives are excluded from mean Recall@20. Lift@20 is mean country Precision@20 divided by the pooled validation base rate. Each country is ranked by score; stable mergesort preserves input row order for exact score ties. The top 10/20 contain at most 10/20 rows per country. Because tied scores are uncommon and the order is deterministic, no fractional tie allocation is used.

Weak-signal positives are validation positives whose `log_export_value` and `density` are both at or below the 25th percentile of their respective training-fold distributions. The subgroup recall is top-20 hits among that subgroup divided by subgroup positives within each country, then averaged over countries with subgroup positives. The high-PCI subgroup uses the training-fold 75th-percentile PCI cut; low-density positives use the training-fold 25th-percentile density cut. These thresholds are fit separately within each training fold. Subgroup metrics are descriptive; denominators are small (weak-positive counts 191, 83, and 74 by fold).

Brier score and training-validation PR-AUC gap are secondary descriptive metrics. Class weighting changes probability calibration; Brier score is not treated as proof of calibration quality. Paired country bootstrap intervals resample countries 2,000 times within each validation year and estimate Precision@20 differences. They measure country-sampling uncertainty within a fold, not uncertainty across historical time periods. No significance claim is made.

## 7. Mean results across three folds

| Candidate | PR-AUC | ROC-AUC | Brier | P@10 | P@20 | R@20 | Lift@20 | Weak-positive R@20 | High-PCI R@20 | Low-density R@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A Baseline | 0.03831 | 0.72786 | 0.15638 | 0.06565 | 0.05173 | 0.15962 | 4.223 | 0.02313 | 0.08302 | 0.07093 |
| B + extensive | 0.03890 | 0.72809 | 0.15504 | 0.06504 | 0.05224 | 0.16115 | 4.306 | 0.02313 | 0.07949 | 0.07646 |
| C + related | 0.03839 | 0.73075 | 0.15567 | 0.06280 | 0.05193 | 0.15898 | 4.298 | 0.03099 | 0.07562 | 0.07399 |
| D + both groups | 0.03815 | 0.73157 | 0.15498 | 0.06565 | 0.05244 | 0.16688 | 4.337 | 0.03099 | 0.07463 | 0.07861 |
| E + interactions | 0.03872 | 0.72974 | 0.15481 | 0.06179 | 0.05152 | 0.16147 | 4.258 | 0.04245 | 0.07072 | 0.09007 |
| F + all feasible | 0.03853 | 0.73167 | 0.15371 | 0.06402 | 0.05305 | 0.16614 | 4.382 | 0.03555 | 0.07599 | 0.08053 |

The complete per-fold metrics and paired intervals are in `outputs/p3_weak_signal_feature_results.csv`; the concise summary is in `outputs/p3_weak_signal_improvement_summary.csv`.

## 8. Per-fold results

Each cell shows PR-AUC / Precision@20 / Recall@20 / weak-positive Recall@20.

| Candidate | 2005 | 2010 | 2015 |
|---|---|---|---|
| A Baseline | .04929 / .06829 / .13208 / .01548 | .03596 / .04909 / .16593 / .02414 | .02968 / .03780 / .18085 / .02976 |
| B + extensive | .04969 / .06707 / .12638 / .01548 | .03693 / .04939 / .16569 / .02414 | .03006 / .04024 / .19139 / .02976 |
| C + related | .05145 / .06585 / .11862 / .01548 | .03486 / .04909 / .16193 / .02989 | .02886 / .04085 / .19640 / .04762 |
| D + both groups | .05199 / .06616 / .13120 / .01548 | .03437 / .05092 / .17119 / .02989 | .02808 / .04024 / .19826 / .04762 |
| E + interactions | .04973 / .06585 / .13701 / .02173 | .03692 / .04817 / .16158 / .03420 | .02951 / .04055 / .18584 / .07143 |
| F + all feasible | .05137 / .06799 / .13398 / .01190 | .03546 / .04939 / .16710 / .04713 | .02876 / .04177 / .19735 / .04762 |

F improves P@20 in the 2010 and 2015 folds but is slightly lower in 2005. PR-AUC increases in 2005 and falls in 2010/2015. Weak-positive recall improves in the latter two folds but declines in 2005. No candidate dominates consistently across all folds and measures.

For completeness, the secondary metrics and training gap for the baseline and full candidate are shown below. The downloadable fold CSV contains these measures for all six candidates.

| Model | Fold | PR-AUC | ROC-AUC | Brier | P@10 | P@20 | R@20 | Lift@20 | Train PR-AUC | Train-validation PR gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 2005 | .04929 | .72107 | .16361 | .08476 | .06829 | .13208 | 3.646 | .09365 | .04436 |
| A | 2010 | .03596 | .72992 | .16103 | .06220 | .04909 | .16593 | 4.601 | .08506 | .04910 |
| A | 2015 | .02968 | .73259 | .14448 | .05000 | .03780 | .18085 | 4.423 | .07236 | .04268 |
| F | 2005 | .05137 | .72749 | .16045 | .07683 | .06799 | .13398 | 3.630 | .09741 | .04604 |
| F | 2010 | .03546 | .73123 | .16179 | .06159 | .04939 | .16710 | 4.629 | .08835 | .05289 |
| F | 2015 | .02876 | .73630 | .13889 | .05366 | .04177 | .19735 | 4.886 | .07506 | .04630 |

## 9. Weak-signal and difficult subgroup interpretation

The weak-positive subgroup has low top-20 recall for every candidate. Candidate F raises the mean from 0.0231 to 0.0356, mostly through the 2010/2015 folds; this subgroup is small, and its 2005 result falls. The all-signal additions are not enough to solve the hard case.

Low-density positive recall improves most consistently for D/E/F over the baseline mean, with E highest at 0.0901. High-PCI positive recall declines for all added groups relative to the baseline's 0.0830. Thus, the experiment supports a limited improvement in ranking weak-signal and low-density positives but does not resolve the high-complexity miss pattern.

F's mean Precision@20 delta is +0.00132 absolute. Averaging the fold-specific paired country-bootstrap interval endpoints gives an indicative range of [-0.00295, +0.00539]; this is not a pooled interval across time and it spans zero. This is evidence of uncertain, modest improvement, not a statistically established gain. There are only three temporal folds, so temporal generalization remains uncertain.

## 10. Ablation and feature interpretation

Extensive features alone (B) produce the highest mean PR-AUC (0.03890), a small gain over A, but little change in weak-signal recall. Related capability alone (C) lifts weak-positive recall from 0.0231 to 0.0310 and low-density recall from 0.0709 to 0.0740, while mean PR-AUC and P@20 are almost unchanged. D has the highest mean Recall@20 (0.1669) and improves low-density recall, while its PR-AUC is lower than baseline. E has the highest mean weak-positive and low-density recall among the individual ablations, but slightly lower P@20 than baseline. F has the best mean P@20 (0.05305) and Lift@20 (4.382), but not the best PR-AUC or weak-positive recall.

Across the three F models, mean XGBoost gain is largest for the three-year export-presence indicator, related-capability x density, related competitive capability, five-year presence count, and related RCA capability. Gain is model-specific feature-use importance; it does not establish an independent effect or causal mechanism. Compared with the existing baseline's SHAP summary, which ranked export scale and density highly, F's importance pattern adds visible history and related-capability signal. However, current RCA and export scale still matter, so the model has not stopped favoring observable capability. The interactions may capture complementary information, but their group-level ablation E does not improve overall P@20. The mean training-minus-validation PR-AUC gap also rises from 0.04538 for A to 0.04841 for F, so the small ranking gain comes with a slightly larger in-sample gap.

No new SHAP run was performed: the prior SHAP analysis already explained the baseline, and gain plus subgroup ablations provide the relevant incremental interpretation here. Individual products or countries are not treated as general evidence.

## 11. Leakage, provenance and reproducibility audit

- The experiment reads the existing P3 dataset and annual trade panel; it does not write to or modify either dataset.
- All fitted models use only `split == "train"`; folds are the exact 2005/2010/2015 historical folds above. No random split is used.
- India and the five peer countries are excluded from the new related-capability reference universe, and no holdout rows enter feature selection, quantile thresholds, model fitting, subgroup decisions, or experiment evaluation.
- New export-history features use no later-than-as-of observations. New related capability rebuilds RCA and product proximity separately at each as-of year from train-country exports.
- The original baseline's density, PCI/ECI, Atlas measures, ubiquity and world growth were constructed as cross-country as-of aggregates. Their inputs include holdout-country observations at that as-of year. These are not future-year features, but they are not country-isolated. A strict holdout-independent end-to-end evaluation would require rebuilding all of those aggregates (and their underlying RCA/complexity measures) with the holdout countries excluded. This experiment isolates the new related features but does not claim the existing baseline is country-isolated.
- The historical proximity generator uses years <= as-of, but its country universe includes holdouts. The new related capability recomputes the same product co-occurrence estimator over train countries only. It uses the same historical export source and avoids future information.
- The data rows are complete for the required feature values in the evaluated train folds. Feature columns and `label` are explicitly separated; labels do not enter feature matrices.
- The previous error-analysis metrics exactly match the stored all-signal candidate metrics for all three folds.
- Runtime versions: Python 3.14.7, pandas 3.1.0, NumPy 2.5.4, scikit-learn 1.9.1, XGBoost 3.4.1, PyArrow 26.0.0.

## 12. Limitations

1. Positive labels are rare (about 0.85%–1.87% per validation fold). Even top-20 precision is only about 3.8%–6.8% by fold, so these scores support triage/ranking, not reliable automatic decisions.
2. There are only three historical validation folds and the same countries recur over time. Country bootstrap intervals do not solve temporal dependence or establish external validity.
3. The folds are reused historical validation data, not a fresh final test set. India/peer holdout outcomes have intentionally not been used here.
4. Baseline global aggregates retain holdout-country contributions. A strict country-isolated full-model evaluation needs a coordinated rebuild of density, PCI/ECI, Atlas, ubiquity and world-growth inputs using a training-country universe.
5. The current ECI and Atlas-related features are project data products with known provenance/measurement limitations; historical availability was checked from the as-of panel, but underlying measures are not independently reconstructed here.
6. The absence-duration feature is left-censored at 1995, and no observed exports in the panel may not mean the product was never exported.
7. The weak-signal subgroup definition uses fold-specific quartiles and a conjunction of scale and density. It is an analysis slice, not an alternative label, and its counts are limited.
8. Class-weighted predicted probabilities are not assumed calibrated. Brier score is descriptive only.
9. PCI and density interactions inherit the baseline aggregate caveat. No causality is claimed for features or errors.
10. The strict RCA3 thresholds and five-year horizon are project choices. Literature and threshold/horizon sensitivity work are still required before strong policy conclusions.

## 13. Conclusion and next steps

Adding extensive-margin history and train-reference-set related capability gives a small, inconsistent improvement in overall ranking. F improves mean P@20, Recall@20, Lift@20, and weak-positive Recall@20, but the Precision@20 confidence interval spans zero; mean PR-AUC barely changes, and high-PCI recall worsens. The work therefore identifies a plausible feature contribution rather than a conclusive model upgrade.

Next steps should be: (1) rebuild all global features with a train-only country universe for a strict evaluation; (2) test threshold and horizon sensitivity without replacing current labels; (3) validate on a genuinely untouched future period or country set; (4) examine subgroup results with larger samples and pre-specified cutoffs; (5) evaluate calibration only after calibration procedures are fit within training folds; and (6) compare against transparent product-space and persistence/rule baselines on exactly the same rows and top-K definition.

## Presentation-ready explanation

**Problem:** India and other countries need a shortlist of products that might become competitive exports in five years. Positive examples are rare, so ordinary accuracy would be misleading.

**Existing approach:** The project trains an XGBoost ranking model on past export, RCA, complexity, growth and product-space signals. It predicts one historical period using only earlier periods, then ranks products within each country.

**Research gap:** Earlier error checks showed that the model often misses products with weak exports and low product-space density. We had not tested whether export-presence history and capability in related products would help those cases.

**Innovation:** We added features for whether and how recently a product was exported, how persistent those exports were, and how much competitive capability the country has in related products. The new related-product network is rebuilt from training countries only. We also tested a small set of economically motivated interactions.

**Method:** Keep the original RCA3 labels and five-year horizon. Train on 2000 to validate 2005; train on 2000 and 2005 to validate 2010; train through 2010 to validate 2015. Compare six feature sets with the same XGBoost settings. Measure PR-AUC and per-country top-10/top-20 precision and recall. Do not use India or peer holdout outcomes.

**Results:** The full feature set raised mean top-20 precision from 5.17% to 5.30% and improved mean top-20 recall from 15.96% to 16.61%. Weak-signal recall also increased on average, but stayed low. The paired uncertainty interval includes no gain, PR-AUC barely moved, and high-complexity recall declined.

**Conclusion:** Related capability and export-history features are promising for prioritizing difficult opportunities, but the gain is small and uncertain. Keep the existing model as the benchmark, treat the expanded model as a challenger, and rebuild all cross-country aggregates before making a strong country-isolated claim.
