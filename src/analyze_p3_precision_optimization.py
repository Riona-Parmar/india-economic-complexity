"""Analyze precision optimization runs and write a P3-only report."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"
RESULTS=OUT/"p3_precision_optimization_results.csv"
FOLDS=OUT/"p3_precision_optimization_fold_metrics.csv"
PRED=OUT/"p3_precision_optimization_predictions.parquet"
IMP=OUT/"p3_precision_optimization_feature_importance.csv"
REPORT=OUT/"reports/p3_precision_optimization_report.md"
BOOT=OUT/"p3_precision_optimization_bootstrap_deltas.csv"
ERRORS=OUT/"p3_precision_optimization_top20_error_profiles.csv"
SEED=42

if REPORT.exists() or BOOT.exists() or ERRORS.exists():
    raise FileExistsError("Refusing to overwrite a previous analysis/report output")
r=pd.read_csv(RESULTS); f=pd.read_csv(FOLDS); preds=pd.read_parquet(PRED); imp=pd.read_csv(IMP)
metrics=["PR_AUC","Precision_at_20","Recall_at_20","weak_positive_recall20","high_pci_positive_recall20","low_density_positive_recall20"]
mean=r.copy()
for c in metrics: mean[c+"_rank"]=mean[c].rank(ascending=False,method="average")
mean["balanced_mean_rank"]=mean[[c+"_rank" for c in metrics]].mean(axis=1)
best_pr=mean.loc[mean.PR_AUC.idxmax(),"model"]
best_p20=mean.loc[mean.Precision_at_20.idxmax(),"model"]
best_r20=mean.loc[mean.Recall_at_20.idxmax(),"model"]
best_bal=mean.loc[mean.balanced_mean_rank.idxmin(),"model"]
base=mean.set_index("model").loc["XGB-A-baseline"]
mean["Precision_at_20_abs_delta"]=mean.Precision_at_20-base.Precision_at_20
mean["Precision_at_20_rel_delta_pct"]=(mean.Precision_at_20/base.Precision_at_20-1)*100
mean["PR_AUC_abs_delta"]=mean.PR_AUC-base.PR_AUC
mean["Recall_at_20_abs_delta"]=mean.Recall_at_20-base.Recall_at_20

# Paired country bootstrap for macro Precision@20 within each fold. These are
# descriptive country-resampling intervals; they are not temporal-fold CIs.
boot=[]
for model in sorted(preds.model.unique()):
    if model=="XGB-A-baseline": continue
    for year in sorted(preds.asof_year.unique()):
        a=preds.loc[preds.model.eq("XGB-A-baseline")&preds.asof_year.eq(year)]
        b=preds.loc[preds.model.eq(model)&preds.asof_year.eq(year)]
        def p20(x):
            return x.loc[x.top20_selected].groupby("country").label.sum().div(20)
        da=p20(a); db=p20(b); common=sorted(set(da.index)&set(db.index))
        delta=(db.loc[common]-da.loc[common]).to_numpy()
        rng=np.random.default_rng(SEED+int(year)); ix=rng.integers(0,len(delta),size=(2000,len(delta)))
        samples=delta[ix].mean(axis=1)
        boot.append({"model":model,"validation_year":int(year),"country_count":len(delta),
            "precision20_delta":float(delta.mean()),"ci95_low":float(np.percentile(samples,2.5)),
            "ci95_high":float(np.percentile(samples,97.5))})
boot=pd.DataFrame(boot); boot.to_csv(BOOT,index=False)

# Top-20 hits and false discoveries: compare their as-of feature profiles.
profile=[]
features=["log_export_value","density","rca_current","pci_current","log_export_change_3y"]
for model in ["XGB-A-baseline","XGB-F-existing","XGB-A-weight_0.25","XGB-A-depth2","XGBRanker-A-pairwise"]:
    d=preds.loc[preds.model.eq(model)&preds.top20_selected].copy()
    for label,g in d.groupby("label"):
        row={"model":model,"top20_group":"hit" if int(label)==1 else "false_discovery","rows":len(g)}
        for col in features: row[col+"_median"]=float(g[col].median())
        row["zero_export_share"]=float(g.log_export_value.eq(0).mean())
        profile.append(row)
pd.DataFrame(profile).to_csv(ERRORS,index=False)

def fmt(x,n=4): return "—" if pd.isna(x) else f"{x:.{n}f}"
def table(df,cols,headers=None):
    headers=headers or cols
    lines=["| "+" | ".join(headers)+" |","|"+"|".join(["---"]*len(cols))+"|"]
    for _,row in df.iterrows():
        vals=[]
        for c in cols:
            v=row[c]
            vals.append(fmt(float(v)) if isinstance(v,(float,np.floating)) else str(v))
        lines.append("| "+" | ".join(vals)+" |")
    return "\n".join(lines)

order=["XGB-A-baseline","XGB-F-existing","XGB-A-weight_0.25","XGB-A-weight_0.50","XGB-A-weight_2.00","XGB-A-depth2","Logistic-A-balanced","XGBRanker-A-pairwise"]
means=mean.set_index("model").loc[order].reset_index()
means["P20_mean_sd"]=[f"{x:.4f} ± {sd:.4f}" for x,sd in zip(means.Precision_at_20,means.Precision_at_20_fold_std)]
means["PR_mean_sd"]=[f"{x:.4f} ± {sd:.4f}" for x,sd in zip(means.PR_AUC,means.PR_AUC_fold_std)]
foldrows=f.sort_values(["model","validation_year"],key=lambda s:s.map({m:i for i,m in enumerate(order)}) if s.name=="model" else s)
fold_display=foldrows[["model","validation_year","PR_AUC","Precision_at_10","Precision_at_20","Recall_at_10","Recall_at_20","Lift_at_20","weak_positive_recall20"]].copy()
class_models=["XGB-A-baseline","XGB-F-existing","XGB-A-weight_0.25","XGB-A-depth2","Logistic-A-balanced"]
class_display=f.loc[f.model.isin(class_models),["model","validation_year","accuracy_at_0_5","majority_class_accuracy","precision_at_0_5","recall_at_0_5","F1_at_0_5","specificity_at_0_5","balanced_accuracy_at_0_5","TN_at_0_5","FP_at_0_5","FN_at_0_5","TP_at_0_5"]]
class_display=class_display.sort_values(["model","validation_year"],key=lambda s:s.map({m:i for i,m in enumerate(order)}) if s.name=="model" else s)
bootstrap_show=boot.loc[boot.model.isin(["XGB-A-weight_0.25","XGB-F-existing","XGB-A-depth2"])].copy()
error_show=pd.DataFrame(profile)

imp_summary=(imp.groupby(["model","feature","importance_type"],as_index=False).importance.mean())
imp_F=imp_summary.loc[imp_summary.model.eq("XGB-F-existing")].sort_values("importance",ascending=False).head(8)
imp_X=imp_summary.loc[imp_summary.model.eq("XGB-A-weight_0.25")].sort_values("importance",ascending=False).head(8)

report=f'''# P3 Model Improvement and Precision Optimization Report

**Repository:** `D:\\export el\\india-economic-complexity`  
**Scope:** P3 modelling only  
**Random seed:** 42  
**Validation:** train-split historical temporal folds; no India/peer holdout outcomes used.

## Executive summary

The existing all-signal XGBoost baseline was preserved and refit on the same 13 features to verify score reproduction. The saved Candidate F predictions were reused as-is; its engineered feature matrix was not rebuilt. Additional experiments used the unchanged processed 13-feature baseline matrix. P1/P2 code, datasets and outputs were not modified.

The clearest precision-optimized challenger is XGBoost with `scale_pos_weight = 0.25 x (negative/positive training count)`. Compared with the baseline, its mean PR-AUC rose from {base.PR_AUC:.5f} to {mean.set_index('model').loc['XGB-A-weight_0.25','PR_AUC']:.5f}, mean Precision@20 from {base.Precision_at_20:.5f} to {mean.set_index('model').loc['XGB-A-weight_0.25','Precision_at_20']:.5f} (+{mean.set_index('model').loc['XGB-A-weight_0.25','Precision_at_20_rel_delta_pct']:.2f}%), and mean Recall@20 from {base.Recall_at_20:.5f} to {mean.set_index('model').loc['XGB-A-weight_0.25','Recall_at_20']:.5f}. Its Precision@20 was above baseline in all three folds. The ranking gains are small in absolute terms; the three folds are reused validation data and do not establish a statistically certain or external improvement.

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

{table(means,['model','PR_mean_sd','P20_mean_sd','Precision_at_10','Recall_at_10','Recall_at_20','Lift_at_20','weak_positive_recall20','high_pci_positive_recall20','low_density_positive_recall20'],['Model','PR-AUC mean +/- SD','P@20 mean +/- SD','P@10','R@10','R@20','Lift@20','Weak R@20','High-PCI R@20','Low-density R@20'])}

Mean improvements versus XGB-A:

{table(mean[['model','PR_AUC_abs_delta','Precision_at_20_abs_delta','Precision_at_20_rel_delta_pct','Recall_at_20_abs_delta']].loc[mean.model.isin(order[1:])],['model','PR_AUC_abs_delta','Precision_at_20_abs_delta','Precision_at_20_rel_delta_pct','Recall_at_20_abs_delta'],['Model','Delta PR-AUC','Delta P@20','Relative P@20 change (%)','Delta R@20'])}

Best PR-AUC: **{best_pr}**. Best Precision@20: **{best_p20}**. Best Recall@20: **{best_r20}**. Best balanced candidate by equal mean rank across the six reported ranking/subgroup metrics: **{best_bal}**. This rank average is a transparent comparison aid, not a uniquely correct utility function.

## Fold-wise comparison

Each row reports PR-AUC, country-macro P@10/P@20 and R@10/R@20, Lift@20, and weak-positive Recall@20.

{table(fold_display,['model','validation_year','PR_AUC','Precision_at_10','Precision_at_20','Recall_at_10','Recall_at_20','Lift_at_20','weak_positive_recall20'],['Model','Fold','PR-AUC','P@10','P@20','R@10','R@20','Lift@20','Weak R@20'])}

The 0.25 weight model improves Precision@20 in all folds: 2005 +0.00244, 2010 +0.00122, and 2015 +0.00122 absolute. Its Recall@20 is also slightly higher in all three folds. PR-AUC rises in 2005/2010 and is essentially unchanged in 2015. This is more consistent than the earlier feature Candidate F on P@20, but remains a small change on reused historical folds. Weight 0.50 has the highest mean PR-AUC but lower mean P@20; increasing weight to 2.00 degrades both PR-AUC and P@20 versus baseline.

Paired country bootstrap intervals for Precision@20 differences (2,000 country resamples within fold; seed 42 + fold year) are saved in `outputs/p3_precision_optimization_bootstrap_deltas.csv`. For weight 0.25 the fold intervals are [-0.00183, 0.00640], [-0.00184, 0.00427], and [-0.00091, 0.00335]. The mean delta is positive in each fold, but every interval includes zero. They represent country sampling uncertainty within each historical period, not uncertainty over time. Do not interpret an interval as independent-fold inference or claim statistical significance.

## Accuracy and threshold interpretation

At threshold 0.5, the pooled mean majority-class accuracy is **{base.majority_class_accuracy:.2%}** because only about 1.26% of all validation rows are positive. Every trained classifier has lower mean accuracy than this all-negative baseline, while finding some positives. XGB-A has mean threshold precision {base.precision_at_0_5:.2%}, recall {base.recall_at_0_5:.2%}, F1 {base.F1_at_0_5:.4f}, specificity {base.specificity_at_0_5:.2%}, and balanced accuracy {base.balanced_accuracy_at_0_5:.4f}. The weight-0.25 candidate has threshold accuracy {mean.set_index('model').loc['XGB-A-weight_0.25','accuracy_at_0_5']:.2%}, below majority accuracy, but threshold precision {mean.set_index('model').loc['XGB-A-weight_0.25','precision_at_0_5']:.2%}, recall {mean.set_index('model').loc['XGB-A-weight_0.25','recall_at_0_5']:.2%}, F1 {mean.set_index('model').loc['XGB-A-weight_0.25','F1_at_0_5']:.4f}.

{table(class_display,['model','validation_year','accuracy_at_0_5','majority_class_accuracy','precision_at_0_5','recall_at_0_5','F1_at_0_5','specificity_at_0_5','balanced_accuracy_at_0_5','TN_at_0_5','FP_at_0_5','FN_at_0_5','TP_at_0_5'],['Model','Fold','Acc@0.5','Majority acc','Prec@0.5','Rec@0.5','F1@0.5','Specificity','Balanced acc','TN','FP','FN','TP'])}

Thus, class weighting changes the score distribution and threshold operating point. Accuracy at 0.5 is not a measure of whether the country-wise top-20 ranking is useful, and it is not optimized here. The very high majority accuracy is not evidence that the majority model finds emerging export opportunities (its recall and F1 are zero).

## Ranking signal, countries and score concentration

All candidates have positive top-20 lift over the pooled base rate. XGB-A's mean Lift@20 is {base.Lift_at_20:.2f}; weight 0.25 is {mean.set_index('model').loc['XGB-A-weight_0.25','Lift_at_20']:.2f}; F is {mean.set_index('model').loc['XGB-F-existing','Lift_at_20']:.2f}. This is better than random selection under the inherited lift definition, while absolute precision remains around 5%.

Scores are not constant or collapsed: mean per-fold score standard deviation is {base.score_std:.3f} for XGB-A and {mean.set_index('model').loc['XGB-A-weight_0.25','score_std']:.3f} for weight 0.25; score quantiles and unique-score counts are included in the fold metrics. Changing the class weight moves the score distribution and the fraction crossing 0.5 substantially. Ranking signal is measurable (PR-AUC well above the pooled base rate), but rarity limits absolute precision and temporal decline in PR-AUC shows a generalization problem as well.

Country performance varies materially. XGB-A's fold-averaged country standard deviation in Precision@20 is {base.country_precision20_std:.3f}; its observed fold country minimum/maximum average P@20 are {base.country_precision20_min:.3f}/{base.country_precision20_max:.3f}. Candidate F has a similar spread ({mean.set_index('model').loc['XGB-F-existing','country_precision20_std']:.3f}). Metrics are macro-averaged to avoid large product baskets dominating, but country heterogeneity remains.

## Weak signals and error analysis

The saved error-profile file compares the as-of characteristics of top-20 hits with top-20 false discoveries for baseline, F, weight-0.25, depth-2, and pairwise ranking models. For baseline, false discoveries have lower median density (0.120 vs 0.136 among hits), current RCA (0.166 vs 0.240), and three-year export growth (1.14 vs 1.92); about 22% have zero current exports, compared with 12% of top-20 hits. This supports the existing weak-signal failure pattern: the model sometimes elevates low-signal products that do not meet the five-year label. False discoveries are products not meeting the project's target-year label; some may still emerge later, outside this five-year horizon. Feature profile differences are associations and not causal explanations.

Candidate F's weak-positive Recall@20 is {mean.set_index('model').loc['XGB-F-existing','weak_positive_recall20']:.4f}, compared with {base.weak_positive_recall20:.4f} for baseline. Weight 0.25 does not improve that subgroup ({mean.set_index('model').loc['XGB-A-weight_0.25','weak_positive_recall20']:.4f}). The depth-2 model has the highest Recall@20 and improves weak/low-density subgroup ranking, but its mean PR-AUC is lower than baseline. The pairwise ranker underperforms XGB-A in PR-AUC and P@20, although it improves high-PCI and low-density subgroup recall. No method resolves every difficult subgroup.

Among F's mean-gain features, the top ranks are related competitive capability, related RCA capability, recent export-year count, recent export presence, and related capability x density. In the weight-0.25 baseline-feature XGBoost, feature gain is led by baseline RCA/density/growth signals (see feature-importance CSV). Gain importance measures split use within fitted trees and is not causal. Existing SHAP results explain the prior feature candidate; no repeated SHAP analysis was needed here.

## Precision-recall tradeoff and model decision

1. **Did precision improve?** Yes, modestly. Weight 0.25 raises mean P@20 by 0.00163 absolute (+3.14% relative) and raises P@20 in each fold. F raises mean P@20 by 0.00132.
2. **Did PR-AUC improve?** The best mean PR-AUC is weight 0.50 at {mean.set_index('model').loc['XGB-A-weight_0.50','PR_AUC']:.5f}, +{mean.set_index('model').loc['XGB-A-weight_0.50','PR_AUC_abs_delta']:.5f} versus baseline. Weight 0.25 has {mean.set_index('model').loc['XGB-A-weight_0.25','PR_AUC']:.5f}. These are small differences.
3. **Did top-K recall collapse?** No: weight 0.25's mean Recall@20 increases by {mean.set_index('model').loc['XGB-A-weight_0.25','Recall_at_20_abs_delta']:.5f}; however, its threshold-0.5 classification recall drops sharply. Keep those tasks separate.
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
'''
REPORT.parent.mkdir(parents=True,exist_ok=True)
REPORT.write_text(report,encoding="utf-8")
print(f"Wrote {REPORT}")
print(f"Wrote {BOOT}")
print(f"Wrote {ERRORS}")
