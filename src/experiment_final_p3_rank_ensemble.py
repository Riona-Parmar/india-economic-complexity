"""Evaluate a fixed rank ensemble on the existing train-only validation scores."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs/p3_precision_optimization_predictions.parquet"
OUT = ROOT / "outputs"
PREDICTIONS = OUT / "p3_final_model_validation_predictions.parquet"
FOLD_METRICS = OUT / "p3_final_model_validation_fold_metrics.csv"
SUMMARY = OUT / "p3_final_model_validation_summary.csv"
OVERLAP = OUT / "p3_final_model_validation_overlap.csv"
BOOTSTRAP = OUT / "p3_final_model_validation_bootstrap.csv"
MODELS = ["XGB-A-baseline", "XGB-F-existing", "XGB-A-weight_0.25"]
FOLDS = [2005, 2010, 2015]
KS = [5, 10, 20, 50, 100]
SEED = 42
BOOTSTRAP_REPS = 2000
KEYS = ["asof_year", "target_year", "country", "product"]


def country_recall_at_k(frame, k, positive_col="label"):
    recalls = []
    for _, group in frame.groupby("country", sort=True):
        positives = group[positive_col].astype(bool).to_numpy()
        if positives.sum():
            recalls.append(float(positives[:k].sum() / positives.sum()))
    return float(np.mean(recalls)) if recalls else np.nan


def subgroup_recall_at_k(frame, k, subgroup):
    recalls = []
    for _, group in frame.groupby("country", sort=True):
        mask = group[subgroup].fillna(False).astype(bool).to_numpy()
        if mask.sum():
            recalls.append(float(mask[:k].sum() / mask.sum()))
    return float(np.mean(recalls)) if recalls else np.nan


def evaluate_fold(frame):
    y = frame.label.astype(int).to_numpy()
    score = frame.score.astype(float).to_numpy()
    result = {
        "PR_AUC_global": average_precision_score(y, score),
        "ROC_AUC_global": roc_auc_score(y, score),
        "base_rate_pooled": float(y.mean()),
        "rows": len(frame),
        "positives": int(y.sum()),
        "country_count": int(frame.country.nunique()),
        "score_std": float(np.std(score)),
        "score_unique": int(np.unique(score).size),
    }
    for k in KS:
        selected = frame.groupby("country", sort=True, group_keys=False).head(k)
        hit_count = int(selected.label.sum())
        selected_count = len(selected)
        p_macro = float(selected.groupby("country").label.mean().mean())
        p_pooled = hit_count / selected_count
        result[f"Precision@{k}_country_macro"] = p_macro
        result[f"Precision@{k}_pooled_selected"] = p_pooled
        result[f"Recall@{k}_country_macro"] = country_recall_at_k(frame, k)
        result[f"Recall@{k}_pooled"] = hit_count / int(y.sum()) if y.sum() else np.nan
        result[f"Lift@{k}_macro"] = p_macro / y.mean() if y.mean() else np.nan
        result[f"Lift@{k}_pooled_selected"] = p_pooled / y.mean() if y.mean() else np.nan
        result[f"Weak_positive_recall@{k}_country_macro"] = subgroup_recall_at_k(frame, k, "weak_positive")
        result[f"High_PCI_recall@{k}_country_macro"] = subgroup_recall_at_k(frame, k, "high_pci_positive")
        result[f"Low_density_recall@{k}_country_macro"] = subgroup_recall_at_k(frame, k, "low_density_positive")
    return result


def select_positive_hits(frame, k):
    selected = frame.loc[frame.country_rank.le(k)]
    return set(map(tuple, selected.loc[selected.label.eq(1), ["country", "product"]].to_numpy()))


def paired_bootstrap(frame, model, baseline, year, k, metric):
    model_rows = frame.loc[frame.model.eq(model) & frame.asof_year.eq(year)]
    base_rows = frame.loc[frame.model.eq(baseline) & frame.asof_year.eq(year)]
    model_by_country = {}
    base_by_country = {}
    for name, source, target in [("m", model_rows, model_by_country), ("b", base_rows, base_by_country)]:
        for country, group in source.groupby("country", sort=True):
            ranked = group.sort_values("score", ascending=False, kind="mergesort")
            actual = ranked.label.astype(int).to_numpy()
            hit = int(actual[:k].sum())
            if metric == "precision":
                target[country] = hit / min(k, len(actual))
            elif actual.sum():
                target[country] = hit / int(actual.sum())
    countries = sorted(set(model_by_country) & set(base_by_country))
    deltas = np.asarray([model_by_country[c] - base_by_country[c] for c in countries], dtype=float)
    rng = np.random.default_rng(SEED + int(year) + k + (0 if metric == "precision" else 1000))
    samples = rng.integers(0, len(deltas), size=(BOOTSTRAP_REPS, len(deltas)))
    means = deltas[samples].mean(axis=1)
    return {
        "model": model, "baseline": baseline, "validation_year": int(year), "K": k,
        "metric": metric, "country_count": len(countries), "delta": float(deltas.mean()),
        "ci95_low": float(np.percentile(means, 2.5)), "ci95_high": float(np.percentile(means, 97.5)),
    }


def main():
    for path in [PREDICTIONS, FOLD_METRICS, SUMMARY, OVERLAP, BOOTSTRAP]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing P3 output: {path}")
    source = pd.read_parquet(INPUT)
    source = source.loc[source.model.isin(MODELS)].copy()
    required = set(KEYS + ["model", "score", "label", "country_rank", "weak_positive",
                           "high_pci_positive", "low_density_positive"])
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Saved predictions lack columns: {missing}")
    if set(source.asof_year.unique()) != set(FOLDS):
        raise ValueError("Saved predictions do not contain the three required folds")
    if source.country.isin(["IND", "BGD", "IDN", "MEX", "TUR", "VNM"]).any():
        raise ValueError("Holdout-country rows found in saved train-split predictions")
    if source.duplicated(["model"] + KEYS).any():
        raise ValueError("Duplicate model-country-product rows")
    source["country_size"] = source.groupby(["model", "asof_year", "country"])["product"].transform("size")
    source["rank_percentile"] = np.where(
        source.country_size.gt(1),
        1 - (source.country_rank - 1) / (source.country_size - 1),
        1.0,
    )
    rank_wide = source.pivot(index=KEYS, columns="model", values="rank_percentile").reset_index()
    labels = source.groupby(KEYS, as_index=False).agg(
        label=("label", "first"), weak_positive=("weak_positive", "first"),
        high_pci_positive=("high_pci_positive", "first"),
        low_density_positive=("low_density_positive", "first"),
    )
    if len(labels) != len(rank_wide):
        raise ValueError("Candidate predictions do not align on validation keys")
    wide = rank_wide.merge(labels, on=KEYS, validate="one_to_one")
    for model in MODELS:
        if wide[model].isna().any():
            raise ValueError(f"Missing rank percentiles for {model}")
    # Fixed equal-weight rank average: it uses predictions only, never labels.
    wide["score"] = wide[MODELS].mean(axis=1)
    ensemble = wide[KEYS + ["score", "label", "weak_positive", "high_pci_positive", "low_density_positive"]].copy()
    ensemble["model"] = "RankEnsemble_equal_A_F_w025"
    ensemble = ensemble.sort_values(
        ["asof_year", "country", "score", "product"],
        ascending=[True, True, False, True], kind="mergesort",
    )
    ensemble["country_rank"] = ensemble.groupby(["asof_year", "country"]).cumcount() + 1
    ensemble["top10_selected"] = ensemble.country_rank.le(10)
    ensemble["top20_selected"] = ensemble.country_rank.le(20)
    ensemble["top50_selected"] = ensemble.country_rank.le(50)
    ensemble["top100_selected"] = ensemble.country_rank.le(100)
    candidates = source[KEYS + ["model", "score", "label", "country_rank", "top10_selected",
        "top20_selected", "weak_positive", "high_pci_positive", "low_density_positive"]].copy()
    all_predictions = pd.concat([candidates, ensemble], ignore_index=True)

    fold_rows = []
    for model, group in all_predictions.groupby("model", sort=False):
        for year, fold in group.groupby("asof_year", sort=True):
            ranked = fold.sort_values(["country", "score"], ascending=[True, False], kind="mergesort")
            row = {"model": model, "validation_year": int(year), **evaluate_fold(ranked)}
            fold_rows.append(row)
    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(FOLD_METRICS, index=False)
    summary_rows = []
    for model, group in fold_metrics.groupby("model", sort=False):
        for metric in [c for c in fold_metrics.columns if c not in {"model", "validation_year"}]:
            summary_rows.append({"model": model, "metric": metric, "mean_over_folds": group[metric].mean(),
                                 "fold_sd": group[metric].std(ddof=1)})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY, index=False)

    overlap_rows = []
    bootstrap_rows = []
    for year in FOLDS:
        fold = all_predictions.loc[all_predictions.asof_year.eq(year)]
        model_frames = {name: fold.loc[fold.model.eq(name)] for name in fold.model.unique()}
        for k in KS:
            hits = {name: select_positive_hits(frame, k) for name, frame in model_frames.items()}
            for other in ["XGB-A-baseline", "XGB-F-existing", "XGB-A-weight_0.25"]:
                candidate = "RankEnsemble_equal_A_F_w025"
                overlap = hits[other] & hits[candidate]
                union = hits[other] | hits[candidate]
                overlap_rows.append({"validation_year": year, "K": k, "reference_model": other,
                    "candidate_model": candidate, "reference_positive_hits": len(hits[other]),
                    "candidate_positive_hits": len(hits[candidate]), "shared_hits": len(overlap),
                    "candidate_only_hits": len(hits[candidate] - hits[other]),
                    "reference_only_hits": len(hits[other] - hits[candidate]),
                    "positive_hit_jaccard": len(overlap) / len(union) if union else np.nan})
            for candidate in ["RankEnsemble_equal_A_F_w025", "XGB-F-existing", "XGB-A-weight_0.25"]:
                for metric in ["precision", "recall"]:
                    bootstrap_rows.append(paired_bootstrap(all_predictions, candidate, "XGB-A-baseline", year, k, metric))
    pd.DataFrame(overlap_rows).to_csv(OVERLAP, index=False)
    pd.DataFrame(bootstrap_rows).to_csv(BOOTSTRAP, index=False)
    all_predictions.to_parquet(PREDICTIONS, index=False)
    print(f"Saved {FOLD_METRICS}")
    print(f"Saved {SUMMARY}")
    print(f"Saved {OVERLAP}")
    print(f"Saved {BOOTSTRAP}")
    print(f"Saved {PREDICTIONS}")


if __name__ == "__main__":
    main()
