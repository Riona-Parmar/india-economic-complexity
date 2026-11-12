"""Build outputs/metrics.parquet: one metrics table for every model, with 95% CIs.

- Point estimates: P2's evaluate() harness (p2/data-p2/src/evaluation.py), run on
  each model's outputs/predictions_*.parquet against labels_v1.
- CIs for Precision@20 and PR-AUC: the same bootstrap as
  src/day7_bootstrap_ci_and_network.py (1,000 row resamples within each
  as-of year x subset, seed 42, 2.5/97.5 percentiles).
- xgboost rows are taken unchanged from outputs/metrics_xgboost_v1.parquet, which
  already holds the Day 7 CIs, so its published numbers don't shift.

Output columns: model, asof_year, subset, Precision@10, Precision@20, Recall@20,
PR-AUC, base_rate, lift, P20_CI_lo, P20_CI_hi, PRAUC_CI_lo, PRAUC_CI_hi
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
LABELS = ROOT / "data" / "processed" / "labels_v1.parquet"
P2_EVAL = ROOT / "p2" / "data-p2" / "src" / "evaluation.py"
XGB_METRICS = OUT / "metrics_xgboost_v1.parquet"

BASELINES = {
    "rca_persistence": OUT / "predictions_rca.parquet",
    "proximity": OUT / "predictions_density.parquet",
}
N_BOOT = 1_000
SEED = 42
COLUMNS = [
    "model", "asof_year", "subset", "Precision@10", "Precision@20", "Recall@20",
    "PR-AUC", "base_rate", "lift", "P20_CI_lo", "P20_CI_hi", "PRAUC_CI_lo", "PRAUC_CI_hi",
]


def load_p2_evaluate():
    """Load evaluate() from P2's file without running its module-level code."""
    tree = ast.parse(P2_EVAL.read_text(encoding="utf-8"))
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate")
    namespace = {"pd": pd, "np": np, "average_precision_score": average_precision_score}
    exec(compile(ast.Module(body=[func], type_ignores=[]), str(P2_EVAL), "exec"), namespace)
    return namespace["evaluate"]


def boot_metrics(df: pd.DataFrame) -> tuple[float, float]:
    """P@20 (country macro average) and pooled PR-AUC; same as the Day 7 script."""
    y = df["label"].astype(int)
    pr_auc = float(average_precision_score(y, df["score"])) if y.nunique() == 2 else np.nan
    p20 = [
        g.sort_values("score", ascending=False)["label"].astype(int).iloc[:20].mean()
        for _, g in df.groupby("country")
    ]
    return (float(np.mean(p20)) if p20 else np.nan), pr_auc


def bootstrap_cis(preds: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    test = labels[labels["split"].isin(["test_india", "test_peer"])].dropna(subset=["label"])
    data = test.merge(
        preds[["asof_year", "country", "product", "score"]],
        on=["asof_year", "country", "product"], how="left", validate="one_to_one",
    )
    if data["score"].isna().any():
        raise ValueError("Missing prediction scores in test set.")

    groups: dict[tuple, pd.DataFrame] = {}
    for (year, split), g in data.groupby(["asof_year", "split"]):
        groups[(year, "india" if split == "test_india" else "peers")] = g.reset_index(drop=True)
    for year, g in data.groupby("asof_year"):
        groups[(year, "all")] = g.reset_index(drop=True)

    rng = np.random.default_rng(SEED)
    draws = {k: ([], []) for k in groups}
    for _ in range(N_BOOT):
        for key, g in groups.items():
            p20, pr = boot_metrics(g.iloc[rng.integers(0, len(g), size=len(g))])
            draws[key][0].append(p20)
            draws[key][1].append(pr)

    rows = []
    for (year, subset), (p20s, prs) in draws.items():
        p_lo, p_hi = np.nanpercentile(p20s, [2.5, 97.5])
        a_lo, a_hi = np.nanpercentile(prs, [2.5, 97.5])
        rows.append({"asof_year": year, "subset": subset, "P20_CI_lo": p_lo,
                     "P20_CI_hi": p_hi, "PRAUC_CI_lo": a_lo, "PRAUC_CI_hi": a_hi})
    return pd.DataFrame(rows)


def main() -> None:
    evaluate = load_p2_evaluate()
    labels = pd.read_parquet(LABELS)

    frames = []
    for model, path in BASELINES.items():
        preds = pd.read_parquet(path)
        point = evaluate(preds, labels)
        print(f"{model}: bootstrapping {N_BOOT} resamples...")
        cis = bootstrap_cis(preds, labels)
        frames.append(point.merge(cis, on=["asof_year", "subset"], how="left").assign(model=model))

    frames.append(pd.read_parquet(XGB_METRICS).assign(model="xgboost"))

    metrics = pd.concat(frames, ignore_index=True)[COLUMNS]
    metrics = metrics.sort_values(["model", "asof_year", "subset"]).reset_index(drop=True)
    metrics.to_parquet(OUT / "metrics.parquet", index=False)
    print(f"Saved outputs/metrics.parquet: {len(metrics)} rows")
    print(metrics[metrics["subset"] == "india"].to_string(index=False))


if __name__ == "__main__":
    main()
