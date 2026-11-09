"""India top-20 backtest from model_features_v1.parquet (reference XGBoost reproduction, not P3's exact model).

For each evaluation window A (default 2010 and 2015):
  - trains XGBoost on TRAIN-country rows of windows whose outcome was known by A (as-of + 5 <= A),
  - scores India's eligible products at as-of A,
  - writes India's top-N list with names, sectors, score, and whether India really reached RCA >= 1 five years later,
  - compares hits with two baselines (RCA persistence, proximity density) and rolls the scores up to sectors.

Run from the project root:
    py -3.13 india_top20.py
    py -3.13 india_top20.py --pathway recovery          (the 'win back lost products' track)
Optional: --density "p2/data-p2/density/density_A*.parquet"  --windows 2010,2015  --topn 20
"""
import argparse, glob, os, re, sys
import numpy as np, pandas as pd
import xgboost as xgb

ap = argparse.ArgumentParser()
ap.add_argument("--features", default="data/processed/model_features_v1.parquet")
ap.add_argument("--density", default="p2/data-p2/density/density_A*.parquet")
ap.add_argument("--pathway", default=None)
ap.add_argument("--windows", default="2010,2015")
ap.add_argument("--topn", type=int, default=20)
ap.add_argument("--weight", type=float, default=0.25, help="multiplier on neg/pos for scale_pos_weight")
ap.add_argument("--out", default="results")
a = ap.parse_args()

d = pd.read_parquet(a.features)
d["product"] = d["product"].astype(str).str.zfill(4)
if "pathway" in d.columns:
    vals = d["pathway"].astype(str).unique().tolist()
    print("pathway values in the file:", vals)
    pw = a.pathway or next((v for v in vals if re.search("new|emerg", v, re.I)), None)
    if pw is None:
        sys.exit(f"Could not tell which pathway is new emergence. Re-run with --pathway <one of {vals}>")
    d = d[d["pathway"].astype(str) == pw]
    print(f"using pathway = {pw}: {len(d):,} rows")
else:
    pw = "all"

# optional density feature from P2 (as-of year taken from the file name when missing)
keys = ["asof_year", "country", "product"]
if a.density and glob.glob(a.density):
    parts = []
    for f in sorted(glob.glob(a.density)):
        x = pd.read_parquet(f)
        if "asof_year" not in x.columns:
            m = re.search(r"A(\d{4})", os.path.basename(f))
            x["asof_year"] = int(m.group(1)) if m else np.nan
        parts.append(x)
    x = pd.concat(parts, ignore_index=True)
    col = next((c for c in ["density", "score"] if c in x.columns), None)
    if col:
        x["product"] = x["product"].astype(str).str.zfill(4)
        x = x[keys + [col]].drop_duplicates(keys).rename(columns={col: "f_density"})
        d = d.merge(x, on=keys, how="left")
        print(f"density joined; missing density on {d.f_density.isna().mean():.1%} of rows")
else:
    print("[note] no density files found: model runs without the density feature")

CAND = ["rca3_current", "rca_current", "pci_current", "eci_current", "log_export_value", "rca3_lag1", "rca3_lag3",
        "rca3_change_3y", "log_export_change_1y", "log_export_change_3y", "lag1_available", "lag3_available", "f_density"]
feats = [c for c in CAND if c in d.columns]
print("features used:", feats, "\n(atlas_distance / atlas_cog are deliberately excluded)")
d["sector"] = d["sector"].fillna("Unknown") if "sector" in d.columns else "Unknown"
os.makedirs(a.out, exist_ok=True)

summ, secs, tops = [], [], []
for A in [int(w) for w in a.windows.split(",")]:
    tr = d[(d.split == "train") & (d.asof_year + 5 <= A) & d.label.notna()]
    te = d[(d.split == "test_india") & (d.asof_year == A)].copy()
    if tr.empty or te.empty:
        print(f"[skip] window {A}: train rows={len(tr)}, India rows={len(te)}")
        continue
    pos, neg = tr.label.sum(), len(tr) - tr.label.sum()
    model = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                              min_child_weight=5, scale_pos_weight=a.weight * neg / max(pos, 1), tree_method="hist",
                              random_state=42, n_jobs=-1, eval_metric="aucpr")
    model.fit(tr[feats], tr.label.astype(int))
    te["score"] = model.predict_proba(te[feats])[:, 1]
    te["label"] = te.label.astype(int)
    print(f"\nA{A}: trained on {len(tr):,} rows from windows {sorted(int(x) for x in tr.asof_year.unique())} "
          f"({int(pos):,} positives); India eligible={len(te)}, India positives={int(te.label.sum())}")

    ranked = te.sort_values(["score", "product"], ascending=[False, True]).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    top = ranked.head(a.topn)
    cols = [c for c in ["rank", "product", "name", "sector", "score", "rca3_current", "pci_current", "f_density", "label"]
            if c in top.columns]
    out = top[cols].rename(columns={"label": "reached_RCA>=1_after_5y"})
    out.insert(0, "asof_year", A)
    tops.append(out)
    out.to_csv(os.path.join(a.out, f"india_top{a.topn}_A{A}_{pw}.csv"), index=False)
    print(out.drop(columns=["asof_year"]).round(4).to_string(index=False))

    # baselines on the same rows
    n, npos = len(te), int(te.label.sum())
    methods = {"xgb_reproduction": te.score, "rca_persistence": te.rca3_current}
    if "f_density" in te.columns and te.f_density.notna().any():
        methods["proximity_density"] = te.f_density.fillna(-1)
    for name, s in methods.items():
        r = te.assign(s=s).sort_values(["s", "product"], ascending=[False, True])
        row = dict(asof_year=A, method=name, eligible=n, positives=npos)
        for k in (20, 50, 100):
            row[f"hits@{k}"] = int(r.label.head(k).sum())
        summ.append(row)
    summ.append(dict(asof_year=A, method="random (expected)", eligible=n, positives=npos,
                     **{f"hits@{k}": round(k * npos / n, 2) for k in (20, 50, 100)}))

    # sector roll-up
    g = te.groupby("sector").agg(eligible=("label", "size"), positives=("label", "sum"), score_sum=("score", "sum"),
                                 top_score=("score", "max")).sort_values("score_sum", ascending=False).reset_index()
    g["asof_year"] = A
    secs.append(g)

if not summ:
    sys.exit("Nothing was scored. Check the split names and windows in the features file.")
S = pd.DataFrame(summ)
S.to_csv(os.path.join(a.out, f"india_backtest_hits_{pw}.csv"), index=False)
print("\n=== India hits in the top-K lists (K=20/50/100), same rows for every method ===")
print(S.to_string(index=False))
G = pd.concat(secs, ignore_index=True)
G.to_csv(os.path.join(a.out, f"india_sector_rollup_{pw}.csv"), index=False)
print("\n=== Sector roll-up (sectors ranked by summed score; top 8 per window) ===")
print(G.groupby("asof_year").head(8).round(4).to_string(index=False))
print(f"\nSaved in {a.out}\\ . Note: India has very few positives in the new-emergence track, so hit counts are tiny and anecdotal.")
