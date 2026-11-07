"""
day7_bootstrap_ci_and_network.py  —  Day 7: Confidence Intervals + Network Update
====================================================================================
Pipeline:
  0. Build p3_dataset_v1 and v2 if they don't exist (runs prepare + engineer steps).
  1. Train / load the final XGBoost predictions (test_india + test_peer, 2010 & 2015).
  2. Bootstrap resample with replacement 1,000 times on the test set.
  3. For each iteration recalculate Precision@20 and PR-AUC using P2's exact harness.
  4. Find 2.5% and 97.5% quantiles → 95% CI.
  5. Write CIs back into metrics.parquet (and also update metrics_xgboost_v1.csv).
  6. Print the final wide-interval numbers.
  7. Overlay XGBoost top-20 India predictions on the Day-6 network plot → save a new HTML.
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

# ------------------------------------------------------------------ paths ---
ROOT      = Path(__file__).resolve().parents[1]
DATA      = ROOT / "data" / "processed"
P2_SRC    = ROOT / "p2" / "data-p2" / "src"
P2_DENS   = ROOT / "p2" / "data-p2" / "density"
P2_MAT    = ROOT / "p2" / "data-p2" / "matrices"
OUT       = ROOT / "outputs"

LABELS_PATH   = DATA / "labels_v1.parquet"
FEATURES_PATH = DATA / "model_features_v1.parquet"
TRADE_PATH    = DATA / "trade_rca_v1.parquet"
DS_V1_PATH    = DATA / "p3_dataset_v1.parquet"
DS_V2_PATH    = DATA / "p3_dataset_v2.parquet"
PRED_PATH     = OUT  / "predictions_xgboost_v1.parquet"
METRICS_CSV   = OUT  / "metrics_xgboost_v1.csv"
METRICS_PAR   = OUT  / "metrics_xgboost_v1.parquet"
NODES_PATH    = OUT  / "network" / "network_A2010_nodes.parquet"
EDGES_PATH    = OUT  / "network" / "network_A2010_edges.parquet"
HTML_OUT      = OUT  / "network" / "product_space_2010_p3.html"

OUT.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "density", "pci_current", "rca3_current",
    "rca3_change_3y", "eci_current", "f_ubiquity", "f_world_growth",
]
YEARS    = [2000, 2005, 2010, 2015]
N_BOOT   = 1_000
SEED     = 42
rng      = np.random.default_rng(SEED)

from sklearn.metrics import average_precision_score as _ap_score

# --- P2's exact evaluate() harness, inlined to avoid module-level side-effects ---
def evaluate(predictions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """
    Exact replica of p2/data-p2/src/evaluation.py::evaluate().
    Scores: Precision@10, Precision@20, Recall@20, PR-AUC, base_rate, lift.
    Groups: per asof_year × {india, peers, all}.
    """
    keys = ["asof_year", "country", "product"]
    test_labels = labels[labels["split"].isin(["test_india", "test_peer"])].copy()
    test_labels = test_labels.dropna(subset=["label"])
    data = test_labels.merge(predictions[keys + ["score"]], on=keys,
                             how="left", validate="one_to_one")
    if data["score"].isna().any():
        raise ValueError("Missing prediction scores for some test rows")

    results = []
    for year in sorted(data["asof_year"].unique()):
        yd = data[data["asof_year"] == year]
        subsets = {
            "india": yd[yd["split"] == "test_india"],
            "peers": yd[yd["split"] == "test_peer"],
            "all":   yd,
        }
        for sname, sub in subsets.items():
            if len(sub) == 0:
                continue
            y_true  = sub["label"].astype(int)
            y_score = sub["score"].astype(float)
            base_rate = y_true.mean()
            pr_auc = float(_ap_score(y_true, y_score)) if y_true.nunique() == 2 else float("nan")
            cp10, cp20, cr20 = [], [], []
            for _, grp in sub.groupby("country"):
                grp = grp.sort_values("score", ascending=False)
                act = grp["label"].astype(int).to_numpy()
                cp10.append(act[:10].sum() / len(act[:10]))
                cp20.append(act[:20].sum() / len(act[:20]))
                if act.sum() > 0:
                    cr20.append(act[:20].sum() / act.sum())
            p10 = float(np.mean(cp10))
            p20 = float(np.mean(cp20))
            r20 = float(np.mean(cr20)) if cr20 else float("nan")
            lift = p20 / base_rate if base_rate > 0 else float("nan")
            results.append({
                "asof_year": year, "subset": sname,
                "Precision@10": p10, "Precision@20": p20,
                "Recall@20": r20, "PR-AUC": pr_auc,
                "base_rate": base_rate, "lift": lift,
            })
    return pd.DataFrame(results)

# =====================================================================
# STEP 0 — Build p3_dataset_v1 and v2 if missing
# =====================================================================
print("=" * 70)
print("STEP 0 — Dataset preparation")
print("=" * 70)

if not DS_V1_PATH.exists():
    print("Building p3_dataset_v1 ...")
    features = pd.read_parquet(FEATURES_PATH)
    features = features[features["pathway"] == "new_emergence"].copy()

    density_frames = []
    for y in YEARS:
        d = pd.read_parquet(P2_DENS / f"density_A{y}.parquet")
        density_frames.append(d)
    density_all = pd.concat(density_frames, ignore_index=True)

    ds1 = features.merge(density_all, on=["asof_year", "country", "product"],
                         how="left", validate="one_to_one")
    if ds1["density"].isna().any():
        raise ValueError("Missing density values in v1 merge.")
    ds1.to_parquet(DS_V1_PATH, index=False)
    print(f"  Saved p3_dataset_v1: {len(ds1):,} rows")
else:
    print(f"  p3_dataset_v1 already exists ({DS_V1_PATH.stat().st_size//1024} KB)")

if not DS_V2_PATH.exists():
    print("Building p3_dataset_v2 (ubiquity + world growth) ...")
    ds = pd.read_parquet(DS_V1_PATH)

    # Ubiquity
    ub_frames = []
    for y in YEARS:
        mat = pd.read_parquet(P2_MAT / f"matrix_A{y}.parquet")
        counts = mat.sum(axis=0)
        ub_frames.append(pd.DataFrame(
            {"product": counts.index, "asof_year": y, "f_ubiquity": counts.values}
        ))
    ubiquity = pd.concat(ub_frames, ignore_index=True)

    # World export growth (3-year log change)
    trade = pd.read_parquet(TRADE_PATH, columns=["year", "product", "export_value"])
    world_exp = trade.groupby(["year", "product"])["export_value"].sum()

    gr_frames = []
    for y in YEARS:
        prods = ds.loc[ds["asof_year"] == y, "product"].unique()
        cur = world_exp.loc[y].reindex(prods, fill_value=0)
        prv = world_exp.loc[y - 3].reindex(prods, fill_value=0)
        growth = np.log1p(cur.values) - np.log1p(prv.values)
        gr_frames.append(pd.DataFrame(
            {"product": prods, "asof_year": y, "f_world_growth": growth}
        ))
    growth_all = pd.concat(gr_frames, ignore_index=True)

    ds = ds.merge(ubiquity, on=["asof_year", "product"], how="left", validate="many_to_one")
    ds = ds.merge(growth_all, on=["asof_year", "product"], how="left", validate="many_to_one")

    if ds[["f_ubiquity", "f_world_growth"]].isna().any().any():
        raise ValueError("Missing engineered feature values.")
    ds.to_parquet(DS_V2_PATH, index=False)
    print(f"  Saved p3_dataset_v2: {len(ds):,} rows")
else:
    print(f"  p3_dataset_v2 already exists ({DS_V2_PATH.stat().st_size//1024} KB)")

# =====================================================================
# STEP 1 — Train / load XGBoost predictions
# =====================================================================
print("\n" + "=" * 70)
print("STEP 1 — XGBoost training / prediction")
print("=" * 70)

if PRED_PATH.exists():
    print(f"  Loading existing predictions: {PRED_PATH.name}")
    predictions_df = pd.read_parquet(PRED_PATH)
else:
    print("  Training XGBoost (baseline, scale_pos_weight, seed=42) ...")
    df = pd.read_parquet(DS_V2_PATH)
    all_preds = []
    for test_year in [2010, 2015]:
        train_years = [y for y in YEARS if y < test_year]
        train_data  = df[(df["split"] == "train") & (df["asof_year"].isin(train_years))].copy()
        test_data   = df[(df["split"].isin(["test_india", "test_peer"])) & (df["asof_year"] == test_year)].copy()

        X_tr, y_tr = train_data[FEATURES], train_data["label"].astype(int)
        scale = (y_tr == 0).sum() / (y_tr == 1).sum()

        model = XGBClassifier(
            objective="binary:logistic", eval_metric="aucpr",
            scale_pos_weight=scale, random_state=42, n_jobs=-1,
        )
        model.fit(X_tr, y_tr)

        preds = test_data[["asof_year", "country", "product"]].copy()
        preds["score"] = model.predict_proba(test_data[FEATURES])[:, 1]
        all_preds.append(preds)
        print(f"    {test_year}: {len(preds):,} test rows")

    predictions_df = pd.concat(all_preds, ignore_index=True)
    predictions_df.to_parquet(PRED_PATH, index=False)
    print(f"  Saved: {PRED_PATH}")

print(f"  Predictions: {len(predictions_df):,} rows, years: {sorted(predictions_df['asof_year'].unique())}")

# =====================================================================
# STEP 2 — Prepare merged test set for bootstrap
# =====================================================================
print("\n" + "=" * 70)
print("STEP 2 — Merge predictions with test labels")
print("=" * 70)

labels = pd.read_parquet(LABELS_PATH)
labels = labels[labels["asof_year"].isin([2010, 2015])].copy()

# P2 evaluate() uses test_india + test_peer; we do the same
test_labels = labels[labels["split"].isin(["test_india", "test_peer"])].copy()
test_labels = test_labels.dropna(subset=["label"])

data = test_labels.merge(
    predictions_df[["asof_year", "country", "product", "score"]],
    on=["asof_year", "country", "product"],
    how="left", validate="one_to_one",
)
if data["score"].isna().any():
    raise ValueError("Missing prediction scores in test set.")

print(f"  Test rows: {len(data):,}  |  Positive rate: {data['label'].mean():.4%}")
print(f"  Positives: {data['label'].sum()}")

# Point-estimate metrics (using P2 harness)
pt_metrics = evaluate(predictions_df, labels)
print("\n  Point-estimate metrics (P2 harness):")
print(pt_metrics.to_string(index=False))

# =====================================================================
# STEP 3 — Bootstrap Precision@20 and PR-AUC (1,000 iterations)
#           Resampling strategy: resample ROWS within each (year, subset)
#           group with replacement, maintaining the same group structure
#           as P2's evaluate() — which pools per subset.
# =====================================================================
print("\n" + "=" * 70)
print(f"STEP 3 — Bootstrap CI ({N_BOOT:,} resamples, seed={SEED})")
print("=" * 70)

def _boot_metrics(subset_df: pd.DataFrame) -> tuple[float, float]:
    """
    Compute Precision@20 (country macro) and PR-AUC on a data slice
    matching P2's exact logic from evaluation.py.
    """
    y_true  = subset_df["label"].astype(int)
    y_score = subset_df["score"].astype(float)

    # PR-AUC (global pooled)
    if y_true.nunique() == 2:
        pr_auc = float(average_precision_score(y_true, y_score))
    else:
        pr_auc = float("nan")

    # Precision@20 (country macro)
    country_p20 = []
    for _, grp in subset_df.groupby("country"):
        grp_sorted = grp.sort_values("score", ascending=False)
        top20 = grp_sorted["label"].astype(int).iloc[:20]
        country_p20.append(top20.sum() / len(top20))

    p20 = float(np.mean(country_p20)) if country_p20 else float("nan")
    return p20, pr_auc


# Collect bootstrap samples for each (asof_year, subset) combination
groups = {}
for (year, split_name), grp in data.groupby(["asof_year", "split"]):
    subset_key = "india" if split_name == "test_india" else "peers"
    groups[(year, subset_key)] = grp.reset_index(drop=True)
    # Merge india+peers into 'all'
    all_key = (year, "all")
    if all_key not in groups:
        groups[all_key] = grp.reset_index(drop=True)
    else:
        groups[all_key] = pd.concat([groups[all_key], grp], ignore_index=True)

boot_records: dict[tuple, dict[str, list]] = {}
for key in groups:
    boot_records[key] = {"p20": [], "pr_auc": []}

t0 = time.time()
for b in range(N_BOOT):
    if (b + 1) % 200 == 0:
        elapsed = time.time() - t0
        print(f"  Iteration {b+1}/{N_BOOT} ... {elapsed:.1f}s elapsed")

    for key, grp in groups.items():
        # Resample rows with replacement
        idx    = rng.integers(0, len(grp), size=len(grp))
        sample = grp.iloc[idx].copy()
        p20, pr_auc = _boot_metrics(sample)
        boot_records[key]["p20"].append(p20)
        boot_records[key]["pr_auc"].append(pr_auc)

print(f"  Bootstrap complete in {time.time() - t0:.1f}s")

# =====================================================================
# STEP 4 — Compute 95% CI (2.5% and 97.5% quantiles)
# =====================================================================
print("\n" + "=" * 70)
print("STEP 4 — 95% Confidence Intervals")
print("=" * 70)

ci_rows = []
for key, arrays in boot_records.items():
    year, subset = key
    p20_arr    = np.array(arrays["p20"])
    pr_arr     = np.array(arrays["pr_auc"])

    # Filter NaN (can occur when resample has only one class)
    p20_valid  = p20_arr[~np.isnan(p20_arr)]
    pr_valid   = pr_arr[~np.isnan(pr_arr)]

    p20_lo, p20_hi = np.nanpercentile(p20_arr, [2.5, 97.5])
    pr_lo,  pr_hi  = np.nanpercentile(pr_arr,  [2.5, 97.5])

    # Look up point estimate from P2 harness output
    pe_row  = pt_metrics[(pt_metrics["asof_year"] == year) & (pt_metrics["subset"] == subset)]
    p20_pt  = float(pe_row["Precision@20"].values[0]) if len(pe_row) else float("nan")
    pr_pt   = float(pe_row["PR-AUC"].values[0])       if len(pe_row) else float("nan")

    ci_rows.append({
        "asof_year":    year,
        "subset":       subset,
        "Precision@20": p20_pt,
        "P20_CI_lo":    p20_lo,
        "P20_CI_hi":    p20_hi,
        "PR_AUC":       pr_pt,
        "PRAUC_CI_lo":  pr_lo,
        "PRAUC_CI_hi":  pr_hi,
        "n_boot_p20_valid":  len(p20_valid),
        "n_boot_pr_valid":   len(pr_valid),
    })
    print(
        f"  {year} / {subset:6s}  "
        f"P@20 = {p20_pt:.4f}  95%CI [{p20_lo:.4f}, {p20_hi:.4f}]  |  "
        f"PR-AUC = {pr_pt:.4f}  95%CI [{pr_lo:.4f}, {pr_hi:.4f}]"
    )

ci_df = pd.DataFrame(ci_rows).sort_values(["asof_year", "subset"]).reset_index(drop=True)

# =====================================================================
# STEP 5 — Write CIs back into metrics.parquet (and CSV)
# =====================================================================
print("\n" + "=" * 70)
print("STEP 5 — Save metrics.parquet and metrics_xgboost_v1.csv")
print("=" * 70)

# Build a combined DataFrame: point-estimate metrics + CI columns
metrics_full = pt_metrics.copy()
metrics_full = metrics_full.merge(
    ci_df[["asof_year", "subset", "P20_CI_lo", "P20_CI_hi", "PRAUC_CI_lo", "PRAUC_CI_hi"]],
    on=["asof_year", "subset"],
    how="left",
)

metrics_full.to_parquet(METRICS_PAR, index=False)
metrics_full.to_csv(METRICS_CSV, index=False)
print(f"  Saved: {METRICS_PAR}")
print(f"  Saved: {METRICS_CSV}")

# =====================================================================
# STEP 6 — Print final wide-interval numbers for team reporting
# =====================================================================
print("\n" + "=" * 70)
print("STEP 6 — FINAL RESULTS FOR REPORTING")
print("=" * 70)

for _, row in ci_df.sort_values(["asof_year","subset"]).iterrows():
    print(
        f"\n  Year {int(row['asof_year'])} | Group: {row['subset']}"
        f"\n    Precision@20 : {row['Precision@20']:.4f}  "
        f"95% CI  [{row['P20_CI_lo']:.4f}, {row['P20_CI_hi']:.4f}]  "
        f"(width = {row['P20_CI_hi'] - row['P20_CI_lo']:.4f})"
        f"\n    PR-AUC       : {row['PR_AUC']:.4f}  "
        f"95% CI  [{row['PRAUC_CI_lo']:.4f}, {row['PRAUC_CI_hi']:.4f}]  "
        f"(width = {row['PRAUC_CI_hi'] - row['PRAUC_CI_lo']:.4f})"
    )

# =====================================================================
# STEP 7 — Overlay XGBoost Top-20 India predictions on Day-6 network
# =====================================================================
print("\n" + "=" * 70)
print("STEP 7 — Updating product_space_2010_p3.html")
print("=" * 70)

import plotly.graph_objects as go

nodes = pd.read_parquet(NODES_PATH)
edges = pd.read_parquet(EDGES_PATH)

# India present (m=1) in 2010
trade_m = pd.read_parquet(TRADE_PATH, columns=["country", "product", "year", "m"])
india_2010 = trade_m[(trade_m["country"] == "IND") & (trade_m["year"] == 2010)]
india_present = set(india_2010.loc[india_2010["m"] == 1, "product"])
del trade_m

# XGBoost top-20 India predictions for 2010 (eligible = m==0 products that the model scores)
# Eligible set: products India doesn't yet have competitively at 2010
ind_preds_2010 = predictions_df[
    (predictions_df["asof_year"] == 2010) &
    (predictions_df["country"] == "IND")
].copy()

# Top-20 by score (these are the model's recommended emerging products)
top20 = set(ind_preds_2010.sort_values("score", ascending=False).head(20)["product"])
print(f"  India top-20 predictions for 2010: {sorted(top20)}")

# Attach scores to nodes
score_map = dict(zip(ind_preds_2010["product"], ind_preds_2010["score"]))
nodes["prob"] = nodes["product"].map(score_map)

nodes["india_present"] = nodes["product"].isin(india_present)
nodes["top20"]         = nodes["product"].isin(top20)

# Mark node size / colour
pci_min, pci_max = nodes["pci"].min(), nodes["pci"].max()
nodes["size"] = 6 + 16 * (nodes["pci"] - pci_min) / (pci_max - pci_min)
nodes["size"] = nodes["size"].fillna(8)

SECTOR_COLORS = {
    "Agriculture, live animals and plant products":                    "#2ca02c",
    "Arms and ammunition":                                            "#7f7f7f",
    "Basic metals and fabricated metal products":                      "#8c564b",
    "Chemicals and petrochemicals":                                    "#9467bd",
    "Electrical machinery, electronics and electrical equipment":      "#1f77b4",
    "Food and animal products, beverages, fats, oils and tobacco":     "#ff7f0e",
    "Footwear, headgear and related products":                         "#e377c2",
    "Furniture, toys, miscellaneous manufactures and special goods":   "#bcbd22",
    "Industrial machinery and mechanical equipment":                   "#17becf",
    "Leather, hides, furs and related products":                       "#d62728",
    "Medical devices, optical and precision instruments":              "#aec7e8",
    "Mining and non-fuel mineral products":                            "#c49c94",
    "Petroleum, coal and refined fuels":                               "#393b79",
    "Pharmaceuticals and medicinal products":                          "#e7969c",
    "Plastics and rubber products":                                    "#de9ed6",
    "Precious metals, gems and jewellery":                             "#ffd92f",
    "Stone, ceramics, glass, cement and related products":             "#c7c7c7",
    "Textiles, apparel and textile articles":                          "#dbdb8d",
    "Transport equipment":                                             "#9edae5",
    "Wood, pulp, paper and printing products":                         "#98df8a",
}
DEFAULT_COLOR = "#cccccc"
nodes["color"] = nodes["sector"].map(SECTOR_COLORS).fillna(DEFAULT_COLOR)

# Build hover text
hover_parts = []
for _, r in nodes.iterrows():
    t = (
        f"<b>{r['name'] or r['product']}</b><br>"
        f"Product: {r['product']}<br>"
        f"Sector: {r['sector'] or 'Unknown'}<br>"
        f"PCI: {r['pci']:.3f}"
    )
    if pd.notna(r["prob"]):
        t += f"<br>P(emergence): {r['prob']:.4f}"
    if r["india_present"]:
        t += "<br><b>🇮🇳 India exports this (2010)</b>"
    if r["top20"]:
        t += "<br>⭐ <b>XGBoost Top-20 prediction</b>"
    hover_parts.append(t)
nodes["hover"] = hover_parts

fig = go.Figure()

# Edges
node_pos = dict(zip(nodes["product"], zip(nodes["x"], nodes["y"])))
edge_x, edge_y = [], []
for _, e in edges.iterrows():
    if e["product_i"] in node_pos and e["product_j"] in node_pos:
        x0, y0 = node_pos[e["product_i"]]
        x1, y1 = node_pos[e["product_j"]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

fig.add_trace(go.Scatter(
    x=edge_x, y=edge_y, mode="lines",
    line=dict(width=0.3, color="rgba(180,180,180,0.35)"),
    hoverinfo="skip", showlegend=False,
))

# Non-India, non-top20 nodes (per sector)
base = nodes[~nodes["india_present"] & ~nodes["top20"]]
for sector in sorted(nodes["sector"].dropna().unique()):
    sub = base[base["sector"] == sector]
    if len(sub) == 0:
        continue
    fig.add_trace(go.Scatter(
        x=sub["x"], y=sub["y"], mode="markers",
        marker=dict(size=sub["size"], color=SECTOR_COLORS.get(sector, DEFAULT_COLOR),
                    line=dict(width=0.5, color="rgba(80,80,80,0.4)"), opacity=0.72),
        text=sub["hover"], hoverinfo="text",
        name=sector, legendgroup=sector,
    ))

# India-present nodes (bold border)
india_nodes = nodes[nodes["india_present"] & ~nodes["top20"]]
for sector in sorted(india_nodes["sector"].dropna().unique()):
    sub = india_nodes[india_nodes["sector"] == sector]
    if len(sub) == 0:
        continue
    fig.add_trace(go.Scatter(
        x=sub["x"], y=sub["y"], mode="markers",
        marker=dict(size=sub["size"] + 3, color=SECTOR_COLORS.get(sector, DEFAULT_COLOR),
                    line=dict(width=2.5, color="#222222"), opacity=1.0),
        text=sub["hover"], hoverinfo="text",
        name=f"IND: {sector}", legendgroup=sector, showlegend=False,
    ))

# XGBoost Top-20 prediction nodes  — bright star ring (gold outline, larger)
top20_nodes = nodes[nodes["top20"]]
fig.add_trace(go.Scatter(
    x=top20_nodes["x"], y=top20_nodes["y"], mode="markers+text",
    marker=dict(
        size=top20_nodes["size"] + 9,
        color=top20_nodes["color"],
        line=dict(width=3.5, color="#FFD700"),  # gold ring
        opacity=1.0,
        symbol="star",
    ),
    text=["★" for _ in range(len(top20_nodes))],
    textposition="top center",
    textfont=dict(size=8, color="#FFD700"),
    hovertext=top20_nodes["hover"],
    hoverinfo="text",
    name="⭐ XGBoost Top-20 (India 2010)",
    showlegend=True,
))

# Legend dummies
fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
    marker=dict(size=12, color="white", line=dict(width=2.5, color="#222222")),
    name="🇮🇳 India exports (2010)", showlegend=True))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
    marker=dict(size=14, color="gold", symbol="star",
                line=dict(width=3.5, color="#FFD700")),
    name="⭐ XGBoost Top-20 prediction", showlegend=True))

fig.update_layout(
    title=dict(
        text=(
            "Product Space — India 2010 + XGBoost Top-20 Predictions<br>"
            "<sup>Node colour = sector · Size = PCI · "
            "Bold border = India present · Gold star = XGBoost Top-20</sup>"
        ),
        x=0.5, font=dict(size=16),
    ),
    showlegend=True,
    legend=dict(
        title="Sectors & overlays", font=dict(size=10),
        bgcolor="rgba(255,255,255,0.85)", bordercolor="#ccc", borderwidth=1,
        itemsizing="constant", x=1.01, y=1, xanchor="left",
    ),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    plot_bgcolor="white", paper_bgcolor="white",
    width=1440, height=920,
    margin=dict(l=20, r=320, t=90, b=20),
    hovermode="closest",
)

fig.write_html(str(HTML_OUT), include_plotlyjs="cdn")
print(f"  Saved: {HTML_OUT}")

print("\n" + "=" * 70)
print("ALL STEPS COMPLETE")
print("=" * 70)
