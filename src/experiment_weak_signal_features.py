"""Compare weak-signal feature groups on train-only temporal folds."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/p3_dataset_v2.parquet"
TRADE = ROOT / "data/processed/trade_rca_v1.parquet"
OUT = ROOT / "outputs"
RESULTS = OUT / "p3_weak_signal_feature_results.csv"
IMPORTANCE = OUT / "p3_weak_signal_feature_importance.csv"
FEASIBILITY = OUT / "p3_weak_signal_feature_feasibility.csv"
PREDICTIONS = OUT / "p3_weak_signal_validation_predictions.parquet"

BASE = ["density", "pci_current", "rca3_current", "rca3_change_3y", "eci_current",
        "f_ubiquity", "f_world_growth", "log_export_change_1y", "log_export_change_3y",
        "log_export_value", "rca_current", "atlas_distance_current", "atlas_cog_current"]
EXT = ["export_present_current", "export_present_3y_window", "export_present_5y_window",
       "years_since_positive_export", "positive_export_years_5y"]
RELATED = ["related_rca_capability", "related_competitive_capability"]
INTERACTIONS = ["density_x_export_growth", "pci_x_density", "pci_x_export_growth",
                "related_capability_x_pci", "related_capability_x_density",
                "export_scale_x_growth"]
SETS = {
    "A_all_signal_baseline": BASE,
    "B_baseline_plus_extensive": BASE + EXT,
    "C_baseline_plus_related": BASE + RELATED,
    "D_baseline_plus_both": BASE + EXT + RELATED,
    "E_baseline_plus_interactions": BASE + INTERACTIONS,
    "F_all_feasible_features": BASE + EXT + RELATED + INTERACTIONS,
}
FOLDS = [(2005, [2000]), (2010, [2000, 2005]), (2015, [2000, 2005, 2010])]
PARAMS = dict(max_depth=3, min_child_weight=5, subsample=.8, colsample_bytree=1.0,
              learning_rate=.05, n_estimators=400)
SEED = 42
BOOTSTRAPS = 2000


def make_extra_features(df):
    # Construct new features only for training countries. Recompute as-of RCA
    # and Phi with the holdout country set removed from the reference universe.
    train_countries = set(df.loc[df.split.eq("train"), "country"].unique())
    trade = pd.read_parquet(TRADE, columns=["country", "product", "year", "export_value"])
    trade = trade.loc[trade.country.isin(train_countries)].copy()
    trade["present"] = trade.export_value.fillna(0).gt(0).astype(np.int8)
    pieces = []
    for year in sorted(df.asof_year.unique()):
        snap = df.loc[df.asof_year.eq(year) & df.split.eq("train"), ["country", "product"]].copy()
        # Presence history is computed from the annual panel only through as-of.
        history = trade.loc[trade.year.le(year)].copy()
        years = np.arange(max(int(trade.year.min()), int(year) - 4), int(year) + 1)
        recent = history.loc[history.year.isin(years)]
        grouped = recent.groupby(["country", "product"], sort=False).present.sum().rename(
            "positive_export_years_5y").reset_index()
        last_positive = history.loc[history.present.eq(1)].groupby(
            ["country", "product"], sort=False).year.max().rename("last_positive_year").reset_index()
        grouped = grouped.merge(last_positive, on=["country", "product"], how="left", validate="one_to_one")
        # Reindex each requested pair; absence in the trade panel is treated as zero exports.
        snap = snap.merge(grouped, on=["country", "product"], how="left", validate="one_to_one")
        snap["positive_export_years_5y"] = snap.positive_export_years_5y.fillna(0).astype(int)
        current = trade.loc[trade.year.eq(year), ["country", "product", "present"]]
        snap = snap.merge(
            current.rename(columns={"present": "export_present_current"}),
            on=["country", "product"], how="left", validate="one_to_one")
        snap["export_present_current"] = snap.export_present_current.fillna(0).astype(int)
        for window, name in [(3, "export_present_3y_window"), (5, "export_present_5y_window")]:
            lo = int(year) - window + 1
            flags = trade.loc[trade.year.between(lo, int(year))].groupby(["country", "product"]).present.max()
            idx = pd.MultiIndex.from_frame(snap[["country", "product"]])
            snap[name] = flags.reindex(idx).fillna(0).to_numpy(dtype=np.int8)
        snap["years_since_positive_export"] = (int(year) - snap.last_positive_year).fillna(int(year)-int(trade.year.min())+1).clip(upper=31)
        snap["asof_year"] = int(year)
        pieces.append(snap.drop(columns="last_positive_year"))
    ext = pd.concat(pieces, ignore_index=True)

    # Reconstruct each as-of product space with training countries only.
    related_parts = []
    for year in sorted(df.asof_year.unique()):
        t = trade.loc[trade.year.eq(year)]
        products = sorted(trade["product"].unique())
        countries_all = sorted(train_countries)
        export_mat = t.pivot(index="country", columns="product", values="export_value").reindex(
            index=countries_all, columns=products).fillna(0).to_numpy(dtype=np.float64)
        country_totals = export_mat.sum(axis=1, keepdims=True)
        world_total = export_mat.sum()
        product_totals = export_mat.sum(axis=0, keepdims=True)
        rca_train = np.divide(export_mat, country_totals, out=np.zeros_like(export_mat), where=country_totals > 0)
        rca_train = np.divide(rca_train, product_totals / world_total,
                              out=np.zeros_like(rca_train), where=(product_totals > 0) & (world_total > 0))
        m_train = (rca_train >= 1).astype(np.float32)
        co = m_train.T @ m_train
        ubiquity = m_train.sum(axis=0)
        denom = np.maximum(ubiquity[:, None], ubiquity[None, :])
        mat = np.divide(co, denom, out=np.zeros_like(co), where=denom > 0).astype(np.float32)
        den = mat.sum(axis=1)
        snap = df.loc[df.asof_year.eq(year) & df.split.eq("train"), ["country", "product"]]
        country_index = {c: i for i, c in enumerate(countries_all)}
        for country in sorted(snap.country.unique()):
            rca = rca_train[country_index[country]].astype(np.float32)
            m = m_train[country_index[country]]
            rcap = mat @ rca
            comp = mat @ m
            q = snap.loc[snap.country.eq(country), ["country", "product"]].copy()
            ix = pd.Index(products).get_indexer(q["product"])
            q["related_rca_capability"] = np.divide(rcap[ix], den[ix], out=np.zeros(len(ix), dtype=np.float32), where=den[ix] > 0)
            q["related_competitive_capability"] = np.divide(comp[ix], den[ix], out=np.zeros(len(ix), dtype=np.float32), where=den[ix] > 0)
            q["asof_year"] = int(year)
            related_parts.append(q)
    rel = pd.concat(related_parts, ignore_index=True)
    extra = ext.merge(rel, on=["asof_year", "country", "product"], how="left", validate="one_to_one")
    out = df.merge(extra, on=["asof_year", "country", "product"], how="left", validate="one_to_one")
    out["density_x_export_growth"] = out.density * out.log_export_change_3y
    out["pci_x_density"] = out.pci_current * out.density
    out["pci_x_export_growth"] = out.pci_current * out.log_export_change_3y
    out["related_capability_x_pci"] = out.related_competitive_capability * out.pci_current
    out["related_capability_x_density"] = out.related_competitive_capability * out.density
    out["export_scale_x_growth"] = out.log_export_value * out.log_export_change_3y
    return out


def metrics(data):
    y = data.label.to_numpy(dtype=int); s = data.score.to_numpy(dtype=float)
    p10=[]; p20=[]; r20=[]; weak_r=[]; highpci_r=[]; lowden_r=[]
    for _, g in data.groupby("country", sort=True):
        g = g.sort_values("score", ascending=False, kind="mergesort")
        hit = g.label.to_numpy(dtype=int)[:20].sum()
        p10.append(g.label.to_numpy(dtype=int)[:10].mean())
        p20.append(hit/20)
        if g.label.sum(): r20.append(hit/g.label.sum())
        for mask, sink in [(g.weak_positive, weak_r), (g.high_pci_positive, highpci_r), (g.low_density_positive, lowden_r)]:
            n = int(mask.sum())
            if n: sink.append(int((g.label.to_numpy(dtype=int)[:20] * mask.to_numpy(dtype=int)[:20]).sum()) / n)
    prec20=float(np.mean(p20)); base=float(y.mean())
    return {"PR-AUC":average_precision_score(y,s),"ROC-AUC":roc_auc_score(y,s),
            "Brier":brier_score_loss(y,s),"Precision@10":float(np.mean(p10)),"Precision@20":prec20,
            "Recall@20":float(np.mean(r20)),"Lift@20":prec20/base,"base_rate":base,
            "weak_positive_recall20":float(np.mean(weak_r)) if weak_r else np.nan,
            "high_pci_positive_recall20":float(np.mean(highpci_r)) if highpci_r else np.nan,
            "low_density_positive_recall20":float(np.mean(lowden_r)) if lowden_r else np.nan}


def main():
    for p in [RESULTS, IMPORTANCE, FEASIBILITY, PREDICTIONS]:
        if p.exists(): raise FileExistsError(f"Refusing to overwrite {p}")
    df = make_extra_features(pd.read_parquet(DATA))
    if not df.split.eq("train").any(): raise ValueError("No train split")
    train = df.loc[df.split.eq("train")].copy()
    rows=[]; imps=[]; preds=[]; country_p20={}
    for name, features in SETS.items():
        for year, train_years in FOLDS:
            tr=train.loc[train.asof_year.isin(train_years)].copy(); va=train.loc[train.asof_year.eq(year)].copy()
            if tr.empty or va.empty: raise ValueError(f"Missing fold {year}")
            if tr[features].isna().any().any() or va[features].isna().any().any(): raise ValueError(f"Missing feature values in {name}/{year}")
            y=tr.label.astype(int); scale=(y.eq(0).sum()/y.eq(1).sum())
            model=XGBClassifier(objective="binary:logistic",eval_metric="aucpr",scale_pos_weight=scale,
                random_state=SEED,n_jobs=-1,**PARAMS)
            model.fit(tr[features],y)
            # Weak-signal cut points are learned only from this training fold.
            qs=tr[["log_export_value","density"]].quantile(.25)
            va["weak_positive"] = va.label.eq(1) & va.log_export_value.le(qs.log_export_value) & va.density.le(qs.density)
            va["high_pci_positive"] = va.label.eq(1) & va.pci_current.ge(tr.pci_current.quantile(.75))
            va["low_density_positive"] = va.label.eq(1) & va.density.le(qs.density)
            va["score"] = model.predict_proba(va[features])[:,1]
            m=metrics(va); train_score=model.predict_proba(tr[features])[:,1]
            m["train_PR-AUC"]=average_precision_score(y,train_score)
            m["PR_AUC_train_minus_validation"]=m["train_PR-AUC"]-m["PR-AUC"]
            m.update({"result_level":"fold","feature_set":name,"validation_year":year,"training_years":str(train_years),
                "training_rows":len(tr),"validation_rows":len(va),"training_positives":int(y.sum()),"positive_count":int(va.label.sum()),
                "weak_positive_count":int(va.weak_positive.sum()),"weak_export_density_q25":float(qs.log_export_value),"density_q25":float(qs.density),
                "seed":SEED,**PARAMS})
            rows.append(m)
            country_p20[(name,year)]={c:float(g.sort_values("score",ascending=False,kind="mergesort").label.head(20).mean()) for c,g in va.groupby("country")}
            for f,v in model.get_booster().get_score(importance_type="gain").items(): imps.append({"feature_set":name,"validation_year":year,"feature":f,"gain":v})
            preds.append(va[["asof_year","target_year","country","product","label","score","weak_positive","high_pci_positive","low_density_positive"]].copy().assign(feature_set=name))
            print(name,year,{k:round(v,5) for k,v in m.items() if k in ["PR-AUC","Precision@20","Recall@20","weak_positive_recall20"]})
    # Paired bootstrap uses countries as clusters, and is descriptive only.
    for row in rows:
        key=(row["feature_set"],row["validation_year"])
        base=country_p20[("A_all_signal_baseline",row["validation_year"])]
        cand=country_p20[key]; cs=sorted(base)
        d=np.array([cand[c]-base[c] for c in cs]); rng=np.random.default_rng(SEED+row["validation_year"])
        ix=rng.integers(0,len(cs),size=(BOOTSTRAPS,len(cs))); bs=d[ix].mean(axis=1)
        row["Precision@20_delta_vs_baseline"]=d.mean(); row["Precision@20_delta_ci95_low"]=np.percentile(bs,2.5); row["Precision@20_delta_ci95_high"]=np.percentile(bs,97.5)
    result=pd.DataFrame(rows)
    summary=result.groupby("feature_set",sort=False).mean(numeric_only=True).reset_index(); summary["result_level"]="mean_over_3_folds"; summary["validation_year"]=np.nan
    result=pd.concat([result,summary],ignore_index=True); result.to_csv(RESULTS,index=False)
    pd.DataFrame(imps).to_csv(IMPORTANCE,index=False)
    pd.concat(preds,ignore_index=True).to_parquet(PREDICTIONS,index=False)
    feas=[
      ("export_present_current","1[exports(c,p,t)>0]","Annual trade_rca_v1 as-of snapshot","Extensive margin at prediction time","Verified, <= as-of","Trade panel zeros/missing interpreted as no exports", "Implemented"),
      ("export_present_3y_window","1[any positive export in t-2..t]","Annual trade_rca_v1","Recent export presence","Verified, <= as-of","Calendar-year panel supports exact window", "Implemented"),
      ("export_present_5y_window","1[any positive export in t-4..t]","Annual trade_rca_v1","Longer export recency","Verified, <= as-of","Calendar-year panel supports exact window", "Implemented"),
      ("years_since_positive_export","t - last positive-export year; capped at 31","Annual trade_rca_v1, 1995-2024","Export absence duration","Verified for available history","Left censoring before 1995; cap marks no observed positive year", "Implemented"),
      ("positive_export_years_5y","count of positive export years in t-4..t","Annual trade_rca_v1","Persistence vs sporadic exports","Verified, <= as-of","Counts annual positive observations", "Implemented"),
      ("related_rca_capability","sum_k phi_train[p,k]*RCA_train[c,k,t] / sum_k phi_train[p,k]","Annual trade exports; as-of RCA and proximity rebuilt from train-split countries only","Capability depth among related products","Recomputed at each of 2000/05/10/15 from exports available by as-of","Country reference set excludes IND/BGD/IDN/MEX/TUR/VNM; baseline global aggregates remain mixed", "Implemented"),
      ("related_competitive_capability","sum_k phi_train[p,k]*1[RCA_train[c,k,t]>=1] / sum_k phi_train[p,k]","Annual trade exports; as-of RCA and proximity rebuilt from train-split countries only","Competitive capability in related products","Recomputed at each as-of year","Country reference set excludes IND/BGD/IDN/MEX/TUR/VNM; new related features are country-isolated", "Implemented"),
      ("related capability reference universe","Recompute RCA and phi using exports from train countries only; phi_ij = co-competitive-country count / max(ubiquity_i, ubiquity_j)","Annual export_value in trade_rca_v1, restricted to train-split countries","Prevents holdout countries from contributing to the new related-product scores","Year-matched 2000/05/10/15 inputs available","Underlying original density/PCI/ECI/Atlas/ubiquity/world-growth remain cross-country aggregates", "Implemented for new related-product scores"),
      ("density_x_export_growth","density * log_export_change_3y","Baseline as-of features","Export momentum may translate differently with product-space proximity","Verified, <= as-of","Interaction of inherited cross-country density and country-level growth", "Implemented"),
      ("pci_x_density","pci_current * density","Baseline as-of features","Complexity may matter differently for nearby products","Verified, <= as-of","PCI and density are inherited global aggregates", "Implemented"),
      ("pci_x_export_growth","pci_current * log_export_change_3y","Baseline as-of features","Growth in complex products may carry distinct signal","Verified, <= as-of","PCI inherited global aggregate", "Implemented"),
      ("related_capability_x_pci","related_competitive_capability * pci_current","Train-only related capability + baseline PCI","Related capabilities may matter differently by target complexity","Verified, <= as-of","PCI inherited global aggregate", "Implemented"),
      ("related_capability_x_density","related_competitive_capability * density","Train-only related capability + baseline density","Separates direct capability overlap from density context","Verified, <= as-of","Density inherited global aggregate", "Implemented"),
      ("export_scale_x_growth","log_export_value * log_export_change_3y","Country-product as-of export scale and growth","Scale may condition the meaning of recent growth","Verified, <= as-of","Both inputs are as-of; can emphasize established exporters", "Implemented")]
    pd.DataFrame(feas,columns=["feature","definition","source","intuition","historical_availability","leakage_or_limit","status"]).to_csv(FEASIBILITY,index=False)
    print(f"Saved {RESULTS}, {IMPORTANCE}, {FEASIBILITY}, {PREDICTIONS}")

if __name__ == "__main__": main()
