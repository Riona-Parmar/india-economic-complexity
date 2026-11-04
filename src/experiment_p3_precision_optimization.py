"""Compare precision-oriented P3 models on the existing temporal folds."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
    balanced_accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRanker

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/p3_dataset_v2.parquet"
SAVED_PREDICTIONS = ROOT / "outputs/p3_weak_signal_validation_predictions.parquet"
SAVED_RESULTS = ROOT / "outputs/p3_weak_signal_feature_results.csv"
SAVED_IMPORTANCE = ROOT / "outputs/p3_weak_signal_feature_importance.csv"
OUT = ROOT / "outputs"
RESULTS = OUT / "p3_precision_optimization_results.csv"
FOLD_METRICS = OUT / "p3_precision_optimization_fold_metrics.csv"
PREDICTIONS = OUT / "p3_precision_optimization_predictions.parquet"
IMPORTANCE = OUT / "p3_precision_optimization_feature_importance.csv"
SEED = 42
THRESHOLD = 0.5
FOLDS = [(2005, [2000]), (2010, [2000, 2005]), (2015, [2000, 2005, 2010])]
BASE_PARAMS = dict(max_depth=3, min_child_weight=5, subsample=.8,
                   colsample_bytree=1.0, learning_rate=.05, n_estimators=400)
BASE = ["density", "pci_current", "rca3_current", "rca3_change_3y", "eci_current",
        "f_ubiquity", "f_world_growth", "log_export_change_1y", "log_export_change_3y",
        "log_export_value", "rca_current", "atlas_distance_current", "atlas_cog_current"]
DIAGNOSTIC_FEATURES = ["log_export_value", "density", "pci_current", "rca_current",
                       "log_export_change_3y"]


def ranking_metrics(frame):
    y = frame.label.astype(int).to_numpy()
    s = frame.score.astype(float).to_numpy()
    p10, p20, r10, r20, random_p = [], [], [], [], []
    country_rows = []
    for country, g in frame.groupby("country", sort=True):
        g = g.sort_values("score", ascending=False, kind="mergesort")
        actual = g.label.astype(int).to_numpy()
        hit10 = int(actual[:10].sum()); hit20 = int(actual[:20].sum())
        p10.append(hit10 / min(10, len(actual))); p20.append(hit20 / min(20, len(actual)))
        random_p.append(float(actual.mean()))
        if actual.sum():
            r10.append(hit10 / int(actual.sum())); r20.append(hit20 / int(actual.sum()))
        country_rows.append((country, hit10/min(10,len(actual)), hit20/min(20,len(actual)),
                             hit10/int(actual.sum()) if actual.sum() else np.nan,
                             hit20/int(actual.sum()) if actual.sum() else np.nan))
    p10m=float(np.mean(p10)); p20m=float(np.mean(p20)); base=float(y.mean())
    cm=confusion_matrix(y, s >= THRESHOLD, labels=[0,1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    country=np.asarray(country_rows, dtype=object)
    return {
        "PR_AUC":average_precision_score(y,s), "ROC_AUC":roc_auc_score(y,s),
        "Precision_at_10":p10m,"Precision_at_20":p20m,
        "Recall_at_10":float(np.mean(r10)) if r10 else np.nan,
        "Recall_at_20":float(np.mean(r20)) if r20 else np.nan,
        "Lift_at_10":p10m/base if base else np.nan,"Lift_at_20":p20m/base if base else np.nan,
        "macro_country_random_precision":float(np.mean(random_p)),"base_rate":base,
        "accuracy_at_0_5":accuracy_score(y,s>=THRESHOLD),
        "majority_class_accuracy":float((y==0).mean()),
        "precision_at_0_5":precision_score(y,s>=THRESHOLD,zero_division=0),
        "recall_at_0_5":recall_score(y,s>=THRESHOLD,zero_division=0),
        "F1_at_0_5":f1_score(y,s>=THRESHOLD,zero_division=0),
        "specificity_at_0_5":tn/(tn+fp) if tn+fp else np.nan,
        "balanced_accuracy_at_0_5":balanced_accuracy_score(y,s>=THRESHOLD),
        "TN_at_0_5":tn,"FP_at_0_5":fp,"FN_at_0_5":fn,"TP_at_0_5":tp,
        "score_mean":float(np.mean(s)),"score_std":float(np.std(s)),
        "score_q50":float(np.quantile(s,.50)),"score_q90":float(np.quantile(s,.90)),
        "score_q99":float(np.quantile(s,.99)),"score_unique":int(np.unique(s).size),
        "score_positive_mean":float(np.mean(s[y==1])) if np.any(y==1) else np.nan,
        "score_negative_mean":float(np.mean(s[y==0])) if np.any(y==0) else np.nan,
        "country_precision20_std":float(np.nanstd(country[:,2].astype(float),ddof=1)),
        "country_precision20_min":float(np.nanmin(country[:,2].astype(float))),
        "country_precision20_max":float(np.nanmax(country[:,2].astype(float))),
        "country_recall20_std":float(np.nanstd(country[:,4].astype(float),ddof=1)),
        "country_count":int(frame.country.nunique()),"validation_rows":len(frame),
        "positive_count":int(y.sum()),"threshold":THRESHOLD,
        "weak_positive_recall20":_subgroup_recall(frame,"weak_positive",20),
        "high_pci_positive_recall20":_subgroup_recall(frame,"high_pci_positive",20),
        "low_density_positive_recall20":_subgroup_recall(frame,"low_density_positive",20),
        "weak_positive_recall10":_subgroup_recall(frame,"weak_positive",10),
        "majority_confusion_TN":int((y==0).sum()),"majority_confusion_FP":0,
        "majority_confusion_FN":int(y.sum()),"majority_confusion_TP":0,
        "top20_false_positives":int(sum(
            (g.sort_values("score",ascending=False,kind="mergesort").label.head(20)==0).sum()
            for _,g in frame.groupby("country"))),
    }


def _subgroup_recall(frame, col, k):
    if col not in frame: return np.nan
    values=[]
    for _, g in frame.groupby("country",sort=True):
        g=g.sort_values("score",ascending=False,kind="mergesort")
        mask=g[col].fillna(False).astype(bool).to_numpy()
        denom=int(mask.sum())
        if denom: values.append(float(mask[:k].sum()/denom))
    return float(np.mean(values)) if values else np.nan


def add_ranks(frame):
    frame=frame.sort_values(["country","score"],ascending=[True,False],kind="mergesort").copy()
    frame["country_rank"]=frame.groupby("country",sort=False).cumcount()+1
    frame["top10_selected"]=frame.country_rank.le(10)
    frame["top20_selected"]=frame.country_rank.le(20)
    frame["threshold_prediction_at_0_5"]=frame.score.ge(THRESHOLD)
    frame["confusion_class_at_0_5"]=np.select(
        [frame.label.eq(1)&frame.threshold_prediction_at_0_5,
         frame.label.eq(0)&frame.threshold_prediction_at_0_5,frame.label.eq(1)],
        ["TP","FP","FN"],default="TN")
    return frame


def make_xgb(weight, params=BASE_PARAMS):
    return XGBClassifier(objective="binary:logistic",eval_metric="aucpr",scale_pos_weight=weight,
                         random_state=SEED,n_jobs=-1,**params)


def main():
    for p in [RESULTS,FOLD_METRICS,PREDICTIONS,IMPORTANCE]:
        if p.exists(): raise FileExistsError(f"Refusing to overwrite {p}")
    if not SAVED_PREDICTIONS.exists():
        raise FileNotFoundError(f"Required existing baseline/F predictions missing: {SAVED_PREDICTIONS}")
    df=pd.read_parquet(DATA)
    if not df.split.eq("train").any(): raise ValueError("Train split absent")
    # Use the processed P3 features as-is; do not rebuild upstream inputs.
    print("Loading existing P3 features; P1/P2 files are read-only...",flush=True)
    train=df.loc[df.split.eq("train")].copy()
    prior=pd.read_parquet(SAVED_PREDICTIONS)
    prior=prior.loc[prior.feature_set.isin(["A_all_signal_baseline","F_all_feasible_features"])].copy()
    prior_results=pd.read_csv(SAVED_RESULTS)
    folds_out=[]; preds_out=[]; importance=[]
    saved_gain=pd.read_csv(SAVED_IMPORTANCE)
    saved_gain=saved_gain.loc[saved_gain.feature_set.eq("F_all_feasible_features")]
    importance.extend([{"model":"XGB-F-existing","validation_year":int(r.validation_year),
        "feature":r.feature,"importance":float(r.gain),"importance_type":"gain"}
        for r in saved_gain.itertuples() if pd.notna(r.validation_year)])
    for year,train_years in FOLDS:
        tr=train.loc[train.asof_year.isin(train_years)].copy()
        va=train.loc[train.asof_year.eq(year)].copy()
        if tr.empty or va.empty: raise ValueError(f"Missing fold {year}")
        if not tr[BASE].notna().all().all() or not va[BASE].notna().all().all(): raise ValueError("Missing model features")
        y=tr.label.astype(int); ratio=float((y==0).sum()/(y==1).sum())
        qscale=tr.log_export_value.quantile(.25); qden=tr.density.quantile(.25)
        weak=va.label.eq(1)&va.log_export_value.le(qscale)&va.density.le(qden)
        pciq=tr.pci_current.quantile(.75)
        highpci=va.label.eq(1)&va.pci_current.ge(pciq)
        lowden=va.label.eq(1)&va.density.le(qden)
        va["weak_positive"]=weak; va["high_pci_positive"]=highpci; va["low_density_positive"]=lowden
        va["export_present_current"]=va.export_value_current.fillna(0).gt(0).astype(int)
        va["density_low_flag"]=va.density.le(qden).astype(int); va["pci_high_flag"]=va.pci_current.ge(pciq).astype(int)
        common=va[["asof_year","target_year","country","product","label",*DIAGNOSTIC_FEATURES,
                   "weak_positive","high_pci_positive","low_density_positive","density_low_flag","pci_high_flag"]].copy()
        existing=prior.loc[prior.asof_year.eq(year)]
        for cand,feature_set in [("XGB-A-baseline","A_all_signal_baseline"),("XGB-F-existing","F_all_feasible_features")]:
            p=existing.loc[existing.feature_set.eq(feature_set),["asof_year","country","product","score"]]
            old=common.merge(p,on=["asof_year","country","product"],how="left",validate="one_to_one")
            if old.score.isna().any(): raise ValueError(f"Missing stored scores for {cand}/{year}")
            if cand=="XGB-A-baseline":
                model=make_xgb(ratio); model.fit(tr[BASE],y)
                frame=common.copy(); frame["score"]=model.predict_proba(va[BASE])[:,1]
                score_diff=float(np.max(np.abs(frame.score.to_numpy()-old.score.to_numpy())))
                if score_diff>1e-7: raise ValueError(f"Stored baseline score reproduction mismatch in {year}: max diff {score_diff}")
                folds_out.append(_row(cand,year,train_years,ratio,BASE_PARAMS,frame,model,tr,BASE))
                importance.extend(_importance(model,cand,year,BASE,"gain"))
            else:
                frame=old
                saved=prior_results.loc[prior_results.result_level.eq("fold")&prior_results.feature_set.eq(feature_set)&prior_results.validation_year.eq(year)].iloc[0]
                m=ranking_metrics(frame)
                m["train_PR_AUC"]=saved["train_PR-AUC"]; m["PR_AUC_train_minus_validation"]=saved["train_PR-AUC"]-m["PR_AUC"]
                m.update({"model":cand,"result_level":"fold","validation_year":year,"training_years":str(train_years),
                    "training_rows":len(tr),"training_positive_count":int(y.sum()),"scale_pos_weight":ratio,
                    "model_parameters":json.dumps(BASE_PARAMS,sort_keys=True),"features":"baseline + previously evaluated feasible P3 additions",
                    "seed":SEED,"top20_false_positive_share_zero_export":_top20_diag(frame,"export_present_current",0),
                    "top20_false_positive_share_low_density":_top20_diag(frame,"density_low_flag",1),
                    "top20_false_positive_share_high_pci":_top20_diag(frame,"pci_high_flag",1)})
                folds_out.append(m)
            preds_out.append(add_ranks(frame.assign(model=cand,training_years=str(train_years))))
        # Reuse F scores; fit new tuning candidates on baseline features.
        for mult in [.25,.5,2.0]:
            name=f"XGB-A-weight_{mult:.2f}"
            print(f"{name}, validation {year}",flush=True)
            model=make_xgb(ratio*mult)
            model.fit(tr[BASE],y)
            frame=common.copy(); frame["score"]=model.predict_proba(va[BASE])[:,1]
            preds_out.append(add_ranks(frame.assign(model=name,training_years=str(train_years))))
            folds_out.append(_row(name,year,train_years,ratio*mult,BASE_PARAMS,frame,model,tr,BASE))
            importance.extend(_importance(model,name,year,BASE,"gain"))
        # Single pre-specified regularization check, motivated by overfit gap.
        params={**BASE_PARAMS,"max_depth":2}
        name="XGB-A-depth2"
        print(f"{name}, validation {year}",flush=True)
        model=make_xgb(ratio,params); model.fit(tr[BASE],y)
        frame=common.copy(); frame["score"]=model.predict_proba(va[BASE])[:,1]
        preds_out.append(add_ranks(frame.assign(model=name,training_years=str(train_years))))
        folds_out.append(_row(name,year,train_years,ratio,params,frame,model,tr,BASE))
        importance.extend(_importance(model,name,year,BASE,"gain"))
        # Transparent linear benchmark; scaler fits on this fold's training rows only.
        name="Logistic-A-balanced"
        print(f"{name}, validation {year}",flush=True)
        logit=make_pipeline(StandardScaler(),LogisticRegression(class_weight="balanced",C=1.0,
                          max_iter=1000,random_state=SEED,solver="lbfgs"))
        logit.fit(tr[BASE],y)
        frame=common.copy(); frame["score"]=logit.predict_proba(va[BASE])[:,1]
        preds_out.append(add_ranks(frame.assign(model=name,training_years=str(train_years))))
        folds_out.append(_row(name,year,train_years,np.nan,{"C":1.0,"class_weight":"balanced","scaler":"StandardScaler"},frame,logit,tr,BASE))
        coef=logit[-1].coef_[0]
        importance.extend([{"model":name,"validation_year":year,"feature":f,"importance":float(v),"importance_type":"standardized_signed_coefficient"} for f,v in zip(BASE,coef)])
        # Pairwise ranking trains within country-year groups, aligned to the
        # country-wise top-K evaluation objective. Use the same binary labels.
        name="XGBRanker-A-pairwise"
        print(f"{name}, validation {year}",flush=True)
        sort_cols=["country","asof_year","product"]
        tr_rank=tr.sort_values(sort_cols,kind="mergesort").copy()
        qid=tr_rank.groupby(["country","asof_year"],sort=True).ngroup().to_numpy()
        ranker=XGBRanker(objective="rank:pairwise",eval_metric="ndcg@20",n_estimators=200,
                         max_depth=3,min_child_weight=5,learning_rate=.05,subsample=.8,
                         colsample_bytree=1.0,random_state=SEED,n_jobs=-1)
        ranker.fit(tr_rank[BASE],tr_rank.label.astype(int),qid=qid)
        frame=common.copy(); frame["score"]=ranker.predict(va[BASE])
        preds_out.append(add_ranks(frame.assign(model=name,training_years=str(train_years))))
        folds_out.append(_row(name,year,train_years,np.nan,{"objective":"rank:pairwise","n_estimators":200,
            "max_depth":3,"min_child_weight":5,"learning_rate":.05,"subsample":.8,"colsample_bytree":1.0,
            "ranking_group":"country x asof_year"},frame,ranker,tr,BASE,threshold_applicable=False))
        importance.extend(_importance(ranker,name,year,BASE,"gain"))
    folds=pd.DataFrame(folds_out)
    folds.to_csv(FOLD_METRICS,index=False)
    numeric=[c for c in folds.select_dtypes(include=np.number).columns if c not in ["validation_year","training_rows","validation_rows","positive_count","training_positive_count"]]
    summaries=[]
    for model,g in folds.groupby("model",sort=False):
        row={"model":model,"result_level":"mean_over_3_folds","fold_count":len(g)}
        for c in numeric:
            row[c]=float(g[c].mean())
            row[c+"_fold_std"]=float(g[c].std(ddof=1))
        row["mean_validation_years"]="2005|2010|2015"
        summaries.append(row)
    summary=pd.DataFrame(summaries)
    metric_names=["PR_AUC","Precision_at_10","Precision_at_20","Recall_at_10","Recall_at_20","Lift_at_10","Lift_at_20",
                  "balanced_accuracy_at_0_5","F1_at_0_5","weak_positive_recall20"]
    base=summary.set_index("model").loc["XGB-A-baseline"]
    for metric in metric_names:
        summary[metric+"_absolute_delta_vs_baseline"]=summary[metric]-base[metric]
        summary[metric+"_relative_delta_vs_baseline"]=summary[metric]/base[metric]-1 if base[metric]!=0 else np.nan
    summary.to_csv(RESULTS,index=False)
    pd.concat(preds_out,ignore_index=True).to_parquet(PREDICTIONS,index=False)
    pd.DataFrame(importance).to_csv(IMPORTANCE,index=False)
    print("Completed; wrote new P3 precision optimization outputs.",flush=True)


def _row(name,year,train_years,weight,params,frame,model,tr,features,threshold_applicable=True):
    m=ranking_metrics(frame)
    # PR-AUC is score-based and also applies to the ranker's raw scores.
    if isinstance(model,LogisticRegression):
        train_scores=model.predict_proba(tr[features])[:,1]
    elif hasattr(model,"predict_proba"):
        train_scores=model.predict_proba(tr[features])[:,1]
    else:
        train_scores=model.predict(tr[features])
    train_pr=average_precision_score(tr.label.astype(int),train_scores)
    m["train_PR_AUC"]=train_pr; m["PR_AUC_train_minus_validation"]=train_pr-m["PR_AUC"]
    if not threshold_applicable:
        for key in ["accuracy_at_0_5","precision_at_0_5","recall_at_0_5","F1_at_0_5",
                    "specificity_at_0_5","balanced_accuracy_at_0_5","TN_at_0_5","FP_at_0_5","FN_at_0_5","TP_at_0_5"]:
            m[key]=np.nan
    m.update({"model":name,"result_level":"fold","validation_year":year,"training_years":str(train_years),
              "training_rows":len(tr),"training_positive_count":int(tr.label.sum()),"scale_pos_weight":weight,
              "model_parameters":json.dumps(params,sort_keys=True),"features":"|".join(features),"seed":SEED,
              "top20_false_positive_share_zero_export":_top20_diag(frame,"export_present_current",0),
              "top20_false_positive_share_low_density":_top20_diag(frame,"density_low_flag",1),
              "top20_false_positive_share_high_pci":_top20_diag(frame,"pci_high_flag",1)})
    return m


def _top20_diag(frame,feature,condition):
    if feature not in frame: return np.nan
    vals=[]
    for _,g in frame.groupby("country"):
        g=g.sort_values("score",ascending=False,kind="mergesort").head(20)
        fp=g.loc[g.label.eq(0),feature]
        if len(fp): vals.append(float((fp==condition).mean()))
    return float(np.mean(vals)) if vals else np.nan


def _importance(model,name,year,features,kind):
    values=model.get_booster().get_score(importance_type=kind)
    return [{"model":name,"validation_year":year,"feature":f,"importance":float(values.get(f,0)),"importance_type":kind} for f in features]


if __name__ == "__main__": main()
