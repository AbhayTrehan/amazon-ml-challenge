#!/usr/bin/env python3
"""Fit one LightGBM model, calibrate E001-E003 on fold 4, freeze, score fold 0."""
from __future__ import annotations

import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "analysis/experiments_v1/e001_classical_baseline"
START = time.time(); PROC = psutil.Process(); PEAK = PROC.memory_info().rss


def peak():
    global PEAK
    PEAK = max(PEAK, PROC.memory_info().rss)


def json_clean(value):
    if isinstance(value, dict): return {k: json_clean(v) for k,v in value.items()}
    if isinstance(value, list): return [json_clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)): return None
    if isinstance(value, np.generic): return value.item()
    return value


NONFEATURE = {"s1", "target_id", "country", "role", "label", "in_k10", "source"}


def load_role(role):
    paths = sorted((EXP / "cache/features").glob(f"{role}_*.parquet"))
    if not paths:
        raise RuntimeError(f"No feature shards for {role}; run prepare_baseline.py first")
    x = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    peak(); return x


def compact_training_pairs(train):
    """Cached preparation keeps all positives and 15 hard negatives per source/S1."""
    return train


def metric_arrays(frame, all_ids, truth_counts, mask):
    idxmap = pd.Series(np.arange(len(all_ids), dtype=np.int32), index=all_ids)
    gi = idxmap.loc[frame.s1].to_numpy()
    use = np.asarray(mask, dtype=bool)
    pred = np.bincount(gi[use], minlength=len(all_ids))
    tp = np.bincount(gi[use], weights=frame.label.to_numpy()[use], minlength=len(all_ids))
    true = truth_counts.astype(np.float64)
    den = 1.25*tp + .25*(true-tp) + (pred-tp)
    f = np.divide(1.25*tp, den, out=np.zeros_like(den), where=den>0)
    f[(true == 0) & (pred == 0)] = 1.0
    total_tp = tp.sum(); total_pred = pred.sum(); total_true = true.sum()
    return {
        "macro_f05": float(f.mean()),
        "precision": float(total_tp/total_pred) if total_pred else 1.0,
        "recall": float(total_tp/total_true) if total_true else 1.0,
        "singleton_accuracy": float(np.mean(pred[true == 0] == 0)) if np.any(true == 0) else np.nan,
        "predicted_edges": int(total_pred), "true_edges": int(total_true), "tp": int(total_tp)
    }


def eval_mask(frame, selected, mask):
    ids = selected.source1_entity_id.to_numpy()
    return metric_arrays(frame, ids, selected.true_matches.to_numpy(), mask)


def threshold_grid(center=None):
    if center is None:
        return np.unique(np.r_[np.arange(0.02, .981, .02), [.001, .005, .01, .99, .995, .999]])
    return np.unique(np.clip(np.arange(center-.04, center+.0401, .002), 0, 1))


def tune_independent(frame, selected, k):
    eligible = np.ones(len(frame), bool) if k == 20 else frame.in_k10.eq(1).to_numpy()
    rows=[]
    for stage, grid in [("coarse", threshold_grid()), ("fine", None)]:
        if grid is None:
            best = max(rows, key=lambda r:r["macro_f05"])["threshold"]
            grid = threshold_grid(best)
        for t in grid:
            m = eval_mask(frame, selected, eligible & frame.score.ge(t).to_numpy())
            rows.append({"experiment":"E001", "candidate_set":f"K{k}", "stage":stage, "threshold":float(t), "margin":np.nan, **m})
    return pd.DataFrame(rows), max(rows,key=lambda r:r["macro_f05"])


def ownership_table(frame, k):
    x = frame if k == 20 else frame[frame.in_k10.eq(1)]
    x = x.sort_values(["target_id","score","s1"], ascending=[True,False,True], kind="mergesort").copy()
    x["owner_rank"] = x.groupby("target_id", sort=False).cumcount()
    winner = x[x.owner_rank.eq(0)].copy()
    second = x[x.owner_rank.eq(1)].set_index("target_id").score
    winner["owner_margin"] = winner.score - winner.target_id.map(second).fillna(0)
    return winner


def tune_owner(frame, selected):
    win = ownership_table(frame, 20)
    rows=[]
    for stage, grid in [("coarse",threshold_grid()),("fine",None)]:
        if grid is None:
            best=max(rows,key=lambda r:r["macro_f05"])["threshold"]; grid=threshold_grid(best)
        for t in grid:
            m=eval_mask(win,selected,win.score.ge(t).to_numpy())
            rows.append({"experiment":"E002","candidate_set":"K20","stage":stage,"threshold":float(t),"margin":0.0,**m})
    return win,pd.DataFrame(rows),max(rows,key=lambda r:r["macro_f05"])


def tune_margin(win, selected, base_t):
    rows=[]
    coarse_t=np.unique(np.clip(np.arange(base_t-.04,base_t+.0401,.01),0,1))
    coarse_m=np.array([0,.005,.01,.02,.03,.05,.075,.10,.15,.20,.30],float)
    for t in coarse_t:
        for margin in coarse_m:
            met=eval_mask(win,selected,(win.score.ge(t)&win.owner_margin.ge(margin)).to_numpy())
            rows.append({"experiment":"E003","candidate_set":"K20","stage":"coarse","threshold":float(t),"margin":float(margin),**met})
    best=max(rows,key=lambda r:r["macro_f05"])
    fine_t=np.unique(np.clip(np.arange(best["threshold"]-.015,best["threshold"]+.0151,.003),0,1))
    fine_m=np.unique(np.clip(np.arange(best["margin"]-.025,best["margin"]+.0251,.005),0,1))
    for t in fine_t:
        for margin in fine_m:
            met=eval_mask(win,selected,(win.score.ge(t)&win.owner_margin.ge(margin)).to_numpy())
            rows.append({"experiment":"E003","candidate_set":"K20","stage":"fine","threshold":float(t),"margin":float(margin),**met})
    return pd.DataFrame(rows),max(rows,key=lambda r:r["macro_f05"])


def save_predictions(frame, selected, experiment, k, threshold, margin=0):
    if experiment == "E001":
        eligible = np.ones(len(frame),bool) if k==20 else frame.in_k10.eq(1).to_numpy()
        use = eligible & frame.score.ge(threshold).to_numpy()
        pred = frame.loc[use,["s1","target_id","score"]].copy()
    else:
        win=ownership_table(frame,k)
        use=win.score.ge(threshold)
        if experiment=="E003": use &= win.owner_margin.ge(margin)
        pred=win.loc[use,["s1","target_id","score","owner_margin"]].copy()
    pred.to_parquet(EXP/f"predictions/{experiment.lower()}_k{k}_evaluation_pairs.parquet",index=False)
    pmap=pred.groupby("s1").target_id.apply(lambda x:",".join(sorted(x))).to_dict()
    out=pd.DataFrame({"source1_entity_id":selected.source1_entity_id,"matched_entity_ids":selected.source1_entity_id.map(pmap).fillna("")})
    out.to_csv(EXP/f"predictions/{experiment.lower()}_k{k}_evaluation.tsv",sep="\t",index=False)
    return pred


def source_metric(frame, selected, mask, source):
    sub=frame[frame.source.eq(source)].copy(); submask=np.asarray(mask)[frame.source.eq(source).to_numpy()]
    truth_col=f"s{source}_matches"
    return metric_arrays(sub,selected.source1_entity_id.to_numpy(),selected[truth_col].to_numpy(),submask)


def slice_table(frame, selected, configs):
    rows=[]
    base=selected.set_index("source1_entity_id")
    for experiment,k,t,m in configs:
        if experiment=="E001":
            scored=frame if k==20 else frame[frame.in_k10.eq(1)].copy(); mask=scored.score.ge(t).to_numpy()
        else:
            scored=ownership_table(frame,k); mask=(scored.score.ge(t)&(scored.owner_margin.ge(m) if experiment=="E003" else True)).to_numpy()
        for src in [2,3]:
            met=source_metric(scored,selected,mask,src)
            rows.append({"experiment":experiment,"candidate_set":f"K{k}","slice":"source","value":f"S{src}","n_s1":len(selected),**met})
        group_defs=[]
        for col in ["country","cardinality_bucket"]:
            for val,ids in selected.groupby(col).source1_entity_id:
                group_defs.append((col,str(val),set(ids)))
        group_defs.append(("singleton","true",set(selected[selected.true_matches.eq(0)].source1_entity_id)))
        # Query ambiguity uses maximum observed name fanout across its candidates.
        fan=frame.groupby("s1").s1_normalized_name_fanout.max()
        amb_ids=set(fan[fan.gt(1)].index); group_defs += [("name_ambiguity","repeated/common",amb_ids),("name_ambiguity","non_ambiguous",set(selected.source1_entity_id)-amb_ids)]
        missing=frame[(frame.label.eq(1))&(frame.candidate_address_missing.eq(1))].s1.unique()
        group_defs += [("missing_true_candidate_address","true",set(missing)),("missing_true_candidate_address","false",set(selected.source1_entity_id)-set(missing))]
        for sl,val,ids in group_defs:
            ss=selected[selected.source1_entity_id.isin(ids)]
            if ss.empty: continue
            take=scored.s1.isin(ids).to_numpy()
            met=metric_arrays(scored.loc[take],ss.source1_entity_id.to_numpy(),ss.true_matches.to_numpy(),mask[take])
            rows.append({"experiment":experiment,"candidate_set":f"K{k}","slice":sl,"value":val,"n_s1":len(ss),**met})
    return pd.DataFrame(rows)


def main():
    selected=pd.read_parquet(EXP/"selected_ids/selected_s1.parquet")
    train=compact_training_pairs(load_role("train"))
    features=[c for c in train.columns if c not in NONFEATURE and pd.api.types.is_numeric_dtype(train[c])]
    config={"objective":"binary","n_estimators":450,"learning_rate":0.05,"num_leaves":31,"max_depth":-1,"min_child_samples":100,"subsample":0.85,"subsample_freq":1,"colsample_bytree":0.9,"reg_alpha":0.1,"reg_lambda":1.0,"random_state":20260925,"n_jobs":6,"deterministic":True,"force_col_wise":True,"verbosity":-1}
    model=lgb.LGBMClassifier(**config)
    model.fit(train[features],train.label)
    (EXP/"model").mkdir(exist_ok=True); model.booster_.save_model(str(EXP/"model/lightgbm_model.txt"))
    joblib.dump(model,EXP/"model/lightgbm_model.joblib")
    (EXP/"model/model_config.json").write_text(json.dumps({"algorithm":"LightGBM","lightgbm_version":lgb.__version__,"config":config,"feature_count":len(features),"features":features,"training_pair_policy":"all surviving positives plus up to 15 hard negatives per S1 per source (30 total); chosen over the 15-total ablation using fold-4 calibration only","training_rows":len(train),"positives":int(train.label.sum()),"negatives":int((train.label==0).sum())},indent=2))
    imp=pd.DataFrame({"feature":features,"importance_gain":model.booster_.feature_importance("gain"),"importance_split":model.booster_.feature_importance("split")}).sort_values("importance_gain",ascending=False)
    imp.to_csv(EXP/"feature_importance.csv",index=False)
    del train; peak()
    cal=load_role("calibration"); cal["score"]=model.predict_proba(cal[features])[:,1].astype(np.float32)
    cal.to_parquet(EXP/"predictions/calibration_pair_scores.parquet",index=False,compression="zstd")
    sc=selected[selected.role.eq("calibration")].copy()
    allcal=[]; bests={}
    for k in [10,20]:
        grid,best=tune_independent(cal,sc,k);allcal.append(grid);bests[f"E001_K{k}"]=best
    win,grid,best=tune_owner(cal,sc);allcal.append(grid);bests["E002_K20"]=best
    grid,best=tune_margin(win,sc,bests["E002_K20"]["threshold"]);allcal.append(grid);bests["E003_K20"]=best
    pd.concat(allcal,ignore_index=True).to_csv(EXP/"calibration_results.csv",index=False)
    (EXP/"model/frozen_decision_parameters.json").write_text(json.dumps(json_clean(bests),indent=2,allow_nan=False))
    del cal,win; peak()
    ev=load_role("evaluation");ev["score"]=model.predict_proba(ev[features])[:,1].astype(np.float32)
    ev.to_parquet(EXP/"predictions/evaluation_pair_scores.parquet",index=False,compression="zstd")
    se=selected[selected.role.eq("evaluation")].copy()
    oracle=pd.read_csv(EXP/"candidate_stats.csv").query("role=='evaluation'").set_index("candidate_set").oracle_grouped_macro_f05
    eval_rows=[]; configs=[]
    for key in ["E001_K10","E001_K20","E002_K20","E003_K20"]:
        exp,k=key.split("_K");k=int(k);b=bests[key]
        if exp=="E001": mask=(ev.in_k10.eq(1) if k==10 else pd.Series(True,index=ev.index))&ev.score.ge(b["threshold"]); scored=ev
        else:
            scored=ownership_table(ev,k);mask=scored.score.ge(b["threshold"])
            if exp=="E003":mask &= scored.owner_margin.ge(b["margin"])
        met=eval_mask(scored,se,mask.to_numpy())
        eval_rows.append({"experiment":exp,"candidate_set":f"K{k}","threshold":b["threshold"],"margin":b.get("margin",0),**met,"oracle_macro_f05":oracle.loc[f"K{k}"],"oracle_actual_gap":oracle.loc[f"K{k}"]-met["macro_f05"]})
        configs.append((exp,k,b["threshold"],b.get("margin",0)))
        save_predictions(ev,se,exp,k,b["threshold"],b.get("margin",0))
    eres=pd.DataFrame(eval_rows);eres.to_csv(EXP/"evaluation_results.csv",index=False)
    slices=slice_table(ev,se,configs);slices.to_csv(EXP/"slice_results.csv",index=False)
    runtime=json.loads((EXP/"runtime.json").read_text()) if (EXP/"runtime.json").exists() else {}
    runtime.update({"status":"complete","training_and_evaluation_seconds":time.time()-START,"total_elapsed_seconds_this_invocation":time.time()-START,"peak_rss_gb_training_process":PEAK/2**30,"ru_maxrss_gb_training_process":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20})
    (EXP/"runtime.json").write_text(json.dumps(runtime,indent=2))
    print(eres.to_string(index=False));print("top importance\n",imp.head(15).to_string(index=False))

if __name__=="__main__": main()
