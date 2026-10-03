#!/usr/bin/env python3
"""Cache-only K10 decoding, shortlist diagnostics, and compact text-pair export."""
from __future__ import annotations

import hashlib
import json
import math
import resource
import time
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
E001 = ROOT / "analysis/experiments_v1/e001_classical_baseline"
OUT = ROOT / "analysis/experiments_v1/e004_k10_decode_and_gpu_bundle"
DB = ROOT / "analysis/data_audit_v1/artifacts/audit.duckdb"
START = time.time()
BASE_T = 0.780
EXPECTED_CAL = 0.9237187372077086
EXPECTED_EVAL = 0.9247151386297013


def metric(frame, selected, mask):
    ids = selected.source1_entity_id.to_numpy()
    idx = pd.Series(np.arange(len(ids), dtype=np.int32), index=ids)
    gi = idx.loc[frame.s1].to_numpy()
    use = np.asarray(mask, dtype=bool)
    pred = np.bincount(gi[use], minlength=len(ids))
    tp = np.bincount(gi[use], weights=frame.label.to_numpy()[use], minlength=len(ids))
    true = selected.true_matches.to_numpy(dtype=np.float64)
    den = 1.25*tp + .25*(true-tp) + (pred-tp)
    scores = np.divide(1.25*tp, den, out=np.zeros_like(den), where=den > 0)
    scores[(true == 0) & (pred == 0)] = 1.0
    return {"macro_f05": float(scores.mean()), "precision": float(tp.sum()/pred.sum()) if pred.sum() else 1.0,
            "recall": float(tp.sum()/true.sum()) if true.sum() else 1.0,
            "singleton_accuracy": float(np.mean(pred[true == 0] == 0)) if np.any(true == 0) else float("nan"), "predicted_edges": int(pred.sum()),
            "true_edges": int(true.sum()), "tp": int(tp.sum())}


def source_metric(frame, selected, mask, source):
    take = frame.source.eq(source).to_numpy()
    s = selected.copy(); s["true_matches"] = s[f"s{source}_matches"]
    return metric(frame.loc[take], s, np.asarray(mask)[take])


def owners(frame):
    x = frame.sort_values(["target_id", "score", "s1"], ascending=[True, False, True], kind="mergesort").copy()
    x["owner_rank"] = x.groupby("target_id", sort=False).cumcount() + 1
    win = x[x.owner_rank.eq(1)].copy()
    second = x[x.owner_rank.eq(2)].set_index("target_id").score
    win["owner_margin"] = win.score - win.target_id.map(second).fillna(0)
    return win, x


def tune_a(win, selected):
    rows=[]
    coarse=np.unique(np.r_[np.arange(.50,.981,.02), [.99,.995]])
    for t in coarse:
        rows.append({"experiment":"E004A","stage":"coarse","threshold_s2":t,"threshold_s3":t,"margin":0.0,**metric(win,selected,win.score.ge(t))})
    b=max(rows,key=lambda z:z["macro_f05"])
    for t in np.unique(np.clip(np.arange(b["threshold_s2"]-.04,b["threshold_s2"]+.0401,.002),0,1)):
        rows.append({"experiment":"E004A","stage":"fine","threshold_s2":t,"threshold_s3":t,"margin":0.0,**metric(win,selected,win.score.ge(t))})
    return rows,max(rows,key=lambda z:z["macro_f05"])


def tune_b(win, selected, base_t):
    rows=[]
    ts=np.unique(np.clip(np.arange(base_t-.05,base_t+.0501,.01),0,1))
    ms=np.array([0,.005,.01,.02,.03,.05,.075,.10,.15,.20,.30,.40,.50])
    for t in ts:
        for m in ms:
            rows.append({"experiment":"E004B","stage":"coarse","threshold_s2":t,"threshold_s3":t,"margin":m,**metric(win,selected,win.score.ge(t)&win.owner_margin.ge(m))})
    b=max(rows,key=lambda z:z["macro_f05"])
    ts=np.unique(np.clip(np.arange(b["threshold_s2"]-.015,b["threshold_s2"]+.0151,.003),0,1))
    ms=np.unique(np.clip(np.arange(b["margin"]-.03,b["margin"]+.0301,.005),0,1))
    for t in ts:
        for m in ms:
            rows.append({"experiment":"E004B","stage":"fine","threshold_s2":t,"threshold_s3":t,"margin":m,**metric(win,selected,win.score.ge(t)&win.owner_margin.ge(m))})
    return rows,max(rows,key=lambda z:z["macro_f05"])


def tune_c(win, selected, parent):
    rows=[]; base=parent["threshold_s2"]; margin=parent["margin"]
    def mask(t2,t3):
        accept=np.where(win.source.eq(2),win.score.ge(t2),win.score.ge(t3))
        if margin > 0: accept &= win.owner_margin.ge(margin)
        return accept
    coarse=np.unique(np.clip(np.arange(base-.06,base+.0601,.01),0,1))
    for t2 in coarse:
        for t3 in coarse:
            rows.append({"experiment":"E004C","stage":"coarse","parent":parent["experiment"],"threshold_s2":t2,"threshold_s3":t3,"margin":margin,**metric(win,selected,mask(t2,t3))})
    b=max(rows,key=lambda z:z["macro_f05"])
    g2=np.unique(np.clip(np.arange(b["threshold_s2"]-.012,b["threshold_s2"]+.0121,.003),0,1))
    g3=np.unique(np.clip(np.arange(b["threshold_s3"]-.012,b["threshold_s3"]+.0121,.003),0,1))
    for t2 in g2:
        for t3 in g3:
            rows.append({"experiment":"E004C","stage":"fine","parent":parent["experiment"],"threshold_s2":t2,"threshold_s3":t3,"margin":margin,**metric(win,selected,mask(t2,t3))})
    return rows,max(rows,key=lambda z:z["macro_f05"])


def frozen_mask(win, cfg):
    m=np.where(win.source.eq(2),win.score.ge(cfg["threshold_s2"]),win.score.ge(cfg["threshold_s3"]))
    if cfg["margin"] > 0: m &= win.owner_margin.ge(cfg["margin"])
    return np.asarray(m)


def add_slices(win, selected, mask, experiment):
    rows=[]
    for source in [2,3]: rows.append({"experiment":experiment,"slice":"source","value":f"S{source}",**source_metric(win,selected,mask,source)})
    for col in ["country"]:
        for val,g in selected.groupby(col):
            ids=set(g.source1_entity_id); take=win.s1.isin(ids).to_numpy()
            rows.append({"experiment":experiment,"slice":col,"value":str(val),**metric(win.loc[take],g,np.asarray(mask)[take])})
    g=selected[selected.true_matches.eq(1)];ids=set(g.source1_entity_id);take=win.s1.isin(ids).to_numpy()
    rows.append({"experiment":experiment,"slice":"cardinality","value":"1",**metric(win.loc[take],g,np.asarray(mask)[take])})
    fan=win.groupby("s1").s1_normalized_name_fanout.max()
    ambiguous=set(fan[fan.gt(1)].index)
    for value,ids in [("repeated/common",ambiguous),("non_ambiguous",set(selected.source1_entity_id)-ambiguous)]:
        g=selected[selected.source1_entity_id.isin(ids)];take=win.s1.isin(ids).to_numpy()
        rows.append({"experiment":experiment,"slice":"name_ambiguity","value":value,**metric(win.loc[take],g,np.asarray(mask)[take])})
    return rows


def shortlist(frame, selected, role):
    x=frame.sort_values(["s1","score","target_id"],ascending=[True,False,True],kind="mergesort").copy()
    x["lgbm_rank"]=(x.groupby("s1",sort=False).cumcount()+1).astype(np.int16)
    x["lightgbm_rank_within_s1"] = x["lgbm_rank"]
    total=len(x); out=[]
    for label,use in [(f"top{n}",x.lgbm_rank.le(n)) for n in [1,2,3,5,8,10]] + [("top5_plus_score_ge_0.05",x.lgbm_rank.le(5)|x.score.ge(.05))]:
        kept=x[use]; te=int(selected.true_matches.sum()); hit=int(kept.label.sum()); pos=selected[selected.true_matches.gt(0)]
        hits=kept[kept.label.eq(1)].groupby("s1").size().reindex(selected.source1_entity_id,fill_value=0)
        anyr=float((hits.loc[pos.source1_entity_id]>0).mean()); allr=float((hits.loc[pos.source1_entity_id].to_numpy()>=pos.true_matches.to_numpy()).mean())
        oracle=metric(kept,selected,kept.label.eq(1).to_numpy())["macro_f05"]
        out.append({"role":role,"shortlist":label,"candidate_pairs":len(kept),"fraction_k10_pairs":len(kept)/total,"true_edges":te,"true_edges_retained":hit,"true_edge_recall":hit/te,"positive_s1_any_truth_recall":anyr,"positive_s1_all_truth_recall":allr,"oracle_grouped_macro_f05":oracle})
    return pd.DataFrame(out),x


def ambiguity(frame, ranked, selected, baseline_mask, role):
    counts=frame.groupby("target_id").s1.nunique()
    dist={"1":int(counts.eq(1).sum()),"2":int(counts.eq(2).sum()),"3":int(counts.eq(3).sum()),"4+":int(counts.ge(4).sum())}
    b=frame.copy();b["pred"]=np.asarray(baseline_mask)
    fp=b[b.pred & b.label.eq(0)]
    ambiguous=set(counts[counts.gt(1)].index)
    fp_frac=float(fp.target_id.isin(ambiguous).mean()) if len(fp) else 0
    # Official false negatives include blocked-out true edges. Load complete truth locally.
    con=duckdb.connect(str(DB),read_only=True); ids=selected[["source1_entity_id"]];con.register("ids",ids)
    truth=con.execute("select e.s1,e.target_id from edges e join ids i on e.s1=i.source1_entity_id").fetchdf();con.close()
    predset=set(zip(b.loc[b.pred,"s1"],b.loc[b.pred,"target_id"]))
    fn=truth[[((s,t) not in predset) for s,t in zip(truth.s1,truth.target_id)]]
    fn_frac=float(fn.target_id.isin(ambiguous).mean()) if len(fn) else 0
    # Correct owner rank; absent true candidates fail, as appropriate for end-to-end recall.
    rr=ranked[ranked.label.eq(1)][["s1","target_id","owner_rank"]]
    rankmap={(s,t):r for s,t,r in rr.itertuples(index=False,name=None)}
    ranks=np.array([rankmap.get((s,t),999) for s,t in zip(truth.s1,truth.target_id)])
    return {"role":role,"distinct_auxiliary_candidates":len(counts),"owner_count_distribution":dist,
            "e001_false_positives":len(fp),"fraction_false_positives_ambiguous_owner":fp_frac,
            "e001_false_negatives":len(fn),"fraction_false_negatives_ambiguous_owner":fn_frac,
            "true_owner_top1_recall":float(np.mean(ranks<=1)),"true_owner_top2_recall":float(np.mean(ranks<=2)),"true_owner_top3_recall":float(np.mean(ranks<=3)),
            "true_edges_denominator":len(truth),"true_edges_present_anywhere_k10":int(np.sum(ranks<999))}


def rank_pairs(x):
    x=x.sort_values(["s1","score","target_id"],ascending=[True,False,True],kind="mergesort").copy()
    x["lightgbm_rank_within_s1"]=(x.groupby("s1",sort=False).cumcount()+1).astype(np.int16)
    return x


def join_text_and_write(pairs, role, path):
    cols=["s1","target_id","source","country","role","label","score","lightgbm_rank_within_s1","name_retrieval_score","name_rank","address_retrieval_score","address_rank"]
    pairs=pairs[cols].copy()
    con=duckdb.connect(str(DB),read_only=True);con.execute("PRAGMA threads=6");con.execute("PRAGMA memory_limit='3GB'");con.register("pairs",pairs)
    out=con.execute("""select p.s1 source1_entity_id,p.target_id candidate_entity_id,
       case when p.source=2 then 'S2' else 'S3' end candidate_source,p.country,p.role,p.label,
       s.business_name s1_business_name,coalesce(s.business_address,'') s1_business_address,
       a.business_name candidate_business_name,coalesce(a.business_address,'') candidate_business_address,
       p.score lightgbm_score,p.lightgbm_rank_within_s1,p.name_retrieval_score,p.name_rank,
       p.address_retrieval_score,p.address_rank
       from pairs p join train_s1 s on p.s1=s.entity_id
       join (select 2 AS src,* from train_s2 union all by name select 3 AS src,* from train_s3) a
         on p.source=a.src and p.target_id=a.entity_id
       order by p.s1,p.lightgbm_rank_within_s1,p.target_id""").to_arrow_table()
    con.close();pq.write_table(out,path,compression="zstd",compression_level=6)


def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(2**20),b""):h.update(chunk)
    return h.hexdigest()


def file_manifest(path):
    table=pq.read_table(path); d=table.to_pandas()
    return {"path":str(path.relative_to(OUT)),"rows":len(d),"positives":int(d.label.sum()),"negatives":int((d.label==0).sum()),
            "unique_s1":int(d.source1_entity_id.nunique()),"source_counts":d.candidate_source.value_counts().sort_index().to_dict(),
            "country_counts":d.country.value_counts().sort_index().to_dict(),"size_bytes":path.stat().st_size,
            "sha256":sha256(path),"columns_dtypes":{c:str(t) for c,t in d.dtypes.items()}}


def export_bundle(cal_ranked,eval_ranked,recommendation):
    bundle=OUT/"gpu_bundle";bundle.mkdir(parents=True,exist_ok=True)
    model=joblib.load(E001/"model/lightgbm_model.joblib")
    features=json.loads((E001/"model/model_config.json").read_text())["features"]
    chunks=[]
    for p in sorted((E001/"cache/features").glob("train_*.parquet")):
        x=pd.read_parquet(p);x=x[x.in_k10.eq(1)].copy();x["score"]=model.predict_proba(x[features])[:,1].astype(np.float32)
        chunks.append(x[["s1","target_id","source","country","role","label","score","name_retrieval_score","name_rank","address_retrieval_score","address_rank"]])
    train=rank_pairs(pd.concat(chunks,ignore_index=True))
    pos=train[train.label.eq(1)]; neg=train[train.label.eq(0)].copy(); within=neg.groupby(["s1","source"],sort=False).cumcount()
    neg=neg[within.lt(5)];train_export=pd.concat([pos,neg],ignore_index=True)
    train_export=rank_pairs(train_export)
    for frame,role,name in [(train_export,"train","train_pairs.parquet"),(cal_ranked,"calibration","calibration_pairs.parquet"),(eval_ranked,"evaluation","evaluation_pairs.parquet")]:
        join_text_and_write(frame,role,bundle/name)
    manifest={"created_by":"E004-lite cache-only export","recommended_neural_shortlist":recommendation,
      "provenance":{"selected_ids":"analysis/experiments_v1/e001_classical_baseline/selected_ids/selected_s1.parquet",
       "calibration_scores":"analysis/experiments_v1/e001_classical_baseline/predictions/calibration_pair_scores.parquet",
       "evaluation_scores":"analysis/experiments_v1/e001_classical_baseline/predictions/evaluation_pair_scores.parquet",
       "train_features":"analysis/experiments_v1/e001_classical_baseline/cache/features/train_*.parquet",
       "model":"analysis/experiments_v1/e001_classical_baseline/model/lightgbm_model.joblib",
       "record_text":"analysis/data_audit_v1/artifacts/audit.duckdb (local supplied challenge data only)",
       "train_negative_note":"Scores/ranks are over the cached E001 hard-pair pool filtered to K10; all surviving positives plus up to five highest-score negatives per source/S1 are exported."},
      "files":{}}
    for name in ["train_pairs.parquet","calibration_pairs.parquet","evaluation_pairs.parquet"]:
        manifest["files"][name]=file_manifest(bundle/name)
    (bundle/"bundle_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    selected=pd.read_parquet(E001/"selected_ids/selected_s1.parquet")
    sc=selected[selected.role.eq("calibration")].copy()
    cal=pd.read_parquet(E001/"predictions/calibration_pair_scores.parquet");cal=cal[cal.in_k10.eq(1)].copy()
    base_cal=metric(cal,sc,cal.score.ge(BASE_T))
    if not math.isclose(base_cal["macro_f05"],EXPECTED_CAL,abs_tol=1e-12): raise RuntimeError(f"Calibration baseline mismatch: {base_cal['macro_f05']} != {EXPECTED_CAL}")
    win_cal,ranked_owner_cal=owners(cal)
    rows_a,best_a=tune_a(win_cal,sc);rows_b,best_b=tune_b(win_cal,sc,best_a["threshold_s2"])
    parent=best_a if best_a["macro_f05"]>=best_b["macro_f05"] else best_b
    rows_c,best_c=tune_c(win_cal,sc,parent)
    pd.DataFrame(rows_a+rows_b+rows_c).to_csv(OUT/"calibration_search.csv",index=False)
    frozen={"E004A":best_a,"E004B":best_b,"E004C":best_c,"E004C_parent":parent["experiment"]}
    (OUT/"frozen_parameters.json").write_text(json.dumps(frozen,indent=2)+"\n")
    # Fold 0 is loaded only after the parameters above are frozen.
    se=selected[selected.role.eq("evaluation")].copy()
    ev=pd.read_parquet(E001/"predictions/evaluation_pair_scores.parquet");ev=ev[ev.in_k10.eq(1)].copy()
    base_eval=metric(ev,se,ev.score.ge(BASE_T))
    if not math.isclose(base_eval["macro_f05"],EXPECTED_EVAL,abs_tol=1e-12): raise RuntimeError(f"Evaluation baseline mismatch: {base_eval['macro_f05']} != {EXPECTED_EVAL}")
    win_ev,ranked_owner_ev=owners(ev)
    eval_rows=[];slice_rows=[]
    for name,cfg in [("E004A",best_a),("E004B",best_b),("E004C",best_c)]:
        m=frozen_mask(win_ev,cfg);res=metric(win_ev,se,m);eval_rows.append({"experiment":name,**res,"delta_vs_e001_k10":res["macro_f05"]-EXPECTED_EVAL,"threshold_s2":cfg["threshold_s2"],"threshold_s3":cfg["threshold_s3"],"margin":cfg["margin"]})
        slice_rows.extend(add_slices(win_ev,se,m,name))
    pd.DataFrame(eval_rows).to_csv(OUT/"evaluation_results.csv",index=False);pd.DataFrame(slice_rows).to_csv(OUT/"slice_results.csv",index=False)
    short_cal,cal_ranked=shortlist(cal,sc,"calibration");short_eval,eval_ranked=shortlist(ev,se,"evaluation")
    shorts=pd.concat([short_cal,short_eval],ignore_index=True);shorts.to_csv(OUT/"shortlist_results.csv",index=False)
    # Recommend smallest plain top-N within 0.002 oracle of full K10 on calibration.
    full_oracle=0.9877138377637186
    plain=short_cal[short_cal.shortlist.str.fullmatch(r"top\d+")].copy();plain["n"]=plain.shortlist.str[3:].astype(int)
    eligible=plain[plain.oracle_grouped_macro_f05.ge(full_oracle-.002)]
    rec=(eligible.sort_values("n").iloc[0].shortlist if len(eligible) else plain.sort_values("oracle_grouped_macro_f05",ascending=False).iloc[0].shortlist)
    plain_rec=short_cal[short_cal.shortlist.eq(rec)].iloc[0]
    hybrid=short_cal[short_cal.shortlist.eq("top5_plus_score_ge_0.05")].iloc[0]
    if hybrid.candidate_pairs < plain_rec.candidate_pairs and hybrid.oracle_grouped_macro_f05 >= plain_rec.oracle_grouped_macro_f05 - .0002:
        rec=hybrid.shortlist
    amb=[ambiguity(cal,ranked_owner_cal,sc,cal.score.ge(BASE_T),"calibration"),ambiguity(ev,ranked_owner_ev,se,ev.score.ge(BASE_T),"evaluation")]
    (OUT/"owner_ambiguity.json").write_text(json.dumps(amb,indent=2)+"\n")
    manifest=export_bundle(cal_ranked,eval_ranked,rec)
    runtime={"elapsed_seconds":time.time()-START,"peak_rss_gb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,"baseline_calibration":base_cal,"baseline_evaluation":base_eval,"recommended_shortlist":rec}
    (OUT/"runtime.json").write_text(json.dumps(runtime,indent=2)+"\n")
    print("BASE",base_cal["macro_f05"],base_eval["macro_f05"])
    print(pd.DataFrame(eval_rows).to_string(index=False));print(shorts.to_string(index=False));print(json.dumps(amb,indent=2));print(json.dumps(manifest["files"],indent=2));print(json.dumps(runtime,indent=2))


if __name__=="__main__": main()
