#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from common import *
from calibrate import pair_matrix, apply_owner


def selected_validation():
    ids=pd.read_parquet(EXP/"data/fold4_s1.parquet").query("not is_calibration")[["source1_entity_id","country","fold"]]
    con=duckdb.connect(str(DB),read_only=True);con.register("ids",ids)
    d=con.execute("""select i.*,count(e.target_id)::integer true_matches,count(e.target_id)filter(where e.target_source=2)::integer s2_matches,
      count(e.target_id)filter(where e.target_source=3)::integer s3_matches from ids i left join edges e on e.s1=i.source1_entity_id group by all""").fetchdf();con.close();return d


def reveal(frame):
    x=frame.drop(columns=["label"],errors="ignore").copy();x["_rid"]=np.arange(len(x),dtype=np.int64)
    con=duckdb.connect(str(DB),read_only=True);con.register("pairs",x)
    out=con.execute("""select p.* exclude(_rid),(e.target_id is not null)::tinyint pair_label from pairs p left join edges e
      on e.s1=p.source1_entity_id and e.target_id=p.candidate_entity_id and e.target_source=p.source order by p._rid""").fetchdf();con.close();return out.rename(columns={"pair_label":"label"})


def p002_decode(d):
    d=d.copy();d["pair_probability"]=(.6*d.historical_neural+.4*d.historical_lgbm).astype(np.float32)
    z=d.sort_values(["candidate_entity_id","pair_probability","source1_entity_id"],ascending=[True,False,True],kind="mergesort");z["rank"]=z.groupby("candidate_entity_id",sort=False).cumcount();sec=z[z["rank"].eq(1)].set_index("candidate_entity_id").pair_probability;w=z[z["rank"].eq(0)].copy();w["margin"]=w.pair_probability-w.candidate_entity_id.map(sec).fillna(0);return w,(w.pair_probability.ge(.51)&w.margin.ge(.48)).to_numpy()


def oracle_metrics(selected,shortlist):
    # Fold-4 feature shards also contain the historical 10k calibration S1s.
    # The frozen validation oracle must be computed on the remainder only.
    keep=shortlist.source1_entity_id.isin(selected.source1_entity_id).to_numpy()
    frame=shortlist.loc[keep].reset_index(drop=True)
    return grouped_metrics(frame,selected,frame.label.eq(1).to_numpy())


def slice_rows(name,frame,mask,selected):
    rows=[]
    for country,g in selected.groupby("country"):
        ids=set(g.source1_entity_id);take=frame.source1_entity_id.isin(ids).to_numpy();rows.append({"model":name,"slice":"country","value":country,"n_s1":len(g),**grouped_metrics(frame.loc[take],g,np.asarray(mask)[take])})
    for source,col in [(2,"s2_matches"),(3,"s3_matches")]:
        sub=frame[frame.source.eq(source)].copy();sm=np.asarray(mask)[frame.source.eq(source).to_numpy()];ss=selected.copy();ss["true_matches"]=ss[col];rows.append({"model":name,"slice":"source","value":f"S{source}","n_s1":len(ss),**grouped_metrics(sub,ss,sm)})
    for label,cond in [("0",selected.true_matches.eq(0)),("1",selected.true_matches.eq(1)),("2",selected.true_matches.eq(2)),("3",selected.true_matches.eq(3)),("4",selected.true_matches.eq(4)),("5+",selected.true_matches.ge(5))]:
        ss=selected[cond];ids=set(ss.source1_entity_id);take=frame.source1_entity_id.isin(ids).to_numpy();rows.append({"model":name,"slice":"cardinality","value":label,"n_s1":len(ss),**grouped_metrics(frame.loc[take],ss,np.asarray(mask)[take])})
    return rows


def run():
    freeze=EXP/"freeze/frozen_config.json"
    if not freeze.exists():raise RuntimeError("validation access refused: freeze manifest absent")
    selected=selected_validation();base=pd.read_parquet(EXP/"cache/shortlist/validation.parquet");base["xlmr_probability"]=pd.read_parquet(EXP/"cache/neural/validation_selected.parquet").xlmr_probability.to_numpy();base=reveal(base)
    stack=joblib.load(EXP/"calibration/pair_stacker.joblib");owner=joblib.load(EXP/"calibration/owner_calibrator.joblib");base["pair_probability"]=stack.predict_proba(pair_matrix(base))[:,1]
    win=apply_owner(base,owner);emask=expected_utility_decode(win);e008=grouped_metrics(win,selected,emask)
    p2=pd.read_parquet(EXP/"cache/shortlist/p002_validation.parquet");p2["historical_neural"]=pd.read_parquet(EXP/"cache/neural/p002_validation.parquet").historical_neural.to_numpy();p2=reveal(p2);pwin,pmask=p002_decode(p2);p002=grouped_metrics(pwin,selected,pmask)
    rows=slice_rows("P002",pwin,pmask,selected)+slice_rows("E008",win,emask,selected)
    # Repeated/common-name and missing-true-address S1 definitions use frozen existing signals.
    glob=str(EXP/"cache/features/fold4/*/s*/part-*.parquet");ids=selected[["source1_entity_id"]]
    con=duckdb.connect(str(DB),read_only=True);con.register("ids",ids)
    rep=con.execute(f"select f.s1 source1_entity_id,max(f.s1_normalized_name_fanout)>1 repeated_name from read_parquet('{glob}') f join ids i on f.s1=i.source1_entity_id group by 1").fetchdf()
    miss=con.execute(f"""select distinct f.s1 source1_entity_id from read_parquet('{glob}') f join ids i on f.s1=i.source1_entity_id join edges e on e.s1=f.s1 and e.target_id=f.target_id and e.target_source=f.source where f.candidate_address_missing=1""").fetchdf();con.close()
    repeated=set(rep[rep.repeated_name].source1_entity_id);missing=set(miss.source1_entity_id)
    for name,frame,mask in [("P002",pwin,pmask),("E008",win,emask)]:
      for sl,val,idset in [("name_ambiguity","repeated/common",repeated),("name_ambiguity","non_ambiguous",set(selected.source1_entity_id)-repeated),("missing_true_candidate_address","true",missing),("missing_true_candidate_address","false",set(selected.source1_entity_id)-missing)]:
        ss=selected[selected.source1_entity_id.isin(idset)];take=frame.source1_entity_id.isin(idset).to_numpy();rows.append({"model":name,"slice":sl,"value":val,"n_s1":len(ss),**grouped_metrics(frame.loc[take],ss,np.asarray(mask)[take])})
    slices=pd.DataFrame(rows);slices.to_csv(EXP/"validation/slices.csv",index=False)
    india={m:float(slices[(slices.model.eq(m))&slices.slice.eq("country")&slices.value.eq("India")].macro_f05.iloc[0]) for m in ["P002","E008"]}
    block=oracle_metrics(selected,reveal(pd.concat([pd.read_parquet(p,columns=["s1","target_id","source"]) for p in feature_files_fold4()],ignore_index=True).rename(columns={"s1":"source1_entity_id","target_id":"candidate_entity_id"})))
    short=oracle_metrics(selected,base)
    pairdiag={"brier":brier_score_loss(base.label,base.pair_probability),"nll":log_loss(base.label,base.pair_probability,labels=[0,1])}
    kmap=win.loc[emask].groupby("source1_entity_id").size();selected_k=selected.source1_entity_id.map(kmap).fillna(0).astype(int)
    behavior={"predicted_matches_per_s1":float(emask.sum()/len(selected)),"true_matches_per_s1":float(selected.true_matches.mean()),"owner_rejections_per_s1":float((len(base)-len(win))/len(selected)),"contested_top_owners":int(win.contested.sum()),"selected_k_distribution":selected_k.value_counts().sort_index().to_dict()}
    gate={"delta_macro_f05":e008["macro_f05"]-p002["macro_f05"],"precision":e008["precision"],"india_delta":india["E008"]-india["P002"]}
    gate["pass"]=bool(gate["delta_macro_f05"]>=.001 and gate["precision"]>=.995 and gate["india_delta"]>=-.001);gate["reason"]="all predefined conditions satisfied" if gate["pass"] else "one or more predefined conditions failed"
    report={"P002":p002,"E008":e008,"blocker_oracle":block,"shortlist_oracle":short,"pair_calibration":pairdiag,"decoder_behavior":behavior,"gate":gate}
    atomic_json(EXP/"validation/results.json",report);atomic_json(EXP/"validation/production_gate.json",gate);print(json.dumps(report,indent=2))


def feature_files_fold4():return sorted((EXP/"cache/features/fold4").glob("*/*/part-*.parquet"))
if __name__=="__main__":run()
