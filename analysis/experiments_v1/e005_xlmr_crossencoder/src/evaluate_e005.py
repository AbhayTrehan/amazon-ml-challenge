#!/usr/bin/env python3
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

from metric import grouped_f05, self_test


def score_column(frame, experiment, weight=None):
    if experiment == "E005A": return frame.neural_probability.to_numpy(dtype=float)
    return weight * frame.neural_probability.to_numpy(dtype=float) + (1-weight) * frame.lightgbm_score.to_numpy(dtype=float)


def threshold_grid(scores):
    base=np.r_[np.arange(.05,.951,.05), [.01,.02,.98,.99,.995]]
    return np.unique(np.clip(base,0,1))


def mask_for(frame, scores, cfg):
    if cfg["decoder"] == "independent":
        return scores >= cfg["threshold"]
    x=frame[["candidate_entity_id","source1_entity_id"]].copy();x["_score"]=scores;x["_pos"]=np.arange(len(x))
    x=x.sort_values(["candidate_entity_id","_score","source1_entity_id"],ascending=[True,False,True],kind="mergesort")
    x["_rank"]=x.groupby("candidate_entity_id",sort=False).cumcount()
    win=x[x._rank.eq(0)].copy();second=x[x._rank.eq(1)].set_index("candidate_entity_id")._score
    win["_margin"]=win._score-win.candidate_entity_id.map(second).fillna(0)
    accepted=win[(win._score>=cfg["threshold"])&(win._margin>=cfg.get("margin",0))]._pos.to_numpy()
    out=np.zeros(len(frame),bool);out[accepted]=True;return out


def search_policy(frame, selected, experiment, weight=None):
    scores=score_column(frame,experiment,weight); rows=[]
    def add(decoder,t,m,stage):
        cfg={"decoder":decoder,"threshold":float(t),"margin":float(m)}
        met=grouped_f05(frame,selected,mask_for(frame,scores,cfg))
        rows.append({"experiment":experiment,"weight":weight,"stage":stage,**cfg,**met})
    for t in threshold_grid(scores): add("independent",t,0,"coarse")
    b=max((r for r in rows if r["decoder"]=="independent"),key=lambda r:r["macro_f05"])
    for t in np.unique(np.clip(np.arange(b["threshold"]-.04,b["threshold"]+.0401,.005),0,1)): add("independent",t,0,"fine")
    # Compact owner search around the independent optimum, followed by a narrow refinement.
    center=max(rows,key=lambda r:r["macro_f05"])["threshold"]
    for t in np.unique(np.clip(np.arange(center-.06,center+.0601,.02),0,1)):
        for m in [0,.01,.025,.05,.10,.20,.30,.40,.50]: add("one_owner_margin",t,m,"owner_coarse")
    b=max((r for r in rows if r["decoder"]=="one_owner_margin"),key=lambda r:r["macro_f05"])
    for t in np.unique(np.clip(np.arange(b["threshold"]-.02,b["threshold"]+.0201,.005),0,1)):
        for m in np.unique(np.clip(np.arange(b["margin"]-.04,b["margin"]+.0401,.01),0,1)): add("one_owner_margin",t,m,"owner_fine")
    return rows,max(rows,key=lambda r:r["macro_f05"])


def calibration(args):
    d=pd.read_parquet(args.predictions);selected=pd.read_parquet(args.metadata).query("role == 'calibration'").copy()
    rows=[];best={}
    r,b=search_policy(d,selected,"E005A");rows+=r;best["E005A"]=b
    if not args.e005a_only:
        for w in [.25,.50,.75]:
            r,bw=search_policy(d,selected,"E005B",w);rows+=r
        allb=[r for r in rows if r["experiment"]=="E005B"]
        coarse_best=max(allb,key=lambda r:r["macro_f05"])
        # At most one small refinement around the best coarse fusion weight.
        refine=[w for w in [coarse_best["weight"]-.10,coarse_best["weight"]+.10] if 0 < w < 1]
        for w in refine:
            r,bw=search_policy(d,selected,"E005B",float(w));rows+=r
        best["E005B"]=max((r for r in rows if r["experiment"]=="E005B"),key=lambda r:r["macro_f05"])
    for cfg in best.values():
        for k in ["macro_f05","precision","recall","singleton_accuracy","predicted_edges","true_edges","tp"]: cfg[k]=float(cfg[k]) if k not in ["predicted_edges","true_edges","tp"] else int(cfg[k])
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/f"calibration_search_epoch{args.epoch}.csv",index=False)
    summary={"epoch":args.epoch,"shortlist_rows":len(d),"best":best}
    (out/f"calibration_summary_epoch{args.epoch}.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


def source_metric(frame,selected,mask,source):
    take=frame.candidate_source.eq(source).to_numpy();return grouped_f05(frame.loc[take],selected,np.asarray(mask)[take],true_col=f"s{source[-1]}_matches")


def slice_rows(frame,complete,selected,mask,experiment):
    rows=[]
    def add(sl,val,ids):
        ss=selected[selected.source1_entity_id.isin(ids)];take=frame.source1_entity_id.isin(ids).to_numpy()
        rows.append({"experiment":experiment,"slice":sl,"value":str(val),"n_s1":len(ss),**grouped_f05(frame.loc[take],ss,np.asarray(mask)[take])})
    for src in ["S2","S3"]: rows.append({"experiment":experiment,"slice":"source","value":src,"n_s1":len(selected),**source_metric(frame,selected,mask,src)})
    for val,g in selected.groupby("country"): add("country",val,set(g.source1_entity_id))
    for val,g in selected.groupby("cardinality_bucket"): add("cardinality",val,set(g.source1_entity_id))
    for val,g in selected.groupby("repeated_name"): add("name_ambiguity","repeated/common" if val else "non_ambiguous",set(g.source1_entity_id))
    missing=set(complete[(complete.label.eq(1))&complete.candidate_business_address.fillna("").str.strip().eq("")].source1_entity_id)
    add("missing_true_candidate_address","true",missing);add("missing_true_candidate_address","false",set(selected.source1_entity_id)-missing)
    return rows


def e004b_mask(d):
    x=d[["candidate_entity_id","source1_entity_id","lightgbm_score"]].copy();x["_pos"]=np.arange(len(x))
    x=x.sort_values(["candidate_entity_id","lightgbm_score","source1_entity_id"],ascending=[True,False,True],kind="mergesort")
    x["rank"]=x.groupby("candidate_entity_id",sort=False).cumcount();win=x[x["rank"].eq(0)].copy()
    second=x[x["rank"].eq(1)].set_index("candidate_entity_id").lightgbm_score
    win["margin"]=win.lightgbm_score-win.candidate_entity_id.map(second).fillna(0)
    pos=win[win.lightgbm_score.ge(.780)&win.margin.ge(.485)]._pos.to_numpy();mask=np.zeros(len(d),bool);mask[pos]=True;return mask


def final_eval(args):
    d=pd.read_parquet(args.predictions);complete=pd.read_parquet(args.complete_input)
    selected=pd.read_parquet(args.metadata).query("role == 'evaluation'").copy()
    frozen=json.loads(Path(args.frozen).read_text());results=[];slices=[];masks={}
    for exp in ["E005A","E005B"]:
        cfg=frozen[exp];scores=score_column(d,exp,cfg.get("weight"));mask=mask_for(d,scores,cfg);masks[exp]=mask
        met=grouped_f05(d,selected,mask);results.append({"experiment":exp,**met,"shortlist_oracle":.986696,
            "oracle_model_gap":.986696-met["macro_f05"],"delta_vs_e004b":met["macro_f05"]-.928336,
            "decoder":cfg["decoder"],"threshold":cfg["threshold"],"margin":cfg.get("margin",0),"fusion_weight":cfg.get("weight")})
        slices += slice_rows(d,complete,selected,mask,exp)
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(results).to_csv(out/"evaluation_results.csv",index=False);pd.DataFrame(slices).to_csv(out/"slice_results.csv",index=False)
    pred=d.copy()
    for exp in ["E005A","E005B"]:
        pred[f"{exp.lower()}_accepted"]=masks[exp]
        pred[f"{exp.lower()}_score"]=score_column(d,exp,frozen[exp].get("weight"))
    pred.to_parquet(out/"evaluation_predictions.parquet",index=False,compression="zstd")

    best=max(results,key=lambda r:r["macro_f05"])["experiment"]; neural_keys=set(zip(d.loc[masks[best],"source1_entity_id"],d.loc[masks[best],"candidate_entity_id"]))
    bm=e004b_mask(complete);lgb_keys=set(zip(complete.loc[bm,"source1_entity_id"],complete.loc[bm,"candidate_entity_id"]))
    x=complete.copy();keys=list(zip(x.source1_entity_id,x.candidate_entity_id));x["lgbm_pred"]=[k in lgb_keys for k in keys];x["neural_pred"]=[k in neural_keys for k in keys]
    x["event"]=""
    x.loc[x.lgbm_pred&~x.neural_pred&x.label.eq(0),"event"]="neural_fixes_lgbm_false_positive"
    x.loc[~x.lgbm_pred&x.neural_pred&x.label.eq(0),"event"]="neural_introduces_new_false_positive"
    x.loc[~x.lgbm_pred&x.neural_pred&x.label.eq(1),"event"]="neural_recovers_lgbm_false_negative"
    x.loc[x.lgbm_pred&~x.neural_pred&x.label.eq(1),"event"]="neural_loses_lgbm_true_positive"
    x=x[x.event.ne("")].copy();meta=selected.set_index("source1_entity_id")
    x["repeated_name"]=x.source1_entity_id.map(meta.repeated_name);x["cardinality_1"]=x.source1_entity_id.map(meta.true_matches).eq(1)
    neural_map=d.set_index(["source1_entity_id","candidate_entity_id"]).neural_probability
    x["neural_probability"]=[neural_map.get(k,np.nan) for k in zip(x.source1_entity_id,x.candidate_entity_id)]
    count_rows=[]
    groups=[("overall",pd.Series(True,index=x.index)),("India",x.country.eq("India")),("repeated_name",x.repeated_name),("cardinality_1",x.cardinality_1)]
    for sl,take in groups:
        for event,n in x.loc[take].event.value_counts().items(): count_rows.append({"best_experiment":best,"slice":sl,"event":event,"count":int(n)})
    pd.DataFrame(count_rows).to_csv(out/"disagreement_counts.csv",index=False)
    cols=["event","source1_entity_id","candidate_entity_id","candidate_source","country","label","s1_business_name","s1_business_address","candidate_business_name","candidate_business_address","lightgbm_score","neural_probability","repeated_name","cardinality_1"]
    examples=x.sort_values(["event","source1_entity_id","candidate_entity_id"]).groupby("event",group_keys=False).head(25)
    examples[cols].to_csv(out/"disagreement_examples.tsv",sep="\t",index=False)
    print(pd.DataFrame(results).to_string(index=False));print(pd.DataFrame(count_rows).to_string(index=False))


def main():
    self_test();ap=argparse.ArgumentParser();ap.add_argument("mode",choices=["calibration","final"])
    ap.add_argument("--predictions",required=True);ap.add_argument("--metadata",required=True);ap.add_argument("--output-dir",required=True)
    ap.add_argument("--epoch",type=int);ap.add_argument("--frozen");ap.add_argument("--complete-input")
    ap.add_argument("--e005a-only", action="store_true")
    args=ap.parse_args();calibration(args) if args.mode=="calibration" else final_eval(args)


if __name__=="__main__":main()
