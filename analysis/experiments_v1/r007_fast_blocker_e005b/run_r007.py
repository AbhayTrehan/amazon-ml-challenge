#!/usr/bin/env python3
"""Frozen E005B evaluation with only the R006B blocker changed."""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
E001 = ROOT / "analysis/experiments_v1/e001_classical_baseline"
E005 = ROOT / "analysis/experiments_v1/e005_xlmr_crossencoder"
R006 = ROOT / "analysis/experiments_v1/r006a_selective_postings"
DB = ROOT / "analysis/data_audit_v1/artifacts/audit.duckdb"
TEST_S1 = 1_732_544

sys.path.insert(0, str(E001 / "scripts")); import prepare_baseline as e001  # noqa: E402
sys.path.insert(0, str(R006)); import run_r006a as r006a; import run_r006b as r006b  # noqa: E402
sys.path.insert(0, str(E005 / "src")); import common as e005_common  # noqa: E402
from evaluate_e005 import mask_for, score_column, slice_rows  # noqa: E402
from metric import grouped_f05, self_test as metric_self_test  # noqa: E402


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(2**20),b""):h.update(chunk)
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,default=float)+"\n");tmp.replace(path)


def full_query(selected_country):
    con=duckdb.connect(str(DB),read_only=True);con.register("qids",selected_country[["source1_entity_id"]])
    q=con.execute("""select s.entity_id,s.business_name,coalesce(s.business_address,'') business_address,
      s.nname,coalesce(s.naddr,'') naddr from train_s1 s join qids q on s.entity_id=q.source1_entity_id
      order by s.entity_id""").fetchdf();con.close();return q


def full_aux(country,source):
    assignment=E001/"cache/auxiliary_fold_assignment.parquet";con=duckdb.connect(str(DB),read_only=True)
    aux=con.execute(f"""select a.entity_id,a.business_name,coalesce(a.business_address,'') business_address,
      a.nname,coalesce(a.naddr,'') naddr from train_s{source} a join read_parquet(?) f
      on f.entity_id=a.entity_id and f.source={source} where a.country=? and f.fold=0 order by a.entity_id""",
      [str(assignment),country]).fetchdf();con.close();return aux


def stream_frame(qids,inds,scores,channel):
    rr,kk=np.where(inds>=0)
    return pd.DataFrame({"qpos":rr.astype(np.int32),"apos":inds[rr,kk].astype(np.int32),
      f"{channel}_rank":(kk+1).astype(np.int16),f"{channel}_retrieval_score":scores[rr,kk].astype(np.float32)})


def generate_features():
    metric_self_test();selected=r006a.cohort(10000);truth=r006a.truth_for(selected)
    main_fn,_=r006b.load_functions();model=joblib.load(E001/"model/lightgbm_model.joblib")
    model_cfg=json.loads((E001/"model/model_config.json").read_text());features=model_cfg["features"]
    candidate_chunks=[];feature_chunks=[];retrieval_seconds=feature_seconds=score_seconds=0.0
    for country,sc0 in selected.groupby("country",sort=False):
        sc=sc0.sort_values("source1_entity_id",kind="mergesort");q=full_query(sc)
        if q.entity_id.tolist()!=sc.source1_entity_id.tolist():raise RuntimeError("query cohort/order mismatch")
        for source in [2,3]:
            cached_aux,indexes=r006a.build_or_load(country,source);aux=full_aux(country,source)
            if cached_aux.entity_id.tolist()!=aux.entity_id.tolist():raise RuntimeError("R006B index row identity mismatch")
            streams={};qmats={}
            for channel in ["name","address"]:
                xa,idf=indexes[channel];xq=r006a.qmatrix(q,channel,idf);qmats[channel]=xq
                t=time.perf_counter();oi,ov,*_=r006b.selective(main_fn,xq,xa,800,20);retrieval_seconds+=time.perf_counter()-t
                streams[channel]=stream_frame(q.entity_id.to_numpy(),oi,ov,channel)
            feature_start=time.perf_counter()
            cand=streams["name"].merge(streams["address"],on=["qpos","apos"],how="outer")
            cand["s1"]=q.entity_id.to_numpy()[cand.qpos];cand["target_id"]=aux.entity_id.to_numpy()[cand.apos]
            cand["source"]=np.int8(source);cand["country"]=country;cand["role"]="evaluation"
            for channel in ["name","address"]:
                xa,_=indexes[channel];cand[f"{channel}_char_tfidf"]=e001.row_cosines(qmats[channel],xa,cand.qpos.to_numpy(),cand.apos.to_numpy())
            nf=aux.nname.value_counts(dropna=False);af=aux.naddr.value_counts(dropna=False)
            cand["s1_normalized_name_fanout"]=q.nname.map(nf).fillna(0).clip(0,65535).astype(np.uint16).to_numpy()[cand.qpos]
            cand["s1_normalized_address_fanout"]=q.naddr.map(af).fillna(0).clip(0,65535).astype(np.uint16).to_numpy()[cand.qpos]
            cand["candidate_normalized_name_fanout"]=aux.nname.map(nf).fillna(0).clip(0,65535).astype(np.uint16).to_numpy()[cand.apos]
            cand["candidate_normalized_address_fanout"]=aux.naddr.map(af).fillna(0).clip(0,65535).astype(np.uint16).to_numpy()[cand.apos]
            qn=q.nname.to_numpy()[cand.qpos];qa=q.naddr.to_numpy()[cand.qpos];an=aux.nname.to_numpy()[cand.apos];aa=aux.naddr.to_numpy()[cand.apos]
            cand["nname_exact"] = ((qn!="")&(qn==an)).astype(np.int8);cand["naddr_exact"] = ((qa!="")&(qa==aa)).astype(np.int8)
            tset=set(map(tuple,truth[truth.source.eq(source)][["s1","target_id"]].itertuples(index=False,name=None)))
            cand["label"]=np.fromiter(((s,t) in tset for s,t in zip(cand.s1,cand.target_id)),np.int8,len(cand))
            cand=e001.add_context(cand)
            feat=e001.build_features(cand.copy(),q,aux);feature_seconds+=time.perf_counter()-feature_start
            t=time.perf_counter();feat["score"]=model.predict_proba(feat[features])[:,1].astype(np.float32);score_seconds+=time.perf_counter()-t
            candidate_chunks.append(cand);feature_chunks.append(feat)
    candidates=pd.concat(candidate_chunks,ignore_index=True);scored=pd.concat(feature_chunks,ignore_index=True)
    scored=scored.sort_values(["s1","score","target_id"],ascending=[True,False,True],kind="mergesort").copy()
    scored["lightgbm_rank_within_s1"]=(scored.groupby("s1",sort=False).cumcount()+1).astype(np.int16)
    text_cols=candidates[["s1","target_id","source","qpos","apos"]].copy()
    # Join raw text by immutable entity IDs through the canonical local database.
    core=scored[["s1","target_id","source","country","role","label","score","lightgbm_rank_within_s1",
      "name_retrieval_score","name_rank","address_retrieval_score","address_rank"]].copy()
    con=duckdb.connect(str(DB),read_only=True);con.register("pairs",core)
    complete=con.execute("""select p.s1 source1_entity_id,p.target_id candidate_entity_id,
      case when p.source=2 then 'S2' else 'S3' end candidate_source,p.country,p.role,p.label,
      s.business_name s1_business_name,coalesce(s.business_address,'') s1_business_address,
      a.business_name candidate_business_name,coalesce(a.business_address,'') candidate_business_address,
      p.score lightgbm_score,p.lightgbm_rank_within_s1,p.name_retrieval_score,p.name_rank,
      p.address_retrieval_score,p.address_rank from pairs p join train_s1 s on p.s1=s.entity_id
      join (select 2 as src,* from train_s2 union all by name select 3 as src,* from train_s3) a
      on p.source=a.src and p.target_id=a.entity_id order by p.s1,p.lightgbm_rank_within_s1,p.target_id""").fetchdf();con.close()
    t=time.perf_counter();shortlist=e005_common.shortlist(complete);shortlist_seconds=time.perf_counter()-t
    historical=pd.read_parquet(E005/"outputs/evaluation_predictions.parquet")
    key=["source1_entity_id","candidate_entity_id","candidate_source"]
    hist_pred=historical[key+["s1_business_name","s1_business_address","candidate_business_name","candidate_business_address","neural_probability"]]
    reuse=shortlist.merge(hist_pred,on=key,how="inner",suffixes=("","_historical"),validate="one_to_one")
    text_fields=["s1_business_name","s1_business_address","candidate_business_name","candidate_business_address"]
    text_equal={c:bool(reuse[c].fillna("").eq(reuse[c+"_historical"].fillna("")).all()) for c in text_fields}
    if not all(text_equal.values()):raise RuntimeError(f"historical XLM-R input text mismatch: {text_equal}")
    hist_keys=pd.MultiIndex.from_frame(historical[key]);short_keys=pd.MultiIndex.from_frame(shortlist[key])
    new=shortlist.loc[~short_keys.isin(hist_keys)].copy();lost=historical.loc[~hist_keys.isin(short_keys)].copy()
    reused=shortlist.merge(historical[key+["neural_probability"]],on=key,how="inner",validate="one_to_one")
    HERE.mkdir(parents=True,exist_ok=True)
    candidates.to_parquet(HERE/"r006b_candidates.parquet",index=False,compression="zstd")
    scored.to_parquet(HERE/"engineered_features_scored.parquet",index=False,compression="zstd")
    complete.to_parquet(HERE/"complete_pairs.parquet",index=False,compression="zstd")
    shortlist.to_parquet(HERE/"shortlist.parquet",index=False,compression="zstd")
    reused.to_parquet(HERE/"reused_predictions.parquet",index=False,compression="zstd")
    new.to_parquet(HERE/"new_shortlist_pairs.parquet",index=False,compression="zstd")
    lost.to_parquet(HERE/"lost_historical_pairs.parquet",index=False,compression="zstd")
    meta={"cohort_rows":len(selected),"candidate_pairs":len(complete),"shortlist_pairs":len(shortlist),"shortlist_per_s1":len(shortlist)/len(selected),
      "historical_shortlist_pairs":len(historical),"reused_pairs":len(reused),"new_pairs":len(new),"lost_historical_pairs":len(lost),
      "retrieval_regeneration_seconds":retrieval_seconds,"feature_generation_seconds":feature_seconds,
      "lightgbm_scoring_seconds":score_seconds,"feature_and_scoring_seconds":feature_seconds+score_seconds,
      "shortlist_construction_seconds":shortlist_seconds,"peak_rss_gb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
      "text_equality":text_equal,"lightgbm_path":str(E001/"model/lightgbm_model.txt"),
      "lightgbm_sha256":sha256(E001/"model/lightgbm_model.txt"),"xlmr_path":str(E005/"model/best_checkpoint/model.safetensors"),
      "xlmr_sha256":sha256(E005/"model/best_checkpoint/model.safetensors")}
    atomic_json(HERE/"prepare_runtime.json",meta);print(json.dumps(meta,indent=2))


def finalize():
    metric_self_test();start=time.perf_counter();selected=pd.read_parquet(E005/"runtime/selected_s1_metadata.parquet").query("role=='evaluation'").copy()
    complete=pd.read_parquet(HERE/"complete_pairs.parquet");shortlist=pd.read_parquet(HERE/"shortlist.parquet")
    reused=pd.read_parquet(HERE/"reused_predictions.parquet");new_input=pd.read_parquet(HERE/"new_shortlist_pairs.parquet")
    new_pred=pd.read_parquet(HERE/"new_predictions.parquet") if len(new_input) else new_input.assign(neural_probability=np.float32([]))
    key=["source1_entity_id","candidate_entity_id","candidate_source"]
    probs=pd.concat([reused[key+["neural_probability"]],new_pred[key+["neural_probability"]]],ignore_index=True)
    d=shortlist.merge(probs,on=key,how="left",validate="one_to_one");
    if d.neural_probability.isna().any():raise RuntimeError("missing neural probability")
    cfg={"decoder":"one_owner_margin","threshold":.51,"margin":.48,"weight":.60}
    scores=score_column(d,"E005B",.60);mask=mask_for(d,scores,cfg);met=grouped_f05(d,selected,mask)
    shortlist_oracle=grouped_f05(d,selected,d.label.eq(1).to_numpy())["macro_f05"]
    blocker_oracle=grouped_f05(complete,selected,complete.label.eq(1).to_numpy())["macro_f05"]
    slices=pd.DataFrame(slice_rows(d,complete,selected,mask,"R006B_E005B"))
    historical=pd.read_parquet(E005/"outputs/evaluation_predictions.parquet");historical_slices=pd.read_csv(E005/"outputs/slice_results.csv")
    hist_cfg=json.loads((E005/"model/frozen_parameters.json").read_text())["E005B"]
    hist_mask=mask_for(historical,score_column(historical,"E005B",hist_cfg["weight"]),hist_cfg)
    hist_met=grouped_f05(historical,selected,hist_mask)
    if abs(hist_met["macro_f05"]-0.976693878247762)>1e-12:raise RuntimeError("historical E005B reproduction failed")
    truth=r006a.truth_for(r006a.cohort(10000));truth_keys=set(map(tuple,truth[["s1","target_id","source"]].itertuples(index=False,name=None)))
    def keys(frame):
      return set((s,t,int(str(src)[-1])) for s,t,src in frame[key].itertuples(index=False,name=None))
    rkeys=keys(complete);hfull=pd.read_parquet(ROOT/"analysis/experiments_v1/e004_k10_decode_and_gpu_bundle/gpu_bundle/evaluation_pairs.parquet");hkeys=keys(hfull)
    recovered=(rkeys-hkeys)&truth_keys;lost=(hkeys-rkeys)&truth_keys;short_keys=keys(d);accepted=keys(d.loc[mask]);hist_accepted=keys(historical.loc[hist_mask])
    diagnostics={"recovered_true_edges":len(recovered),"lost_true_edges":len(lost),
      "recovered_surviving_shortlist":len(recovered&short_keys),"recovered_finally_accepted":len(recovered&accepted),
      "lost_edges_historically_accepted":len(lost&hist_accepted)}
    slice_specs=[("US","country","US"),("India","country","India"),("S2","source","S2"),("S3","source","S3"),
      ("cardinality-1","cardinality","1"),("repeated/common names","name_ambiguity","repeated/common")]
    slice_out=[]
    for label,sl,val in slice_specs:
      a=historical_slices[(historical_slices.experiment.eq("E005B"))&historical_slices.slice.eq(sl)&historical_slices.value.astype(str).eq(val)]
      b=slices[(slices.slice.eq(sl))&slices.value.astype(str).eq(val)]
      slice_out.append({"slice":label,"historical":float(a.macro_f05.iloc[0]),"r006b":float(b.macro_f05.iloc[0]),"delta":float(b.macro_f05.iloc[0]-a.macro_f05.iloc[0])})
    fusion_seconds=time.perf_counter()-start;prep=json.loads((HERE/"prepare_runtime.json").read_text());infer=json.loads((HERE/"new_predictions.runtime.json").read_text())
    hist_short=len(historical);ratio=TEST_S1/10000
    projections={"retrieval_minutes":29.0,
      "features_lightgbm_minutes":prep["feature_and_scoring_seconds"]*ratio/60,
      "xlmr_minutes":len(d)*ratio/infer["pairs_per_second"]/60,
      "fusion_submission_minutes":fusion_seconds*ratio/60}
    projections["total_single_gpu_minutes"]=sum(projections.values())
    projections["total_two_gpu_minutes"]=projections["retrieval_minutes"]+projections["features_lightgbm_minutes"]+projections["xlmr_minutes"]/2+projections["fusion_submission_minutes"]
    report={"historical":{"metrics":hist_met,"blocker_oracle":.9874557321200704,"shortlist_oracle":.9866963636808983},
      "r006b":{"metrics":met,"blocker_oracle":blocker_oracle,"shortlist_oracle":shortlist_oracle,
      "oracle_model_gap":shortlist_oracle-met["macro_f05"]},"slices":slice_out,"diagnostics":diagnostics,
      "fusion_decoder_seconds":fusion_seconds,"projections":projections,"prepare":prep,"inference":infer,
      "correctness":{"same_cohort":len(selected)==10000 and set(selected.source1_entity_id)==set(r006a.cohort(10000).source1_entity_id),
      "r006b_candidate_count":len(complete),"feature_count":49,"lightgbm_hash_ok":prep["lightgbm_sha256"]=="226a35a7354a26f6edcab07893947115696bd1187d5e82aca70c0e0a3aa6f333",
      "xlmr_hash_ok":prep["xlmr_sha256"]=="e7079b798b866b27116a0059c9c7cf1531817e42659ec0c9442654aee5d0c462",
      "cached_text_identity":all(prep["text_equality"].values()),"frozen_policy":cfg,"official_metric_reused":True,"no_training_or_calibration":True}}
    d.assign(e005b_score=scores,e005b_accepted=mask).to_parquet(HERE/"evaluation_predictions.parquet",index=False,compression="zstd")
    slices.to_csv(HERE/"slice_results.csv",index=False);atomic_json(HERE/"results.json",report);print(json.dumps(report,indent=2))


def main():
    ap=argparse.ArgumentParser();ap.add_argument("stage",choices=["prepare","finalize"]);args=ap.parse_args()
    generate_features() if args.stage=="prepare" else finalize()


if __name__=="__main__":main()
