#!/usr/bin/env python3
"""Full frozen 10k R007 production-regression gate (CPU and final decode)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "analysis/production/p002_final"
E001 = ROOT / "analysis/experiments_v1/e001_classical_baseline"
E005 = ROOT / "analysis/experiments_v1/e005_xlmr_crossencoder"
R006 = ROOT / "analysis/experiments_v1/r006a_selective_postings"
R007 = ROOT / "analysis/experiments_v1/r007_fast_blocker_e005b"
sys.path.insert(0, str(E001 / "scripts")); import prepare_baseline as e001  # noqa: E402
sys.path.insert(0, str(R006)); import run_r006a, run_r006b  # noqa: E402
sys.path.insert(0, str(R007)); import run_r007  # noqa: E402
sys.path.insert(0, str(E005 / "src")); from evaluate_e005 import mask_for  # noqa: E402
from metric import grouped_f05  # noqa: E402
from feature_builder import add_context, build_features  # noqa: E402
from production import atomic_json, atomic_parquet  # noqa: E402


KEY = ["source1_entity_id", "candidate_entity_id", "candidate_source"]


def compare_keys(a, b, cols): return set(map(tuple,a[cols].itertuples(index=False,name=None))) == set(map(tuple,b[cols].itertuples(index=False,name=None)))


def cpu():
    start=time.time();selected=run_r006a.cohort(10000);frozen_cand=pd.read_parquet(R007/"r006b_candidates.parquet")
    frozen_feat=pd.read_parquet(R007/"engineered_features_scored.parquet");model=joblib.load(E001/"model/lightgbm_model.joblib")
    fs=json.loads((E001/"model/model_config.json").read_text())["features"];fn,_=run_r006b.load_functions();cand_chunks=[];feat_chunks=[];group_rows=[]
    for (country,source),expected in frozen_cand.groupby(["country","source"],sort=False):
        source=int(source);sc=selected[selected.country.eq(country)].sort_values("source1_entity_id",kind="mergesort");q=run_r007.full_query(sc);aux,indexes=run_r006a.build_or_load(country,source);fullaux=run_r007.full_aux(country,source)
        if aux.entity_id.tolist()!=fullaux.entity_id.tolist():raise RuntimeError("preflight aux identity mismatch")
        frames=[];mats={}
        for channel in ["name","address"]:
            xa,idf=indexes[channel];xq=run_r006a.qmatrix(q,channel,idf);mats[channel]=xq;oi,ov,*_=run_r006b.selective(fn,xq,xa,800,20);frames.append(run_r007.stream_frame(q.entity_id.to_numpy(),oi,ov,channel))
        cand=frames[0].merge(frames[1],on=["qpos","apos"],how="outer");cand["s1"]=q.entity_id.to_numpy()[cand.qpos];cand["target_id"]=fullaux.entity_id.to_numpy()[cand.apos]
        cand["source"]=np.int8(source);cand["country"]=country;cand["role"]="evaluation"
        for channel in ["name","address"]:cand[f"{channel}_char_tfidf"]=e001.row_cosines(mats[channel],indexes[channel][0],cand.qpos.to_numpy(),cand.apos.to_numpy())
        nf=fullaux.nname.value_counts(dropna=False);af=fullaux.naddr.value_counts(dropna=False)
        cand["s1_normalized_name_fanout"]=q.nname.map(nf).fillna(0).clip(0,65535).astype(np.uint16).to_numpy()[cand.qpos]
        cand["s1_normalized_address_fanout"]=q.naddr.map(af).fillna(0).clip(0,65535).astype(np.uint16).to_numpy()[cand.qpos]
        cand["candidate_normalized_name_fanout"]=fullaux.nname.map(nf).fillna(0).clip(0,65535).astype(np.uint16).to_numpy()[cand.apos]
        cand["candidate_normalized_address_fanout"]=fullaux.naddr.map(af).fillna(0).clip(0,65535).astype(np.uint16).to_numpy()[cand.apos]
        qn=q.nname.to_numpy()[cand.qpos];qa=q.naddr.to_numpy()[cand.qpos];an=fullaux.nname.to_numpy()[cand.apos];aa=fullaux.naddr.to_numpy()[cand.apos]
        cand["nname_exact"]=((qn!="")&(qn==an)).astype(np.int8);cand["naddr_exact"]=((qa!="")&(qa==aa)).astype(np.int8);cand["label"]=np.int8(0);cand=add_context(cand)
        exp=expected.sort_values(["s1","target_id"],kind="mergesort").reset_index(drop=True);got=cand.sort_values(["s1","target_id"],kind="mergesort").reset_index(drop=True)
        if not got[["s1","target_id","source"]].equals(exp[["s1","target_id","source"]]):raise RuntimeError(f"R006B key parity failed {country} S{source}")
        cols=[c for c in expected.columns if c not in ["label"] and pd.api.types.is_numeric_dtype(expected[c])]
        maxdiff=max(float(np.max(np.abs(got[c].to_numpy(float)-exp[c].to_numpy(float)))) for c in cols)
        if maxdiff>1e-6:raise RuntimeError(f"candidate numeric parity failed {country} S{source}: {maxdiff}")
        # Restore frozen labels solely for the held-out metric regression.
        lab=expected[["s1","target_id","label"]];cand=cand.drop(columns="label").merge(lab,on=["s1","target_id"],validate="one_to_one")
        feat=build_features(cand,q,fullaux,workers=20);feat["score"]=model.predict_proba(feat[fs])[:,1].astype(np.float32)
        group_rows.append({"country":country,"source":source,"pairs":len(cand),"candidate_max_abs":maxdiff});cand_chunks.append(cand);feat_chunks.append(feat)
    candidates=pd.concat(cand_chunks,ignore_index=True);scored=pd.concat(feat_chunks,ignore_index=True)
    a=scored.sort_values(["s1","target_id","source"],kind="mergesort").reset_index(drop=True);b=frozen_feat.sort_values(["s1","target_id","source"],kind="mergesort").reset_index(drop=True)
    fmax=max(float(np.max(np.abs(a[c].to_numpy(float)-b[c].to_numpy(float)))) for c in fs);smax=float(np.max(np.abs(a.score-b.score)))
    scored=scored.sort_values(["s1","score","target_id"],ascending=[True,False,True],kind="mergesort");scored["lightgbm_rank_within_s1"]=(scored.groupby("s1",sort=False).cumcount()+1).astype(np.int16)
    core=scored[["s1","target_id","source","country","role","label","score","lightgbm_rank_within_s1","name_retrieval_score","name_rank","address_retrieval_score","address_rank"]].copy()
    con=__import__('duckdb').connect(str(ROOT/"analysis/data_audit_v1/artifacts/audit.duckdb"),read_only=True);con.register("pairs",core)
    complete=con.execute("""select p.s1 source1_entity_id,p.target_id candidate_entity_id,case when p.source=2 then 'S2'else'S3'end candidate_source,p.country,p.role,p.label,s.business_name s1_business_name,coalesce(s.business_address,'')s1_business_address,a.business_name candidate_business_name,coalesce(a.business_address,'')candidate_business_address,p.score lightgbm_score,p.lightgbm_rank_within_s1,p.name_retrieval_score,p.name_rank,p.address_retrieval_score,p.address_rank from pairs p join train_s1 s on p.s1=s.entity_id join(select 2 src,*from train_s2 union all by name select 3 src,*from train_s3)a on p.source=a.src and p.target_id=a.entity_id order by p.s1,p.lightgbm_rank_within_s1,p.target_id""").fetchdf();con.close()
    shortlist=complete[complete.lightgbm_rank_within_s1.le(5)|complete.lightgbm_score.ge(.05)].copy();frozen_short=pd.read_parquet(R007/"shortlist.parquet")
    short_ok=compare_keys(shortlist,frozen_short,KEY);texts=["s1_business_name","s1_business_address","candidate_business_name","candidate_business_address"]
    z=shortlist.merge(frozen_short[KEY+texts],on=KEY,suffixes=("","_f"),validate="one_to_one");text_ok=all(z[c].fillna("").eq(z[c+"_f"].fillna("")).all()for c in texts)
    if not(short_ok and text_ok and fmax<=1e-6 and smax<=1e-7):raise RuntimeError("CPU production regression failed")
    atomic_parquet(shortlist,EXP/"preflight/shortlist.parquet")
    report={"candidate_count":len(candidates),"candidate_identity":True,"candidate_groups":group_rows,"feature_count":len(fs),"feature_max_abs":fmax,"lightgbm_max_abs":smax,"shortlist_count":len(shortlist),"shortlist_identity":short_ok,"text_identity":text_ok,"seconds":time.time()-start,"ok":True}
    atomic_json(EXP/"diagnostics/preflight_cpu.json",report);print(json.dumps(report,indent=2))


def finalize(neural:Path):
    start=time.time();d=pd.read_parquet(EXP/"preflight/shortlist.parquet");n=pd.read_parquet(neural);d=d.merge(n,on=KEY,validate="one_to_one");hist=pd.read_parquet(R007/"evaluation_predictions.parquet")
    z=d.merge(hist[KEY+["neural_probability"]],on=KEY,suffixes=("","_historical"),validate="one_to_one");diff=np.abs(z.neural_probability-z.neural_probability_historical)
    cfg={"decoder":"one_owner_margin","threshold":.51,"margin":.48,"weight":.60};score=.6*d.neural_probability.to_numpy(float)+.4*d.lightgbm_score.to_numpy(float);mask=mask_for(d,score,cfg)
    hscore=.6*hist.neural_probability.to_numpy(float)+.4*hist.lightgbm_score.to_numpy(float);hmask=mask_for(hist,hscore,cfg)
    keys=lambda x,m:set(map(tuple,x.loc[m,KEY].itertuples(index=False,name=None)));pred_ok=keys(d,mask)==keys(hist,hmask)
    selected=pd.read_parquet(E005/"runtime/selected_s1_metadata.parquet").query("role=='evaluation'");met=grouped_f05(d,selected,mask);target=0.9771333813046491
    report={"xlmr_probability_max_abs":float(diff.max()),"xlmr_probability_p999_abs":float(np.quantile(diff,.999)),"xlmr_rows_over_1e-3":int((diff>1e-3).sum()),"final_prediction_identity":pred_ok,"macro_f05":met["macro_f05"],"expected_macro_f05":target,"absolute_difference":abs(met["macro_f05"]-target),"seconds":time.time()-start}
    report["ok"]=pred_ok and report["absolute_difference"]<1e-12
    atomic_json(EXP/"diagnostics/preflight_final.json",report);print(json.dumps(report,indent=2))
    if not report["ok"]:raise RuntimeError("final preflight failed")


def main():
    ap=argparse.ArgumentParser();ap.add_argument("stage",choices=["cpu","finalize"]);ap.add_argument("--neural",type=Path);a=ap.parse_args();(EXP/"preflight").mkdir(parents=True,exist_ok=True)
    cpu() if a.stage=="cpu" else finalize(a.neural)
if __name__=="__main__":main()
