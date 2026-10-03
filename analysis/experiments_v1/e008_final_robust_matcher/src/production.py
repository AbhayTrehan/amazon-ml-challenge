#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import glob
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import *
from calibrate import PAIR_FEATURES, OWNER_FEATURES, pair_matrix
from neural import EncodedDataset, Collator, encode_frame, model_and_tokenizer

sys.path.insert(0,str(P002/"src"));from feature_builder import add_context,build_features  # noqa

INDEX=P001/"cache/indexes";TEST=ROOT/"student_resource/dataset/test";COUNTRIES=["France","India","US"];SHARD=20000
FEATURES=json.loads((E001/"model/model_config.json").read_text())["features"]


def init():
    for d in ["cache/lightgbm","cache/shortlist","cache/neural","cache/scored","cache/ducktmp","final","diagnostics","logs","runtime"]:(P003/d).mkdir(parents=True,exist_ok=True)


def qpath(c):return INDEX/c/"queries.parquet"
def apath(c,s):return INDEX/c/f"s{s}/aux.parquet"
def blocker(c,s,p):return P002/f"cache/blocker/{c}/s{s}/part-{p:05d}.parquet"


def feature_stage():
    init();model=joblib.load(EXP/"model/lightgbm/model.joblib");t0=time.time();rows=0
    for country in COUNTRIES:
      qfull=pd.read_parquet(qpath(country))
      for source in [2,3]:
        aux=pd.read_parquet(apath(country,source));nf=aux.nname.value_counts(dropna=False);af=aux.naddr.value_counts(dropna=False)
        for part,start in enumerate(range(0,len(qfull),SHARD)):
            op=P003/f"cache/lightgbm/{country}/s{source}/part-{part:05d}.parquet"
            if op.exists():continue
            q=qfull.iloc[start:start+SHARD].reset_index(drop=True);cand=pd.read_parquet(blocker(country,source,part));cand["source"]=np.int8(source);cand["country"]=country;cand["role"]="test";cand["label"]=np.int8(0)
            cand["s1_normalized_name_fanout"]=q.nname.map(nf).fillna(0).clip(0,65535).astype(np.uint16).to_numpy()[cand.qpos];cand["s1_normalized_address_fanout"]=q.naddr.map(af).fillna(0).clip(0,65535).astype(np.uint16).to_numpy()[cand.qpos]
            cand["candidate_normalized_name_fanout"]=aux.candidate_normalized_name_fanout.to_numpy()[cand.apos];cand["candidate_normalized_address_fanout"]=aux.candidate_normalized_address_fanout.to_numpy()[cand.apos]
            qn=q.nname.to_numpy()[cand.qpos];qa=q.naddr.fillna("").to_numpy()[cand.qpos];an=aux.nname.to_numpy()[cand.apos];aa=aux.naddr.fillna("").to_numpy()[cand.apos]
            cand["nname_exact"]=((qn!="")&(qn==an)).astype(np.int8);cand["naddr_exact"]=((qa!="")&(qa==aa)).astype(np.int8);cand=add_context(cand);feat=build_features(cand,q,aux,workers=20)
            keep=["s1","target_id","source","country","name_retrieval_score","name_rank","address_retrieval_score","address_rank","nname_exact","naddr_exact","numeric_token_jaccard","numeric_exact_overlap_count","obvious_numeric_conflict"]
            out=feat[keep].copy();out["lightgbm_score"]=model.predict_proba(feat[FEATURES])[:,1].astype(np.float32);atomic_parquet(out,op);rows+=len(out);print("P003_FEATURE",country,source,part,len(out),flush=True)
            del cand,feat,out;gc.collect()
        del aux;gc.collect()
    atomic_json(P003/"runtime/features.json",{"rows":rows,"seconds":time.time()-t0,"reused_blocker":str(P002/"cache/blocker")})


def shortlist_stage():
    init();t0=time.time();total=0
    for country in COUNTRIES:
      qfull=pd.read_parquet(qpath(country));aux={s:pd.read_parquet(apath(country,s)).set_index("entity_id") for s in [2,3]}
      for part,start in enumerate(range(0,len(qfull),SHARD)):
        op=P003/f"cache/shortlist/{country}/part-{part:05d}.parquet"
        if op.exists():continue
        d=pd.concat([pd.read_parquet(P003/f"cache/lightgbm/{country}/s{s}/part-{part:05d}.parquet") for s in [2,3]],ignore_index=True)
        d=d.sort_values(["s1","lightgbm_score","target_id"],ascending=[True,False,True],kind="mergesort");d["lightgbm_rank_within_s1"]=(d.groupby("s1",sort=False).cumcount()+1).astype(np.int16);d=d[d.lightgbm_rank_within_s1.le(5)|d.lightgbm_score.ge(.05)].copy()
        q=qfull.iloc[start:start+SHARD].set_index("entity_id");d["s1_business_name"]=d.s1.map(q.business_name);d["s1_business_address"]=d.s1.map(q.business_address).fillna("");d["s1_nname"]=d.s1.map(q.nname);d["s1_naddr"]=d.s1.map(q.naddr).fillna("")
        for col in ["candidate_business_name","candidate_business_address","candidate_nname","candidate_naddr"]:d[col]=""
        for s in [2,3]:
            take=d.source.eq(s);ids=d.loc[take,"target_id"];d.loc[take,"candidate_business_name"]=ids.map(aux[s].business_name).to_numpy();d.loc[take,"candidate_business_address"]=ids.map(aux[s].business_address).fillna("").to_numpy();d.loc[take,"candidate_nname"]=ids.map(aux[s].nname).to_numpy();d.loc[take,"candidate_naddr"]=ids.map(aux[s].naddr).fillna("").to_numpy()
        d.rename(columns={"s1":"source1_entity_id","target_id":"candidate_entity_id"},inplace=True);d["candidate_source"]=np.where(d.source.eq(2),"S2","S3");d["s1_numbers"]=d.s1_naddr.map(numeric_tokens);d["candidate_numbers"]=d.candidate_naddr.map(numeric_tokens)
        atomic_parquet(d,op);total+=len(d);print("P003_SHORTLIST",country,part,len(d),flush=True)
    atomic_json(P003/"runtime/shortlist.json",{"rows":total,"seconds":time.time()-t0})


def neural_stage():
    init();epoch=int((EXP/"model/xlmr/selected_epoch.txt").read_text());model,tok=model_and_tokenizer(EXP/f"model/xlmr/epoch-{epoch}");model.cuda().eval();total=trunc=0;t0=time.time()
    for ip in sorted((P003/"cache/shortlist").glob("*/*.parquet")):
        rel=ip.relative_to(P003/"cache/shortlist");op=P003/"cache/neural"/rel
        if op.exists():continue
        d=pd.read_parquet(ip);ids,tr=encode_frame(d,tok);ds=EncodedDataset(ids);loader=DataLoader(ds,batch_size=512,shuffle=False,num_workers=4,pin_memory=True,collate_fn=Collator(ds,tok));probs=[]
        with torch.inference_mode():
            for bt in loader:
                bt={k:v.cuda(non_blocking=True) for k,v in bt.items()}
                with torch.autocast("cuda",dtype=torch.bfloat16):probs.append(torch.sigmoid(model(**bt).logits.squeeze(-1).float()).cpu().numpy())
        atomic_parquet(pd.DataFrame({"xlmr_probability":np.concatenate(probs).astype(np.float32)}),op);total+=len(d);trunc+=tr;print("P003_NEURAL",rel,len(d),flush=True)
    atomic_json(P003/"runtime/neural.json",{"rows":total,"truncated":trunc,"truncation_rate":trunc/total,"seconds":time.time()-t0,"gpu":"RTX A6000 physical 0"})


def score_stage():
    init();stack=joblib.load(EXP/"calibration/pair_stacker.joblib");t0=time.time();total=0
    for ip in sorted((P003/"cache/shortlist").glob("*/*.parquet")):
        rel=ip.relative_to(P003/"cache/shortlist");op=P003/"cache/scored"/rel
        if op.exists():continue
        d=pd.read_parquet(ip);d["xlmr_probability"]=pd.read_parquet(P003/"cache/neural"/rel).xlmr_probability.to_numpy();d["pair_probability"]=stack.predict_proba(pair_matrix(d))[:,1].astype(np.float32)
        keep=["source1_entity_id","candidate_entity_id","candidate_source","source","country","pair_probability","name_retrieval_score","address_retrieval_score"]
        atomic_parquet(d[keep],op);total+=len(d);print("P003_SCORE",rel,len(d),flush=True)
    atomic_json(P003/"runtime/scoring.json",{"rows":total,"seconds":time.time()-t0})


def decode_stage():
    init();t0=time.time();sglob=str(P003/"cache/scored/*/*.parquet");con=duckdb.connect();con.execute("PRAGMA threads=20");con.execute("PRAGMA memory_limit='100GB'");con.execute(f"PRAGMA temp_directory='{P003/'cache/ducktmp'}'")
    winners=P003/"cache/owner_winners.parquet"
    con.execute(f"""copy(select * exclude(owner_rank) from(select *,row_number()over(partition by candidate_entity_id order by pair_probability desc,source1_entity_id)owner_rank,
      lead(pair_probability)over(partition by candidate_entity_id order by pair_probability desc,source1_entity_id)second_pair_probability,
      count(*)over(partition by candidate_entity_id)owner_count from read_parquet('{sglob}',union_by_name=true))where owner_rank=1)to '{winners}'(format parquet,compression zstd)""")
    w=pd.read_parquet(winners);w["best_pair_probability"]=w.pair_probability;w["second_pair_probability"]=w.second_pair_probability.fillna(0.0);w["logit_gap"]=logit(w.best_pair_probability)-logit(w.second_pair_probability);w["probability_ratio"]=w.best_pair_probability/np.clip(w.second_pair_probability,1e-6,None);w["source3"]=w.candidate_source.eq("S3").astype(np.int8);w["best_lexical_retrieval_score"]=w[["name_retrieval_score","address_retrieval_score"]].max(axis=1);w["owner_probability"]=1.0
    owner=joblib.load(EXP/"calibration/owner_calibrator.joblib");take=w.owner_count.gt(1);w.loc[take,"owner_probability"]=owner.predict_proba(w.loc[take,OWNER_FEATURES])[:,1];w["effective_probability"]=w.pair_probability*w.owner_probability
    mask=expected_utility_decode(w);accepted=w.loc[mask,["source1_entity_id","candidate_entity_id","candidate_source","country","pair_probability","owner_probability","effective_probability"]].copy();atomic_parquet(accepted,P003/"final/accepted_pairs.parquet")
    con.register("accepted",accepted)
    con.execute(f"""copy(with q as(select *,row_number()over()rn from read_csv('{TEST/'test_source1.tsv'}',delim='\t',header=true,all_varchar=true,quote=''))select q.entity_id source1_entity_id,string_agg(a.candidate_entity_id,',' order by a.candidate_entity_id)matched_entity_ids from q left join accepted a on q.entity_id=a.source1_entity_id group by q.entity_id,q.rn order by q.rn)to '{P003/'final/matching_results.tsv'}'(header,delimiter '\t')""")
    con.execute(f"""copy(with q as(select *,row_number()over()rn from read_csv('{TEST/'test_source1.tsv'}',delim='\t',header=true,all_varchar=true,quote='')),s as(select source1_entity_id,candidate_entity_id from read_parquet('{sglob}',union_by_name=true))select q.entity_id source1_entity_id,string_agg(s.candidate_entity_id,',' order by s.candidate_entity_id)candidate_entity_ids from q left join s on q.entity_id=s.source1_entity_id group by q.entity_id,q.rn order by q.rn)to '{P003/'final/candidate_pairs.tsv'}'(header,delimiter '\t')""");con.close()
    k=w.loc[mask].groupby("source1_entity_id").size();diag={"shortlist_pairs":sum(len(pd.read_parquet(p,columns=["pair_probability"])) for p in (P003/"cache/scored").glob("*/*.parquet")),"owner_winners":len(w),"contested":int(take.sum()),"owner_rejections":int((w.owner_count-1).sum()),"predicted_edges":len(accepted),"predicted_singletons":1732544-int(accepted.source1_entity_id.nunique()),"k_distribution":k.value_counts().sort_index().to_dict(),"seconds":time.time()-t0};atomic_json(P003/"diagnostics/decode.json",diag);print(json.dumps(diag,indent=2))


def validate_stage():
    cmd=[sys.executable,str(ROOT/"student_resource/utils/validate_submission.py"),"--matching",str(P003/"final/matching_results.tsv"),"--candidate",str(P003/"final/candidate_pairs.tsv"),"--test-dir",str(TEST),"--check-ids"]
    p=subprocess.run(cmd,text=True,capture_output=True);text=p.stdout+p.stderr;(P003/"logs/official_validator.log").write_text(text);ok=p.returncode==0 and "PASS — no blocking issues found" in text and "WARNING:" not in text
    hashes={x:sha256(P003/"final"/x) for x in ["matching_results.tsv","candidate_pairs.tsv"]};atomic_json(P003/"diagnostics/validation.json",{"official_pass":ok,"hashes":hashes})
    if not ok:raise RuntimeError(text)
    shutil.copy2(P003/"final/matching_results.tsv",P003/"final/matching_results.tsv") if False else None;print(text)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("stage",choices=["features","shortlist","neural","score","decode","validate"]);a=ap.parse_args();{"features":feature_stage,"shortlist":shortlist_stage,"neural":neural_stage,"score":score_stage,"decode":decode_stage,"validate":validate_stage}[a.stage]()
if __name__=="__main__":main()
