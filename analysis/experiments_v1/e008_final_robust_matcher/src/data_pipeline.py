#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import sys
import time
from pathlib import Path

import duckdb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer

from common import *

sys.path.insert(0, str(E001 / "scripts")); import prepare_baseline as e001  # noqa
sys.path.insert(0, str(R006)); import run_r006a, run_r006b  # noqa
sys.path.insert(0, str(P002 / "src")); from feature_builder import add_context, build_features  # noqa

SHARD = 20_000
FEATURES = json.loads((E001 / "model/model_config.json").read_text())["features"]


class FrozenSelectiveIndex:
    def __init__(self, matrix):
        self.a = matrix.tocsr(); c = self.a.tocsc()
        self.cip = run_r006a.i32(c.indptr); self.cix = run_r006a.i32(c.indices); self.cv = run_r006a.f32(c.data)
        self.aip = run_r006a.i32(self.a.indptr); self.aix = run_r006a.i32(self.a.indices); self.av = run_r006a.f32(self.a.data)
        self.fn, _ = run_r006b.load_functions()

    def query(self, xq):
        q=xq.tocsr(); qip=run_r006a.i32(q.indptr); qix=run_r006a.i32(q.indices); qv=run_r006a.f32(q.data)
        n=q.shape[0]; oi=np.full((n,10),-1,np.int32); ov=np.zeros((n,10),np.float32)
        visits=np.zeros(n,np.int64); full=np.zeros(n,np.int64); counts=np.zeros(n,np.int32); fallback=np.zeros(n,np.int32); timing=np.zeros(5,np.float64)
        ip=ctypes.POINTER(ctypes.c_int32); fp=ctypes.POINTER(ctypes.c_float); i64p=ctypes.POINTER(ctypes.c_int64); dp=ctypes.POINTER(ctypes.c_double)
        t=time.perf_counter(); rc=self.fn(qip.ctypes.data_as(ip),qix.ctypes.data_as(ip),qv.ctypes.data_as(fp),n,
            self.cip.ctypes.data_as(ip),self.cix.ctypes.data_as(ip),self.cv.ctypes.data_as(fp),
            self.aip.ctypes.data_as(ip),self.aix.ctypes.data_as(ip),self.av.ctypes.data_as(fp),self.a.shape[0],
            20000,800,20,oi.ctypes.data_as(ip),ov.ctypes.data_as(fp),visits.ctypes.data_as(i64p),full.ctypes.data_as(i64p),
            counts.ctypes.data_as(ip),fallback.ctypes.data_as(ip),timing.ctypes.data_as(dp))
        if rc: raise RuntimeError(f"R006B failed {rc}")
        return oi,ov,time.perf_counter()-t


def qmatrix(frame, channel, idf):
    tf=TfidfTransformer(); tf.idf_=idf
    col="business_name" if channel=="name" else "business_address"
    return tf.transform(e001.vec_for(channel).transform(frame[col].fillna(""))).astype(np.float32).tocsr()


def stream(inds,scores,channel):
    rr,kk=np.where(inds>=0)
    return pd.DataFrame({"qpos":rr.astype(np.int32),"apos":inds[rr,kk].astype(np.int32),
        f"{channel}_rank":(kk+1).astype(np.int16),f"{channel}_retrieval_score":scores[rr,kk].astype(np.float32)})


def prepare_ids():
    mkdirs(); split=pd.read_csv(SPLIT,sep="\t")
    pool=split[split.fold.isin([0,1,2,3])].copy(); pool["_h"]=pool.source1_entity_id.map(lambda x:stable_hash(x,"e008-final-cohort-20260925"))
    train=pd.concat([pool[pool.country.eq(c)].nsmallest(100000,"_h") for c in ["US","India"]],ignore_index=True).drop(columns="_h")
    if len(train)!=200000 or train.groupby("country").size().to_dict()!={"India":100000,"US":100000}: raise RuntimeError("cohort selection failed")
    hist=pd.read_parquet(E001/"selected_ids/selected_s1.parquet").query("role=='calibration'")[["source1_entity_id"]]
    fold4=split.loc[split.fold.eq(4),["source1_entity_id","country","fold"]].copy(); fold4["is_calibration"]=fold4.source1_entity_id.isin(hist.source1_entity_id)
    if int(fold4.is_calibration.sum())!=10000: raise RuntimeError("historical calibration identity failure")
    atomic_parquet(train,EXP/"data/training_s1.parquet"); atomic_parquet(fold4,EXP/"data/fold4_s1.parquet")
    for role,d in [("train",train),("fold4",fold4)]:
        for country,g in d.groupby("country",sort=True): atomic_parquet(g.sort_values("source1_entity_id"),EXP/f"data/{role}_{country}_queries.parquet")
    ids="\n".join(train.source1_entity_id.sort_values())+"\n"; (EXP/"data/training_ids.txt").write_text(ids)
    con=duckdb.connect(str(DB),read_only=True); con.register("ids",train[["source1_entity_id"]])
    dist=con.execute("""select q.country,e.target_source,count(*) positives from edges e join ids i on e.s1=i.source1_entity_id join train_s1 q on q.entity_id=e.s1 group by all order by all""").fetchdf()
    card=train.groupby(["country","true_matches"]).size().reset_index(name="s1")
    report={"s1_by_country":train.country.value_counts().sort_index().to_dict(),"positive_by_country_source":dist.to_dict("records"),
            "positive_total":int(dist.positives.sum()),"cardinality":card.to_dict("records"),"training_ids_sha256":sha256(EXP/"data/training_ids.txt"),
            "fold4_total":len(fold4),"calibration":10000,"validation":int((~fold4.is_calibration).sum())}
    con.close(); atomic_json(EXP/"data/cohort_report.json",report); print(json.dumps(report,indent=2))


def build_indexes():
    mkdirs(); con=duckdb.connect(str(DB),read_only=True); con.execute("PRAGMA threads=20"); con.execute("PRAGMA memory_limit='24GB'")
    for scope,folds in [("train","0,1,2,3"),("fold4","4")]:
      for country in ["India","US"]:
       for source in [2,3]:
        root=EXP/f"cache/index/{scope}/{country}/s{source}"; root.mkdir(parents=True,exist_ok=True)
        ap=root/"aux.parquet"
        if not ap.exists():
            aux=con.execute(f"""select a.entity_id,a.business_name,coalesce(a.business_address,'')business_address,a.nname,coalesce(a.naddr,'')naddr,a.country
              from train_s{source} a join read_parquet(?) f on f.source={source} and f.entity_id=a.entity_id
              where a.country=? and f.fold in ({folds}) order by a.entity_id""",[str(ASSIGNMENT),country]).fetchdf()
            nf=aux.nname.value_counts(dropna=False);af=aux.naddr.value_counts(dropna=False)
            aux["candidate_normalized_name_fanout"]=aux.nname.map(nf).clip(0,65535).astype(np.uint16)
            aux["candidate_normalized_address_fanout"]=aux.naddr.map(af).clip(0,65535).astype(np.uint16)
            atomic_parquet(aux,ap)
        else: aux=pd.read_parquet(ap)
        for channel,col in [("name","business_name"),("address","business_address")]:
            mp=root/f"{channel}.npz";ip=root/f"{channel}_idf.npy"
            if mp.exists() and ip.exists(): continue
            raw=e001.vec_for(channel).transform(aux[col].fillna(""));tf=TfidfTransformer();mat=tf.fit_transform(raw).astype(np.float32).tocsr()
            sparse.save_npz(mp,mat,compressed=True);np.save(ip,tf.idf_.astype(np.float32));print("INDEX",scope,country,source,channel,mat.shape,mat.nnz,flush=True)
        del aux;gc.collect()
    con.close()


def load_queries(role,country):
    ids=pd.read_parquet(EXP/f"data/{role}_{country}_queries.parquet")
    con=duckdb.connect(str(DB),read_only=True);con.register("ids",ids[["source1_entity_id"]])
    q=con.execute("""select s.entity_id,s.business_name,coalesce(s.business_address,'')business_address,s.nname,coalesce(s.naddr,'')naddr,s.country
      from train_s1 s join ids i on s.entity_id=i.source1_entity_id order by s.entity_id""").fetchdf();con.close();return q


def retrieve_role(role):
    scope="train" if role=="train" else "fold4"
    for country in ["India","US"]:
      qfull=load_queries(role,country)
      for source in [2,3]:
        root=EXP/f"cache/index/{scope}/{country}/s{source}";aux=pd.read_parquet(root/"aux.parquet");idx={};idfs={}
        for ch in ["name","address"]: idx[ch]=FrozenSelectiveIndex(sparse.load_npz(root/f"{ch}.npz"));idfs[ch]=np.load(root/f"{ch}_idf.npy")
        apos=pd.Series(np.arange(len(aux),dtype=np.int32),index=aux.entity_id)
        con=duckdb.connect(str(DB),read_only=True)
        for part,start in enumerate(range(0,len(qfull),SHARD)):
            op=EXP/f"cache/blocker/{role}/{country}/s{source}/part-{part:05d}.parquet"
            if op.exists(): continue
            t0=time.time();q=qfull.iloc[start:start+SHARD].reset_index(drop=True);frames=[];mats={};kernel=0
            for ch in ["name","address"]:
                mats[ch]=qmatrix(q,ch,idfs[ch]);oi,ov,sec=idx[ch].query(mats[ch]);kernel+=sec;frames.append(stream(oi,ov,ch))
            cand=frames[0].merge(frames[1],on=["qpos","apos"],how="outer")
            cand["s1"]=q.entity_id.to_numpy()[cand.qpos];cand["target_id"]=aux.entity_id.to_numpy()[cand.apos]
            # All training positives are retained even when the blocker misses them.
            if role=="train":
                qids=q[["entity_id"]].rename(columns={"entity_id":"s1"});con.register("qids",qids)
                truth=con.execute("select e.s1,e.target_id from edges e join qids q using(s1) where target_source=?",[source]).fetchdf()
                existing=pd.MultiIndex.from_frame(cand[["s1","target_id"]]); missing=truth[~pd.MultiIndex.from_frame(truth).isin(existing)].copy()
                if len(missing):
                    qpos=pd.Series(np.arange(len(q),dtype=np.int32),index=q.entity_id)
                    missing["qpos"]=missing.s1.map(qpos).astype(np.int32);missing["apos"]=missing.target_id.map(apos)
                    if missing.apos.isna().any(): raise RuntimeError("truth auxiliary outside training fold index")
                    missing["apos"]=missing.apos.astype(np.int32);cand=pd.concat([cand,missing],ignore_index=True,sort=False)
            for ch in ["name","address"]: cand[f"{ch}_char_tfidf"]=e001.row_cosines(mats[ch],idx[ch].a,cand.qpos.to_numpy(),cand.apos.to_numpy())
            atomic_parquet(cand,op);atomic_json(op.with_suffix(".meta.json"),{"role":role,"country":country,"source":source,"part":part,"s1":len(q),"pairs":len(cand),"kernel_seconds":kernel,"seconds":time.time()-t0})
            print("BLOCKER",role,country,source,part,len(cand),flush=True)
        con.close();del aux,idx;qfull=qfull;gc.collect()


def feature_role(role):
    scope="train" if role=="train" else "fold4"; hist=lgb.Booster(model_file=str(E001/"model/lightgbm_model.txt"))
    con=duckdb.connect(str(DB),read_only=True)
    calset=set(pd.read_parquet(EXP/"data/fold4_s1.parquet").query("is_calibration").source1_entity_id) if role=="fold4" else None
    for country in ["India","US"]:
      qfull=load_queries(role,country)
      for source in [2,3]:
        aux=pd.read_parquet(EXP/f"cache/index/{scope}/{country}/s{source}/aux.parquet");nf=aux.nname.value_counts(dropna=False);af=aux.naddr.value_counts(dropna=False)
        files=sorted((EXP/f"cache/blocker/{role}/{country}/s{source}").glob("part-*.parquet"))
        for ip in files:
            op=EXP/f"cache/features/{role}/{country}/s{source}/{ip.name}"
            if op.exists(): continue
            part=int(ip.stem.split("-")[-1]);q=qfull.iloc[part*SHARD:(part+1)*SHARD].reset_index(drop=True);cand=pd.read_parquet(ip)
            cand["source"]=np.int8(source);cand["country"]=country;cand["role"]=role
            qids=q[["entity_id"]].rename(columns={"entity_id":"s1"})
            if role=="fold4": qids=qids[qids.s1.isin(calset)]
            con.register("qids",qids)
            truth=con.execute("select e.s1,e.target_id from edges e join qids q using(s1) where target_source=?",[source]).fetchdf();truthset=set(map(tuple,truth.itertuples(index=False,name=None)))
            if role=="fold4":
                cand["label"]=np.fromiter(((int((a,b) in truthset) if a in calset else -1) for a,b in cand[["s1","target_id"]].itertuples(index=False,name=None)),np.int8,len(cand))
            else:
                cand["label"]=np.fromiter(((a,b) in truthset for a,b in cand[["s1","target_id"]].itertuples(index=False,name=None)),np.int8,len(cand))
            cand["s1_normalized_name_fanout"]=q.nname.map(nf).fillna(0).clip(0,65535).astype(np.uint16).to_numpy()[cand.qpos]
            cand["s1_normalized_address_fanout"]=q.naddr.map(af).fillna(0).clip(0,65535).astype(np.uint16).to_numpy()[cand.qpos]
            cand["candidate_normalized_name_fanout"]=aux.candidate_normalized_name_fanout.to_numpy()[cand.apos]
            cand["candidate_normalized_address_fanout"]=aux.candidate_normalized_address_fanout.to_numpy()[cand.apos]
            qn=q.nname.to_numpy()[cand.qpos];qa=q.naddr.to_numpy()[cand.qpos];an=aux.nname.to_numpy()[cand.apos];aa=aux.naddr.to_numpy()[cand.apos]
            cand["nname_exact"]=((qn!="")&(qn==an)).astype(np.int8);cand["naddr_exact"]=((qa!="")&(qa==aa)).astype(np.int8);cand=add_context(cand)
            feat=build_features(cand,q,aux,workers=20);feat["historical_lgbm"]=hist.predict(feat[FEATURES],num_threads=20).astype(np.float32)
            atomic_parquet(feat,op);print("FEATURE",role,country,source,part,len(feat),int(feat.label.sum()),flush=True)
            del cand,feat;gc.collect()
    con.close()


def main():
    ap=argparse.ArgumentParser();ap.add_argument("stage",choices=["prepare","index","retrieve-train","features-train","retrieve-fold4","features-fold4"]);a=ap.parse_args()
    {"prepare":prepare_ids,"index":build_indexes,"retrieve-train":lambda:retrieve_role("train"),"features-train":lambda:feature_role("train"),
     "retrieve-fold4":lambda:retrieve_role("fold4"),"features-fold4":lambda:feature_role("fold4")}[a.stage]()


if __name__=="__main__": main()
