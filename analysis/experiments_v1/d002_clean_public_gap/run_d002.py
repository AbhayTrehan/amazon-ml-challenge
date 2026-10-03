#!/usr/bin/env python3
"""Frozen P002 evaluation on fold-0 excluding the contaminated historical 10k."""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "analysis/experiments_v1/d002_clean_public_gap"
E001 = ROOT / "analysis/experiments_v1/e001_classical_baseline"
E004 = ROOT / "analysis/experiments_v1/e004_k10_decode_and_gpu_bundle"
E005 = ROOT / "analysis/experiments_v1/e005_xlmr_crossencoder"
R006 = ROOT / "analysis/experiments_v1/r006a_selective_postings"
R007 = ROOT / "analysis/experiments_v1/r007_fast_blocker_e005b"
P002 = ROOT / "analysis/production/p002_final"
DB = ROOT / "analysis/data_audit_v1/artifacts/audit.duckdb"
SPLIT = ROOT / "analysis/data_audit_v1/artifacts/proposed_validation_split.tsv"
ASSIGNMENT = E001 / "cache/auxiliary_fold_assignment.parquet"
INDEX = R006 / "artifacts/indexes"
MODEL = E001 / "model/lightgbm_model.txt"
MODEL_CONFIG = E001 / "model/model_config.json"
SHARD = 20_000
COUNTRIES = ["India", "US"]
KEY = ["source1_entity_id", "candidate_entity_id", "candidate_source"]

sys.path.insert(0, str(E001 / "scripts"))
import prepare_baseline as e001  # noqa: E402
sys.path.insert(0, str(R006))
import run_r006a, run_r006b  # noqa: E402
sys.path.insert(0, str(P002 / "src"))
from feature_builder import add_context, build_features  # noqa: E402


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=float) + "\n")
    tmp.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def append_log(name: str, row) -> None:
    p = OUT / "logs" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(row, default=float) + "\n")


def init() -> None:
    start = time.time()
    for d in ["cache/queries", "cache/aux", "cache/blocker", "cache/lightgbm",
              "cache/shortlist", "cache/neural", "output", "logs", "runtime"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)
    split = pd.read_csv(SPLIT, sep="\t")
    old = pd.read_csv(E001 / "selected_ids/evaluation_s1.tsv", sep="\t")
    clean = split[split.fold.eq(0) & ~split.source1_entity_id.isin(old.source1_entity_id)].copy()
    clean.sort_values("source1_entity_id", kind="mergesort").to_csv(
        OUT / "clean_eval_ids.tsv", sep="\t", index=False
    )
    if len(clean) != 431_284 or not clean.source1_entity_id.is_unique:
        raise RuntimeError(f"clean cohort contract failed: {len(clean)}")

    con = duckdb.connect(str(DB), read_only=True)
    con.register("clean", clean[["source1_entity_id"]])
    checks = {}
    specs = [
        ("lightgbm_train", str(E001 / "cache/features/train_*.parquet"), "s1"),
        ("xlmr_train", str(E004 / "gpu_bundle/train_pairs.parquet"), "source1_entity_id"),
        ("lightgbm_calibration", str(E001 / "predictions/calibration_pair_scores.parquet"), "s1"),
        ("matcher_calibration", str(E005 / "outputs/calibration_predictions.parquet"), "source1_entity_id"),
        ("r007_evaluation", str(R007 / "evaluation_predictions.parquet"), "source1_entity_id"),
    ]
    for name, path, col in specs:
        checks[name] = int(con.execute(
            f"select count(distinct p.{col}) from read_parquet('{path}') p join clean c on c.source1_entity_id=p.{col}"
        ).fetchone()[0])
    con.close()
    r006_ids = set(old.source1_entity_id)
    checks["r006a_r006b_selection"] = len(r006_ids & set(clean.source1_entity_id))

    # Static provenance audit of every code/log reference that can select a
    # frozen P002 setting. Broad aggregate EDA is intentionally excluded.
    selection_evidence = {
        "r006a_r006b": "run_r006a.cohort() reads only evaluation_s1.tsv; run_r006b uses that cohort",
        "lightgbm": "train_and_evaluate.py loads only cache/features/train_*.parquet",
        "xlmr": "train_e005.py receives gpu_bundle/train_pairs.parquet",
        "matcher_policy": "evaluate_e005.py calibration mode reads calibration predictions/metadata only",
        "p002": "production.py and xlmr.py hard-code the already frozen settings and perform no search",
        "ambiguity": "No immutable git history exists; audit is based on extant executable code, logs, manifests, and realized artifacts.",
    }
    status = "PASS" if all(v == 0 for v in checks.values()) else "FAIL"
    report = {
        "status": status, "clean_s1": len(clean), "excluded_historical_s1": len(old),
        "overlap_clean_s1": checks, "selection_evidence": selection_evidence,
        "split_sha256": sha256(SPLIT), "clean_ids_sha256": sha256(OUT / "clean_eval_ids.tsv"),
        "frozen": {"posting_budget": 20000, "coarse_m": 800, "top_per_stream": 10,
                   "feature_count": 49, "shortlist": "rank<=5 OR lightgbm>=0.05",
                   "max_length": 104, "fusion_xlmr": .60, "fusion_lgbm": .40,
                   "threshold": .51, "decoder": "one-owner", "owner_margin": .48},
    }
    atomic_json(OUT / "integrity.json", report)
    atomic_json(OUT / "config.json", report["frozen"])
    if status != "PASS":
        raise RuntimeError(f"clean population leakage: {checks}")

    # Query and enriched auxiliary metadata caches. Existing R006 matrices are
    # read-only and their row identity is proven before use.
    con = duckdb.connect(str(DB), read_only=True)
    con.register("clean_ids", clean[["source1_entity_id"]])
    for country in COUNTRIES:
        q = con.execute("""select s.entity_id,s.business_name,coalesce(s.business_address,'') business_address,
                                  s.nname,coalesce(s.naddr,'') naddr,s.country
                           from train_s1 s join clean_ids c on c.source1_entity_id=s.entity_id
                           where s.country=? order by s.entity_id""", [country]).fetchdf()
        atomic_parquet(q, OUT / f"cache/queries/{country}.parquet")
        for source in [2, 3]:
            aux = con.execute(f"""select a.entity_id,a.business_name,coalesce(a.business_address,'') business_address,
                                         a.nname,coalesce(a.naddr,'') naddr,a.country
                                  from train_s{source} a join read_parquet(?) f
                                  on f.entity_id=a.entity_id and f.source={source}
                                  where f.fold=0 and a.country=? order by a.entity_id""",
                              [str(ASSIGNMENT), country]).fetchdf()
            idx_aux = pd.read_parquet(INDEX / country / f"s{source}/aux.parquet", columns=["entity_id"])
            if aux.entity_id.tolist() != idx_aux.entity_id.tolist():
                raise RuntimeError(f"index identity mismatch {country} S{source}")
            nf = aux.nname.value_counts(dropna=False)
            af = aux.naddr.value_counts(dropna=False)
            aux["candidate_normalized_name_fanout"] = aux.nname.map(nf).clip(0, 65535).astype(np.uint16)
            aux["candidate_normalized_address_fanout"] = aux.naddr.map(af).clip(0, 65535).astype(np.uint16)
            atomic_parquet(aux, OUT / f"cache/aux/{country}_s{source}.parquet")
    con.close()
    atomic_json(OUT / "runtime/init.json", {"seconds": time.time() - start, "status": "complete"})
    print(json.dumps(report, indent=2))


class FrozenIndex:
    def __init__(self, matrix):
        self.a = matrix.tocsr(); c = self.a.tocsc()
        self.c_indptr = run_r006a.i32(c.indptr); self.c_indices = run_r006a.i32(c.indices); self.c_data = run_r006a.f32(c.data)
        self.a_indptr = run_r006a.i32(self.a.indptr); self.a_indices = run_r006a.i32(self.a.indices); self.a_data = run_r006a.f32(self.a.data)
        self.fn, _ = run_r006b.load_functions()

    def query(self, xq):
        q = xq.tocsr(); qip=run_r006a.i32(q.indptr);qix=run_r006a.i32(q.indices);qv=run_r006a.f32(q.data)
        nq=q.shape[0];oi=np.full((nq,10),-1,np.int32);ov=np.zeros((nq,10),np.float32)
        visits=np.zeros(nq,np.int64);full=np.zeros(nq,np.int64);counts=np.zeros(nq,np.int32);fallback=np.zeros(nq,np.int32);timing=np.zeros(5,np.float64)
        ip=ctypes.POINTER(ctypes.c_int32);fp=ctypes.POINTER(ctypes.c_float);i64p=ctypes.POINTER(ctypes.c_int64);dp=ctypes.POINTER(ctypes.c_double)
        t=time.perf_counter();rc=self.fn(qip.ctypes.data_as(ip),qix.ctypes.data_as(ip),qv.ctypes.data_as(fp),nq,
            self.c_indptr.ctypes.data_as(ip),self.c_indices.ctypes.data_as(ip),self.c_data.ctypes.data_as(fp),
            self.a_indptr.ctypes.data_as(ip),self.a_indices.ctypes.data_as(ip),self.a_data.ctypes.data_as(fp),self.a.shape[0],
            20000,800,20,oi.ctypes.data_as(ip),ov.ctypes.data_as(fp),visits.ctypes.data_as(i64p),full.ctypes.data_as(i64p),
            counts.ctypes.data_as(ip),fallback.ctypes.data_as(ip),timing.ctypes.data_as(dp))
        if rc: raise RuntimeError(f"kernel rc={rc}")
        return oi,ov,time.perf_counter()-t


def qmatrix(q, channel, idf):
    tf=TfidfTransformer();tf.idf_=idf
    col="business_name" if channel=="name" else "business_address"
    return tf.transform(e001.vec_for(channel).transform(q[col].fillna(""))).astype(np.float32).tocsr()


def stream(inds,scores,channel):
    rr,kk=np.where(inds>=0)
    return pd.DataFrame({"qpos":rr.astype(np.int32),"apos":inds[rr,kk].astype(np.int32),
        f"{channel}_rank":(kk+1).astype(np.int16),f"{channel}_retrieval_score":scores[rr,kk].astype(np.float32)})


def retrieve(worker: int = 0, nworkers: int = 1) -> None:
    start_all=time.time(); total=0
    task_no = 0
    for country in COUNTRIES:
        qfull=pd.read_parquet(OUT/f"cache/queries/{country}.parquet")
        for source in [2,3]:
            base=INDEX/country/f"s{source}"; indexes={};idfs={}
            for channel in ["name","address"]:
                indexes[channel]=FrozenIndex(sparse.load_npz(base/f"{channel}.npz").tocsr())
                idfs[channel]=np.load(base/f"{channel}_idf.npy")
            for part,start in enumerate(range(0,len(qfull),SHARD)):
                mine = task_no % nworkers == worker
                task_no += 1
                if not mine:
                    continue
                op=OUT/f"cache/blocker/{country}/s{source}/part-{part:05d}.parquet";mp=op.with_suffix(".meta.json")
                if op.exists() and mp.exists(): total += len(pd.read_parquet(op,columns=["s1"])); continue
                q=qfull.iloc[start:start+SHARD].reset_index(drop=True);frames=[];mats={};kernel=0;t0=time.time()
                for channel in ["name","address"]:
                    xq=qmatrix(q,channel,idfs[channel]);mats[channel]=xq;oi,ov,sec=indexes[channel].query(xq);kernel+=sec;frames.append(stream(oi,ov,channel))
                cand=frames[0].merge(frames[1],on=["qpos","apos"],how="outer")
                aux=pd.read_parquet(OUT/f"cache/aux/{country}_s{source}.parquet",columns=["entity_id"])
                cand["s1"]=q.entity_id.to_numpy()[cand.qpos];cand["target_id"]=aux.entity_id.to_numpy()[cand.apos]
                for channel in ["name","address"]:
                    cand[f"{channel}_char_tfidf"]=e001.row_cosines(mats[channel],indexes[channel].a,cand.qpos.to_numpy(),cand.apos.to_numpy())
                atomic_parquet(cand,op);meta={"country":country,"source":source,"part":part,"s1":len(q),"pairs":len(cand),"kernel_seconds":kernel,"wall_seconds":time.time()-t0}
                atomic_json(mp,meta);append_log("retrieval.jsonl",meta);total+=len(cand);print("BLOCKER_READY",json.dumps(meta),flush=True)
                del cand,frames,mats,q;gc.collect()
            del indexes;gc.collect()
    atomic_json(OUT/f"runtime/retrieve_worker_{worker}.json",{"seconds":time.time()-start_all,"pairs":total,"worker":worker,"workers":nworkers,"status":"complete"})


def tasks():
    for country in COUNTRIES:
        n=math.ceil(len(pd.read_parquet(OUT/f"cache/queries/{country}.parquet",columns=["entity_id"]))/SHARD)
        for source in [2,3]:
            for part in range(n): yield country,source,part


def feature_worker(worker:int,nworkers:int) -> None:
    start_all=time.time();model=lgb.Booster(model_file=str(MODEL));fs=json.loads(MODEL_CONFIG.read_text())["features"];done=rows=0
    con=duckdb.connect(str(DB),read_only=True)
    current=None;cache=None
    for ti,(country,source,part) in enumerate(tasks()):
        if ti%nworkers!=worker: continue
        op=OUT/f"cache/lightgbm/{country}/s{source}/part-{part:05d}.parquet";mp=op.with_suffix(".meta.json")
        if op.exists() and mp.exists(): continue
        if current!=(country,source):
            qfull=pd.read_parquet(OUT/f"cache/queries/{country}.parquet");aux=pd.read_parquet(OUT/f"cache/aux/{country}_s{source}.parquet")
            nf=aux.nname.value_counts(dropna=False);af=aux.naddr.value_counts(dropna=False);cache=(qfull,aux,nf,af);current=(country,source)
        qfull,aux,nf,af=cache;q=qfull.iloc[part*SHARD:(part+1)*SHARD].reset_index(drop=True);cand=pd.read_parquet(OUT/f"cache/blocker/{country}/s{source}/part-{part:05d}.parquet")
        t0=time.time();cand["source"]=np.int8(source);cand["country"]=country;cand["role"]="clean_evaluation"
        cand["s1_normalized_name_fanout"]=q.nname.map(nf).fillna(0).clip(0,65535).astype(np.uint16).to_numpy()[cand.qpos]
        cand["s1_normalized_address_fanout"]=q.naddr.map(af).fillna(0).clip(0,65535).astype(np.uint16).to_numpy()[cand.qpos]
        cand["candidate_normalized_name_fanout"]=aux.candidate_normalized_name_fanout.to_numpy()[cand.apos]
        cand["candidate_normalized_address_fanout"]=aux.candidate_normalized_address_fanout.to_numpy()[cand.apos]
        qn=q.nname.to_numpy()[cand.qpos];qa=q.naddr.to_numpy()[cand.qpos];an=aux.nname.to_numpy()[cand.apos];aa=aux.naddr.to_numpy()[cand.apos]
        cand["nname_exact"]=((qn!="")&(qn==an)).astype(np.int8);cand["naddr_exact"]=((qa!="")&(qa==aa)).astype(np.int8)
        qids=q[["entity_id"]].rename(columns={"entity_id":"s1"});con.register("qids",qids)
        truth=con.execute("select e.s1,e.target_id from edges e join qids q using(s1) where e.target_source=?",[source]).fetchdf();con.unregister("qids")
        tset=set(map(tuple,truth.itertuples(index=False,name=None)));cand["label"]=np.fromiter(((s,t) in tset for s,t in zip(cand.s1,cand.target_id)),np.int8,len(cand))
        cand=add_context(cand);feat=build_features(cand,q,aux,workers=5);x=feat[fs].to_numpy();nonfinite=(~np.isfinite(x)).sum(axis=1).astype(np.int16)
        scores=model.predict(feat[fs],num_threads=1).astype(np.float32)
        if not np.isfinite(scores).all() or nonfinite.any(): raise RuntimeError("nonfinite frozen features/scores")
        cols=["s1","target_id","source","country","label","name_rank","name_retrieval_score","address_rank","address_retrieval_score","retrieved_by_name","retrieved_by_address"]
        out=cand[cols].copy()
        for diagnostic_col in ["candidate_address_missing","s1_numeric_token_count","candidate_numeric_token_count"]:
            out[diagnostic_col]=feat[diagnostic_col].to_numpy()
        out["lightgbm_score"]=scores;out["feature_nonfinite_count"]=nonfinite;atomic_parquet(out,op)
        meta={"worker":worker,"country":country,"source":source,"part":part,"pairs":len(out),"seconds":time.time()-t0};atomic_json(mp,meta);append_log(f"feature_worker_{worker}.jsonl",meta);done+=1;rows+=len(out);print("FEATURE_READY",json.dumps(meta),flush=True)
    con.close();atomic_json(OUT/f"runtime/feature_worker_{worker}.json",{"seconds":time.time()-start_all,"parts":done,"rows":rows,"status":"complete"})


def shortlist() -> None:
    start_all=time.time();total=0
    for country in COUNTRIES:
        qfull=pd.read_parquet(OUT/f"cache/queries/{country}.parquet").set_index("entity_id")
        aux={s:pd.read_parquet(OUT/f"cache/aux/{country}_s{s}.parquet").set_index("entity_id") for s in [2,3]}
        n=math.ceil(len(qfull)/SHARD)
        for part in range(n):
            op=OUT/f"cache/shortlist/{country}/part-{part:05d}.parquet";mp=op.with_suffix(".meta.json")
            if op.exists() and mp.exists(): total+=len(pd.read_parquet(op,columns=["source1_entity_id"]));continue
            d=pd.concat([pd.read_parquet(OUT/f"cache/lightgbm/{country}/s{s}/part-{part:05d}.parquet") for s in [2,3]],ignore_index=True)
            d=d.sort_values(["s1","lightgbm_score","target_id"],ascending=[True,False,True],kind="mergesort")
            d["lightgbm_rank_within_s1"]=(d.groupby("s1",sort=False).cumcount()+1).astype(np.int16)
            d=d[d.lightgbm_rank_within_s1.le(5)|d.lightgbm_score.ge(.05)].copy()
            d["s1_business_name"]=d.s1.map(qfull.business_name);d["s1_business_address"]=d.s1.map(qfull.business_address).fillna("")
            d["candidate_business_name"]="";d["candidate_business_address"]=""
            for s in [2,3]:
                take=d.source.eq(s);ids=d.loc[take,"target_id"];d.loc[take,"candidate_business_name"]=ids.map(aux[s].business_name).to_numpy();d.loc[take,"candidate_business_address"]=ids.map(aux[s].business_address).fillna("").to_numpy()
            d.rename(columns={"s1":"source1_entity_id","target_id":"candidate_entity_id"},inplace=True);d["candidate_source"]=np.where(d.source.eq(2),"S2","S3");d["role"]="clean_evaluation"
            cols=KEY+["country","role","label","s1_business_name","s1_business_address","candidate_business_name","candidate_business_address","lightgbm_score","lightgbm_rank_within_s1","name_retrieval_score","name_rank","address_retrieval_score","address_rank","feature_nonfinite_count","candidate_address_missing","s1_numeric_token_count","candidate_numeric_token_count"]
            atomic_parquet(d[cols],op);meta={"country":country,"part":part,"rows":len(d)};atomic_json(mp,meta);append_log("shortlist.jsonl",meta);total+=len(d);print("SHORTLIST_READY",json.dumps(meta),flush=True)
    atomic_json(OUT/"runtime/shortlist.json",{"seconds":time.time()-start_all,"rows":total,"status":"complete"})


def f05_array(tp,true,pred):
    den=1.25*tp+.25*(true-tp)+(pred-tp);out=np.divide(1.25*tp,den,out=np.zeros_like(den,dtype=float),where=den>0);out[(true==0)&(pred==0)]=1;return out


def evaluate() -> None:
    start=time.time();tmp=OUT/"cache/ducktmp";tmp.mkdir(parents=True,exist_ok=True);con=duckdb.connect();con.execute("pragma threads=20");con.execute("pragma memory_limit='100GB'");con.execute(f"pragma temp_directory='{tmp}'")
    s=str(OUT/"cache/shortlist/*/*.parquet");n=str(OUT/"cache/neural/*/*.parquet");b=str(OUT/"cache/lightgbm/*/s*/*.parquet")
    con.execute(f"create table scored as select s.*,n.neural_probability,n.encoded_token_length,(.6*n.neural_probability+.4*s.lightgbm_score)::float fused_score from read_parquet('{s}',union_by_name=true)s join read_parquet('{n}',union_by_name=true)n using(source1_entity_id,candidate_entity_id,candidate_source)")
    con.execute("create table owned as select *,row_number()over(partition by candidate_entity_id order by fused_score desc,source1_entity_id) owner_rank,lead(fused_score)over(partition by candidate_entity_id order by fused_score desc,source1_entity_id) second_score from scored")
    con.execute("create table predictions as select *,case when owner_rank=1 then fused_score-coalesce(second_score,0) end owner_margin,(owner_rank=1 and fused_score>=.51 and fused_score-coalesce(second_score,0)>=.48) accepted from owned")
    con.execute(f"copy(select * from predictions order by source1_entity_id,candidate_entity_id)to '{OUT/'output/clean_predictions.parquet'}'(format parquet,compression zstd)")
    ids=pd.read_csv(OUT/"clean_eval_ids.tsv",sep="\t");con.register("ids",ids)
    # Existing cached labels provide blocker and shortlist truth hits.
    per=con.execute(f"""with ba as(select s1,count(*) blocker_pairs,sum(label) blocker_tp from read_parquet('{b}',union_by_name=true)group by s1),
      sh as(select source1_entity_id s1,count(*) shortlist_pairs,sum(label) shortlist_tp from scored group by 1),
      pr as(select source1_entity_id s1,count_if(accepted) pred,sum(case when accepted then label else 0 end)tp from predictions group by 1)
      select i.*,coalesce(ba.blocker_pairs,0)blocker_pairs,coalesce(ba.blocker_tp,0)blocker_tp,
        coalesce(sh.shortlist_pairs,0)shortlist_pairs,coalesce(sh.shortlist_tp,0)shortlist_tp,
        coalesce(pr.pred,0)predicted_edges,coalesce(pr.tp,0)tp
      from ids i left join ba on i.source1_entity_id=ba.s1 left join sh on i.source1_entity_id=sh.s1 left join pr on i.source1_entity_id=pr.s1
      order by i.source1_entity_id""").fetchdf()
    # Repeated name and missing true-candidate-address metadata.
    dbc=duckdb.connect(str(DB),read_only=True);dbc.register("ids",ids[["source1_entity_id"]])
    rep=dbc.execute(f"""with aux as(select a.country,a.nname,count(*)n from train_s2 a join read_parquet('{ASSIGNMENT}')f on f.source=2 and f.entity_id=a.entity_id where f.fold=0 group by 1,2 union all select a.country,a.nname,count(*)n from train_s3 a join read_parquet('{ASSIGNMENT}')f on f.source=3 and f.entity_id=a.entity_id where f.fold=0 group by 1,2), z as(select country,nname,max(n)n from aux group by 1,2)select q.entity_id source1_entity_id,coalesce(z.n>1,false)repeated_name from train_s1 q join ids i on i.source1_entity_id=q.entity_id left join z using(country,nname)""").fetchdf()
    miss=dbc.execute("""select i.source1_entity_id,coalesce(bool_or(coalesce(a.business_address,'')=''),false) missing_true_candidate_address from ids i left join edges e on e.s1=i.source1_entity_id left join(select entity_id,business_address from train_s2 union all select entity_id,business_address from train_s3)a on a.entity_id=e.target_id group by 1""").fetchdf();dbc.close()
    per=per.merge(rep,on="source1_entity_id").merge(miss,on="source1_entity_id");per["f05"]=f05_array(per.tp.to_numpy(),per.true_matches.to_numpy(),per.predicted_edges.to_numpy());per["blocker_oracle_f05"]=f05_array(per.blocker_tp.to_numpy(),per.true_matches.to_numpy(),per.blocker_tp.to_numpy());per["shortlist_oracle_f05"]=f05_array(per.shortlist_tp.to_numpy(),per.true_matches.to_numpy(),per.shortlist_tp.to_numpy());atomic_parquet(per,OUT/"output/per_s1_scores.parquet")
    rows=[]
    def metric(name,sub):
        rows.append({"slice":name,"n_s1":len(sub),"macro_f05":sub.f05.mean(),"precision":sub.tp.sum()/sub.predicted_edges.sum() if sub.predicted_edges.sum() else 1.,"recall":sub.tp.sum()/sub.true_matches.sum() if sub.true_matches.sum() else 1.,"singleton_accuracy":((sub.predicted_edges==0)[sub.true_matches==0].mean() if (sub.true_matches==0).any() else np.nan),"predicted_edges":int(sub.predicted_edges.sum()),"true_edges":int(sub.true_matches.sum())})
    metric("overall",per)
    for country,g in per.groupby("country"):metric(country,g)
    for val,g in per.groupby(per.true_matches.map(lambda x:str(x) if x<5 else "5+")):metric(f"cardinality_{val}",g)
    metric("repeated/common names",per[per.repeated_name]);metric("non-repeated names",per[~per.repeated_name]);metric("true candidate address missing",per[per.missing_true_candidate_address]);metric("true candidate address nonmissing",per[~per.missing_true_candidate_address])
    # Source metrics retain all S1, with source-specific truth and predictions.
    for src,col in [("S2","s2_matches"),("S3","s3_matches")]:
        z=con.execute(f"select source1_entity_id,count_if(accepted)pred,sum(case when accepted then label else 0 end)tp from predictions where candidate_source='{src}' group by 1").fetchdf();x=per[["source1_entity_id",col]].merge(z,on="source1_entity_id",how="left").fillna(0);x["f05"]=f05_array(x.tp.to_numpy(),x[col].to_numpy(),x.pred.to_numpy());rows.append({"slice":src,"n_s1":len(x),"macro_f05":x.f05.mean(),"precision":x.tp.sum()/x.pred.sum(),"recall":x.tp.sum()/x[col].sum(),"singleton_accuracy":np.nan,"predicted_edges":int(x.pred.sum()),"true_edges":int(x[col].sum())})
    pd.DataFrame(rows).to_csv(OUT/"output/slice_metrics.csv",index=False)
    overall=rows[0];se=float(per.f05.std(ddof=1)/np.sqrt(len(per)));summary={**overall,"blocker_pairs":int(per.blocker_pairs.sum()),"blocker_oracle":float(per.blocker_oracle_f05.mean()),"shortlist_pairs":int(per.shortlist_pairs.sum()),"shortlist_oracle":float(per.shortlist_oracle_f05.mean()),"true_edges_per_s1":float(per.true_matches.mean()),"predicted_edges_per_s1":float(per.predicted_edges.mean()),"f05_standard_error":se,"f05_ci95_low":overall["macro_f05"]-1.96*se,"f05_ci95_high":overall["macro_f05"]+1.96*se,"seconds":time.time()-start}
    atomic_json(OUT/"output/clean_results.json",summary);con.close();atomic_json(OUT/"runtime/evaluate.json",{"seconds":time.time()-start,"status":"complete"});print(json.dumps(summary,indent=2))


def main():
    ap=argparse.ArgumentParser();ap.add_argument("stage",choices=["init","retrieve","features","shortlist","evaluate"]);ap.add_argument("--worker",type=int,default=0);ap.add_argument("--workers",type=int,default=1);a=ap.parse_args()
    {"init":init,"retrieve":lambda:retrieve(a.worker,a.workers),"features":lambda:feature_worker(a.worker,a.workers),"shortlist":shortlist,"evaluate":evaluate}[a.stage]()


if __name__=="__main__":main()
