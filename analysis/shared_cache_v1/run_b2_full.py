#!/usr/bin/env python3
"""Frozen-B2 full-scale candidate materialization and validation gate.

B2 is immutable here: four forward R006B streams at K20, union four exact
reverse streams at K1, always within literal country equality.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
import resource
import shutil
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
from sparse_dot_topn import sp_matmul_topn

ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "analysis/shared_cache_v1"
MANIFESTS = SHARED / "manifests"
INDEXES = SHARED / "indexes"
CANDIDATES = SHARED / "candidates"
REPORT = SHARED / "reports/b2_full_lgbm"
DB = ROOT / "analysis/data_audit_v1/artifacts/audit.duckdb"
SPLIT = ROOT / "analysis/data_audit_v1/artifacts/proposed_validation_split.tsv"
ASSIGNMENT = ROOT / "analysis/experiments_v1/e001_classical_baseline/cache/auxiliary_fold_assignment.parquet"
E001 = ROOT / "analysis/experiments_v1/e001_classical_baseline"
R006 = ROOT / "analysis/experiments_v1/r006a_selective_postings"
P001 = ROOT / "analysis/production/p001_e005b"
VAL_INDEX = R006 / "artifacts/indexes"
TEST_INDEX = P001 / "cache/indexes"
OLD_IDS = E001 / "selected_ids/evaluation_s1.tsv"
VAL_IDS = MANIFESTS / "clean_validation_s1.tsv"
THREADS = 20
FORWARD_K = 20
REVERSE_K = 1
POSTING_BUDGET = 20_000
COARSE_M = 800
QUERY_SHARD = 20_000
AUX_SHARD = 50_000
ROW_SHARD = 3_000_000
COUNTRY_CODE = {"India": 0, "US": 1, "France": 2}
POP_COUNTRIES = {"development_train": ["India", "US"], "clean_validation": ["India", "US"], "test": ["France", "India", "US"]}
PROC = psutil.Process()

sys.path.insert(0, str(SHARED))
from cache_utils import atomic_json, complete_manifest, content_key, sha256_file, validate_complete  # noqa: E402
sys.path.insert(0, str(E001 / "scripts"))
import prepare_baseline as e001  # noqa: E402
sys.path.insert(0, str(R006))
import run_r006a  # noqa: E402


def log(event: str, **values) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {event} " + json.dumps(values, sort_keys=True, default=float), flush=True)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def frame_hash(frame: pd.DataFrame, column: str) -> str:
    digest = hashlib.sha256()
    for value in frame[column].astype(str): digest.update(value.encode()); digest.update(b"\n")
    return digest.hexdigest()


def populations() -> dict[str, dict]:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    split = pd.read_csv(SPLIT, sep="\t", dtype={"source1_entity_id": str})
    clean = pd.read_csv(VAL_IDS, sep="\t", dtype={"source1_entity_id": str})
    if len(clean) != 431_284 or sha256_file(VAL_IDS) != "5084362ccd992845ddf65508293552b2919b8a4f1c13190e82e7bf026ee5e74f":
        raise RuntimeError("frozen clean-validation identity failed")
    dev = split[~split.source1_entity_id.isin(clean.source1_entity_id)].copy()
    if len(dev) != 1_775_537 or not dev.source1_entity_id.is_unique: raise RuntimeError(f"development population mismatch {len(dev)}")
    con = duckdb.connect(str(DB), read_only=True)
    test = con.execute("select entity_id source1_entity_id,country from test_s1 order by entity_id").fetchdf(); con.close()
    test["true_matches"] = np.int16(-1); test["s2_matches"] = np.int16(-1); test["s3_matches"] = np.int16(-1); test["fold"] = np.int8(-1)
    result = {}
    for name, frame in [("clean_validation", clean), ("development_train", dev), ("test", test)]:
        frame = frame.sort_values("source1_entity_id", kind="mergesort").reset_index(drop=True)
        path = MANIFESTS / f"b2_{name}_s1.parquet"; atomic_parquet(frame, path)
        identity = {"kind": "b2_population", "population": name, "logical_ids_sha256": frame_hash(frame,"source1_entity_id"),
                    "canonical_split_sha256": sha256_file(SPLIT) if name != "test" else None,
                    "clean_exclusion_sha256": sha256_file(VAL_IDS) if name == "development_train" else None,
                    "rule": "all train S1 not in authoritative clean validation" if name == "development_train" else ("authoritative D002 clean validation" if name == "clean_validation" else "all test S1")}
        manifest = complete_manifest(MANIFESTS/f"b2_{name}_s1.manifest.json", identity, [path],
            {"created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"count":len(frame),"schema":{c:str(t) for c,t in frame.dtypes.items()}})
        result[name] = {"frame":frame,"path":path,"manifest":manifest}
    # Repository fact: components are one-S1 stars. Prove old contaminated 10k
    # share no true leaf with clean validation before including them in dev.
    dbc=duckdb.connect(str(DB),read_only=True)
    overlap=dbc.execute(f"""with o as(select source1_entity_id s1 from read_csv_auto('{OLD_IDS}',delim='\\t',header=true)),
      c as(select source1_entity_id s1 from read_csv_auto('{VAL_IDS}',delim='\\t',header=true)),
      ot as(select target_id from edges join o using(s1)),ct as(select target_id from edges join c using(s1))
      select count(*) from(select * from ot intersect select * from ct)""").fetchone()[0]; dbc.close()
    if overlap: raise RuntimeError(f"historical 10k component overlap with clean validation: {overlap}")
    return result


def query_map(population: str, country: str, info: dict) -> Path:
    base = INDEXES / "b2_query_maps" / population; path = base / f"{country}.parquet"
    if path.exists(): return path
    ids = info["frame"].loc[info["frame"].country.eq(country), ["source1_entity_id"]].copy()
    con=duckdb.connect(str(DB),read_only=True); con.register("ids",ids); prefix="test" if population=="test" else "train"
    q=con.execute(f"""select row_number()over(order by s.entity_id)-1 qrow,s.entity_id,
      s.business_name,coalesce(s.business_address,'')business_address,s.nname,coalesce(s.naddr,'')naddr,s.country
      from {prefix}_s1 s join ids i on s.entity_id=i.source1_entity_id order by s.entity_id""").fetchdf(); con.close()
    q["qrow"]=q.qrow.astype(np.int32)
    if len(q)!=len(ids): raise RuntimeError(f"query map mismatch {population} {country}")
    atomic_parquet(q,path); return path


def build_dev_aux(country: str, source: int) -> Path:
    base=INDEXES/"b2_development_aux"/country/f"s{source}"; path=base/"aux.parquet"
    if path.exists(): return path
    con=duckdb.connect(str(DB),read_only=True)
    con.execute("pragma threads=20"); con.execute("pragma memory_limit='100GB'")
    clean=str(VAL_IDS); assignment=str(ASSIGNMENT)
    aux=con.execute(f"""with clean as(select source1_entity_id s1 from read_csv_auto('{clean}',delim='\\t',header=true)),
      dev as(select entity_id s1 from train_s1 anti join clean on entity_id=clean.s1),
      dev_targets as(select distinct e.target_id from edges e join dev using(s1) where e.target_source={source}),
      all_targets as(select distinct target_id from edges where target_source={source})
      select row_number()over(order by a.entity_id)-1 apos,a.entity_id,a.business_name,coalesce(a.business_address,'')business_address,
        a.nname,coalesce(a.naddr,'')naddr,a.country
      from train_s{source} a join read_parquet('{assignment}') f on f.source={source}and f.entity_id=a.entity_id
      left join dev_targets d on d.target_id=a.entity_id left join all_targets t on t.target_id=a.entity_id
      where a.country=? and(d.target_id is not null or(t.target_id is null and f.fold<>0)) order by a.entity_id""",[country]).fetchdf(); con.close()
    aux["apos"]=aux.apos.astype(np.int32); nf=aux.nname.value_counts(dropna=False);af=aux.naddr.value_counts(dropna=False)
    aux["candidate_normalized_name_fanout"]=aux.nname.map(nf).clip(0,65535).astype(np.uint16)
    aux["candidate_normalized_address_fanout"]=aux.naddr.map(af).clip(0,65535).astype(np.uint16)
    atomic_parquet(aux,path); return path


def forward_index(population: str, country: str, source: int) -> Path:
    if population=="clean_validation": return VAL_INDEX/country/f"s{source}"
    if population=="test": return TEST_INDEX/country/f"s{source}"
    aux_path=build_dev_aux(country,source); base=aux_path.parent
    aux=pd.read_parquet(aux_path)
    for channel,column in [("name","business_name"),("address","business_address")]:
        matrix_path=base/f"{channel}.npz"; idf_path=base/f"{channel}_idf.npy"
        if matrix_path.exists() and idf_path.exists(): continue
        started=time.perf_counter(); raw=e001.vec_for(channel).transform(aux[column].fillna("")); tf=TfidfTransformer();matrix=tf.fit_transform(raw).astype(np.float32).tocsr()
        tmp=matrix_path.with_name(matrix_path.name+f".tmp-{os.getpid()}.npz");sparse.save_npz(tmp,matrix,compressed=False);tmp.replace(matrix_path)
        tmpi=idf_path.with_name(idf_path.name+f".tmp-{os.getpid()}");
        with tmpi.open("wb") as handle:np.save(handle,tf.idf_.astype(np.float32))
        tmpi.replace(idf_path);log("DEV_FORWARD_INDEX_READY",country=country,source=source,channel=channel,rows=len(aux),seconds=time.perf_counter()-started)
        del raw,matrix;gc.collect()
    return base


def normalized_aux(population: str,country: str,source: int,base: Path) -> pd.DataFrame:
    aux=pd.read_parquet(base/"aux.parquet")
    if "apos" not in aux: aux.insert(0,"apos",np.arange(len(aux),dtype=np.int32))
    if "nname" not in aux.columns:
        con=duckdb.connect(str(DB),read_only=True);con.register("ids",aux[["entity_id"]]);prefix="test" if population=="test" else "train"
        z=con.execute(f"select a.entity_id,a.nname,coalesce(a.naddr,'')naddr,a.country from {prefix}_s{source} a join ids using(entity_id)").fetchdf();con.close();aux=aux.merge(z,on="entity_id",how="left",validate="one_to_one")
    if "country" not in aux: aux["country"]=country
    if "candidate_normalized_name_fanout" not in aux:
        aux["candidate_normalized_name_fanout"]=aux.nname.map(aux.nname.value_counts(dropna=False)).clip(0,65535).astype(np.uint16)
        aux["candidate_normalized_address_fanout"]=aux.naddr.map(aux.naddr.value_counts(dropna=False)).clip(0,65535).astype(np.uint16)
    return aux


def register_indexes(population: str, info: dict) -> tuple[dict,dict]:
    forward={};reverse={}
    for country in POP_COUNTRIES[population]:
        qpath=query_map(population,country,info);q=pd.read_parquet(qpath)
        for source in [2,3]:
            base=forward_index(population,country,source);aux=normalized_aux(population,country,source,base)
            # Exact identity against source tables and literal country equality.
            if not aux.country.eq(country).all() or not aux.entity_id.is_unique:raise RuntimeError(f"aux identity/country failure {population} {country} s{source}")
            meta=INDEXES/"b2_forward_refs"/population/country/f"s{source}.manifest.json"
            artifacts=[base/"aux.parquet",base/"name.npz",base/"name_idf.npy",base/"address.npz",base/"address_idf.npy"]
            identity={"kind":"b2_forward_index","population_key":info["manifest"]["content_key"],"population":population,"country":country,"source":source,
              "aux_ids_sha256":frame_hash(aux,"entity_id"),"vectorizer":"E001 frozen hashing TFIDF","files":{p.name:sha256_file(p) for p in artifacts}}
            forward[f"{country}_s{source}"]=complete_manifest(meta,identity,artifacts,{"storage":"in_place_reference","row_count":len(aux)})["content_key"]
        for channel,column in [("name","business_name"),("address","business_address")]:
            identity={"kind":"b2_reverse_s1_index","population_key":info["manifest"]["content_key"],"country":country,"channel":channel,"vectorizer":"E001 frozen hashing TFIDF"}
            key=content_key(identity);base=INDEXES/key;manifest_path=base/"manifest.json";cached=validate_complete(manifest_path)
            if not cached:
                base.mkdir(parents=True,exist_ok=True);raw=e001.vec_for(channel).transform(q[column].fillna(""));tf=TfidfTransformer();matrix=tf.fit_transform(raw).astype(np.float32).tocsr()
                cp=base/"corpus.parquet";mp=base/"matrix.npz";ip=base/"idf.npy";atomic_parquet(q[["qrow","entity_id","country"]],cp)
                tmp=mp.with_name(mp.name+f".tmp-{os.getpid()}.npz");sparse.save_npz(tmp,matrix,compressed=False);tmp.replace(mp)
                tmpi=ip.with_name(ip.name+f".tmp-{os.getpid()}");
                with tmpi.open("wb") as handle:np.save(handle,tf.idf_.astype(np.float32))
                tmpi.replace(ip);cached=complete_manifest(manifest_path,identity,[cp,mp,ip],{"row_count":len(q),"schema":"CSR float32"})
                log("REVERSE_INDEX_READY",population=population,country=country,channel=channel,rows=len(q))
            reverse[f"{country}_{channel}"]=key
    return forward,reverse


def qmatrix(frame,channel,idf):
    tf=TfidfTransformer();tf.idf_=idf;column="business_name" if channel=="name" else "business_address"
    return tf.transform(e001.vec_for(channel).transform(frame[column].fillna(""))).astype(np.float32).tocsr()


def load_kernel():
    lib=ctypes.CDLL(str(run_r006a.compile_kernel()));fn=lib.selective_postings_topk
    ip=ctypes.POINTER(ctypes.c_int32);fp=ctypes.POINTER(ctypes.c_float);i64=ctypes.POINTER(ctypes.c_int64);dp=ctypes.POINTER(ctypes.c_double)
    fn.argtypes=[ip,ip,fp,ctypes.c_int,ip,ip,fp,ip,ip,fp,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ip,fp,i64,i64,ip,ip,dp];fn.restype=ctypes.c_int
    return fn


def selective(fn,xq,xa):
    q=xq.tocsr();a=xa.tocsr();c=xa.tocsc();arr=[run_r006a.i32(q.indptr),run_r006a.i32(q.indices),run_r006a.f32(q.data),run_r006a.i32(c.indptr),run_r006a.i32(c.indices),run_r006a.f32(c.data),run_r006a.i32(a.indptr),run_r006a.i32(a.indices),run_r006a.f32(a.data)]
    n=len(q.indptr)-1;oi=np.full((n,FORWARD_K),-1,np.int32);ov=np.zeros((n,FORWARD_K),np.float32);vis=np.zeros(n,np.int64);full=np.zeros(n,np.int64);counts=np.zeros(n,np.int32);fallback=np.zeros(n,np.int32);timing=np.zeros(5,np.float64)
    ip=ctypes.POINTER(ctypes.c_int32);fp=ctypes.POINTER(ctypes.c_float);i64=ctypes.POINTER(ctypes.c_int64)
    started=time.perf_counter();rc=fn(arr[0].ctypes.data_as(ip),arr[1].ctypes.data_as(ip),arr[2].ctypes.data_as(fp),n,arr[3].ctypes.data_as(ip),arr[4].ctypes.data_as(ip),arr[5].ctypes.data_as(fp),arr[6].ctypes.data_as(ip),arr[7].ctypes.data_as(ip),arr[8].ctypes.data_as(fp),a.shape[0],POSTING_BUDGET,COARSE_M,THREADS,FORWARD_K,oi.ctypes.data_as(ip),ov.ctypes.data_as(fp),vis.ctypes.data_as(i64),full.ctypes.data_as(i64),counts.ctypes.data_as(ip),fallback.ctypes.data_as(ip),timing.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    if rc:raise RuntimeError(f"kernel failed {rc}")
    return oi,ov,time.perf_counter()-started


def raw_root(population: str, info: dict, forward_keys:dict,reverse_keys:dict) -> tuple[Path,dict]:
    identity={"kind":"b2_raw_retrieval","population_key":info["manifest"]["content_key"],"forward_index_keys":forward_keys,"reverse_index_keys":reverse_keys,
      "b2":{"forward_k":20,"reverse_k":1,"posting_budget":20000,"coarse_m":800,"country_equality":"literal string"},"kernel_sha256":sha256_file(R006/"selective_postings.cpp")}
    key=content_key(identity);return CANDIDATES/key,identity


def retrieve(population: str,info:dict,forward_keys:dict,reverse_keys:dict) -> tuple[Path,dict]:
    base,identity=raw_root(population,info,forward_keys,reverse_keys);manifest_path=base/"raw.manifest.json";cached=validate_complete(manifest_path)
    if cached:log("RAW_RETRIEVAL_CACHE_HIT",population=population,key=cached["content_key"]);return base,cached
    fn=load_kernel();runtime=[];started_all=time.perf_counter()
    for country in POP_COUNTRIES[population]:
        q=pd.read_parquet(query_map(population,country,info));qids=q.qrow.to_numpy(np.int32)
        for source in [2,3]:
            idx=forward_index(population,country,source);aux=normalized_aux(population,country,source,idx);apos=aux.apos.to_numpy(np.int32)
            for channel in ["name","address"]:
                xa=sparse.load_npz(idx/f"{channel}.npz").tocsr();idf=np.load(idx/f"{channel}_idf.npy")
                for part,start in enumerate(range(0,len(q),QUERY_SHARD)):
                    out=base/"forward"/country/f"s{source}_{channel}_{part:05d}.parquet";meta=out.with_suffix(".meta.json")
                    if out.exists() and meta.exists():continue
                    block=q.iloc[start:start+QUERY_SHARD];t=time.perf_counter();xq=qmatrix(block,channel,idf);oi,ov,ksec=selective(fn,xq,xa);rr,kk=np.where(oi>=0)
                    frame=pd.DataFrame({"country_code":np.int8(COUNTRY_CODE[country]),"qrow":qids[start:start+len(block)][rr],"source":np.int8(source),"apos":apos[oi[rr,kk]],"channel":np.int8(0 if channel=="name" else 1),"rank":(kk+1).astype(np.int16),"score":ov[rr,kk].astype(np.float32)})
                    atomic_parquet(frame,out);row={"direction":"forward","country":country,"source":source,"channel":channel,"part":part,"queries":len(block),"rows":len(frame),"kernel_seconds":ksec,"wall_seconds":time.perf_counter()-t};atomic_json(meta,row);runtime.append(row);log("FORWARD_SHARD_READY",population=population,**row)
                    del block,xq,oi,ov,frame;gc.collect()
                del xa;gc.collect()
        # Reverse: each source aux queries the population-specific S1 corpus.
        for channel in ["name","address"]:
            ridx=INDEXES/reverse_keys[f"{country}_{channel}"];corpus=pd.read_parquet(ridx/"corpus.parquet");s1rows=corpus.qrow.to_numpy(np.int32);matrix=sparse.load_npz(ridx/"matrix.npz").tocsr();idf=np.load(ridx/"idf.npy")
            for source in [2,3]:
                idx=forward_index(population,country,source);aux=normalized_aux(population,country,source,idx);apos=aux.apos.to_numpy(np.int32)
                for part,start in enumerate(range(0,len(aux),AUX_SHARD)):
                    out=base/"reverse"/country/f"s{source}_{channel}_{part:05d}.parquet";meta=out.with_suffix(".meta.json")
                    if out.exists() and meta.exists():continue
                    block=aux.iloc[start:start+AUX_SHARD];t=time.perf_counter();xq=qmatrix(block,channel,idf);top=sp_matmul_topn(xq,matrix.T,top_n=1,threshold=1e-8,sort=True,n_threads=THREADS).tocsr();qpos=np.repeat(np.arange(top.shape[0]),np.diff(top.indptr))
                    frame=pd.DataFrame({"country_code":np.int8(COUNTRY_CODE[country]),"qrow":s1rows[top.indices],"source":np.int8(source),"apos":apos[start:start+len(block)][qpos],"channel":np.int8(0 if channel=="name" else 1),"rank":np.int16(1),"score":top.data.astype(np.float32)})
                    atomic_parquet(frame,out);row={"direction":"reverse","country":country,"source":source,"channel":channel,"part":part,"queries":len(block),"rows":len(frame),"wall_seconds":time.perf_counter()-t};atomic_json(meta,row);runtime.append(row);log("REVERSE_SHARD_READY",population=population,**row)
                    del block,xq,top,qpos,frame;gc.collect()
            del matrix;gc.collect()
    runtime_path=base/"raw_runtime.json";atomic_json(runtime_path,{"stages":runtime,"wall_seconds":time.perf_counter()-started_all})
    artifacts=sorted(base.glob("forward/*/*.parquet"))+sorted(base.glob("forward/*/*.meta.json"))+sorted(base.glob("reverse/*/*.parquet"))+sorted(base.glob("reverse/*/*.meta.json"))+[runtime_path]
    manifest=complete_manifest(manifest_path,identity,artifacts,{"population":population,"forward_shards":len(list(base.glob('forward/*/*.parquet'))),"reverse_shards":len(list(base.glob('reverse/*/*.parquet'))),"wall_seconds":time.perf_counter()-started_all})
    return base,manifest


def assemble(population:str,info:dict,raw:Path,raw_manifest:dict) -> tuple[Path,dict]:
    identity={"kind":"b2_candidates","population_key":info["manifest"]["content_key"],"raw_retrieval_key":raw_manifest["content_key"],"definition":"forward K20 union reverse top1","row_shard_target":ROW_SHARD}
    key=content_key(identity);base=CANDIDATES/key;manifest_path=base/"manifest.json";cached=validate_complete(manifest_path)
    if cached:log("B2_CANDIDATE_CACHE_HIT",population=population,key=key);return base,cached
    base.mkdir(parents=True,exist_ok=True);db=base/"assemble.duckdb";
    if db.exists():db.unlink()
    con=duckdb.connect(str(db));con.execute("pragma threads=20");con.execute("pragma memory_limit='100GB'");started=time.perf_counter()
    fg=str(raw/"forward/*/*.parquet");rg=str(raw/"reverse/*/*.parquet")
    con.execute(f"""create table f as select country_code,qrow,source,apos,
      min(rank)filter(channel=0)name_rank,max(score)filter(channel=0)name_retrieval_score,
      min(rank)filter(channel=1)address_rank,max(score)filter(channel=1)address_retrieval_score
      from read_parquet('{fg}',union_by_name=true)group by all""")
    con.execute(f"""create table r as select country_code,qrow,source,apos,
      max(score)filter(channel=0)reverse_name_score,max(score)filter(channel=1)reverse_address_score
      from read_parquet('{rg}',union_by_name=true)group by all""")
    con.execute("""create table candidates as select coalesce(f.country_code,r.country_code)::tinyint country_code,
      coalesce(f.qrow,r.qrow)::integer qrow,coalesce(f.source,r.source)::tinyint as "source",coalesce(f.apos,r.apos)::integer apos,
      f.name_rank::smallint name_rank,f.name_retrieval_score::float name_retrieval_score,
      f.address_rank::smallint address_rank,f.address_retrieval_score::float address_retrieval_score,
      r.reverse_name_score::float reverse_name_score,r.reverse_address_score::float reverse_address_score,
      ((f.qrow is not null)::tinyint+2*(r.qrow is not null)::tinyint)::tinyint origin_mask
      from f full outer join r using(country_code,qrow,source,apos)""")
    dup=con.execute("select count(*)-count(distinct(country_code,qrow,source,apos))from candidates").fetchone()[0]
    if dup:raise RuntimeError(f"duplicate B2 candidates: {dup}")
    con.execute(f"""create table groups as with g as(select country_code,qrow,count(*)n from candidates group by all),z as(
      select *,sum(n)over(order by country_code,qrow rows unbounded preceding)cum from g)
      select *,floor((cum-n)/{ROW_SHARD})::integer shard_id from z""")
    out_tmp=base/f"shards.tmp-{os.getpid()}";out=base/"shards"
    if out_tmp.exists():shutil.rmtree(out_tmp)
    con.execute("pragma threads=1");con.execute(f"""copy(select c.*,g.shard_id from candidates c join groups g using(country_code,qrow)
      order by shard_id,country_code,qrow,source,apos)to '{out_tmp}'(format parquet,compression zstd,partition_by(shard_id),overwrite_or_ignore)""")
    if out.exists():shutil.rmtree(out)
    out_tmp.replace(out)
    stats=con.execute("""with x as(select country_code,qrow,count(*)n from candidates group by all)
      select count(*)s1,sum(n)pairs,avg(n)mean,median(n)median,quantile_cont(n,.95)p95,quantile_cont(n,.99)p99,
      quantile_cont(n,.999)p999,max(n)maximum from x""").fetchdf().iloc[0].to_dict()
    stats_path=base/"candidate_stats.json";atomic_json(stats_path,stats);con.close()
    shard_files=sorted(out.glob("*/*.parquet"));manifest=complete_manifest(manifest_path,identity,shard_files+[stats_path],{"population":population,"query_count":len(info["frame"]),"row_count":int(stats["pairs"]),"physical_shards":len(shard_files),"measured_seconds":time.perf_counter()-started,"schema":"integer-coded qrow/apos with forward/reverse provenance"})
    log("B2_CANDIDATES_READY",population=population,**stats,seconds=time.perf_counter()-started);return base,manifest


def validation_gate(info:dict,base:Path,manifest:dict) -> dict:
    out=REPORT/"full_b2_validation";out.mkdir(parents=True,exist_ok=True);result_path=out/"metrics.json"
    identity={"kind":"full_b2_validation","candidate_key":manifest["content_key"],"clean_ids_sha256":sha256_file(VAL_IDS),"metric":"official grouped macro F0.5 oracle"}
    completed=validate_complete(out/"manifest.json")
    if completed:return json.loads(result_path.read_text())
    con=duckdb.connect();con.execute("pragma threads=20");con.execute("pragma memory_limit='100GB'")
    cg=str(base/"shards/*/*.parquet");ids=info["frame"].copy();rows=[];per=[]
    for country in POP_COUNTRIES["clean_validation"]:
      q=pd.read_parquet(query_map("clean_validation",country,info),columns=["qrow","entity_id"]);q["country"]=country
      for source in [2,3]:
        aux=normalized_aux("clean_validation",country,source,forward_index("clean_validation",country,source))[["apos","entity_id","business_address"]]
        con.register("qmap",q);con.register("amap",aux)
        hit=con.execute(f"""with c as(select qrow,apos from read_parquet('{cg}')where country_code={COUNTRY_CODE[country]}and source={source}),
          t as(select e.s1,e.target_id from '{DB}'.edges e) select 1""") if False else None
        # Attach audit DB for truth join while keeping candidate integers.
        try:con.execute(f"attach '{DB}' as audit(read_only)")
        except Exception:pass
        z=con.execute(f"""with ca as(select qrow,count(*)candidates from read_parquet('{cg}')where country_code={COUNTRY_CODE[country]}and source={source}group by 1),
          h as(select c.qrow,count(*)tp from read_parquet('{cg}')c join qmap q using(qrow)join amap a using(apos)
            join audit.edges e on e.s1=q.entity_id and e.target_id=a.entity_id and e.target_source={source}
            where c.country_code={COUNTRY_CODE[country]}and c.source={source}group by 1)
          select q.qrow,q.entity_id,coalesce(ca.candidates,0)candidates,coalesce(h.tp,0)tp from qmap q left join ca using(qrow)left join h using(qrow)order by qrow""").fetchdf()
        z["country"]=country;z["source"]=source;per.append(z);con.unregister("qmap");con.unregister("amap")
    p=pd.concat(per,ignore_index=True);wide=p.pivot(index=["entity_id","country"],columns="source",values=["candidates","tp"]).fillna(0);wide.columns=[f"{a}_s{b}" for a,b in wide.columns];wide=wide.reset_index()
    truth=ids.set_index("source1_entity_id");wide["true_matches"]=wide.entity_id.map(truth.true_matches);wide["s2_matches"]=wide.entity_id.map(truth.s2_matches);wide["s3_matches"]=wide.entity_id.map(truth.s3_matches);wide["candidates"]=wide.candidates_s2+wide.candidates_s3;wide["tp"]=wide.tp_s2+wide.tp_s3
    def score(tp,true):
        tp=np.asarray(tp,float);true=np.asarray(true,float);den=1.25*tp+.25*(true-tp);v=np.divide(1.25*tp,den,out=np.zeros_like(tp),where=den>0);v[(true==0)&(tp==0)]=1;return v
    wide["oracle"]=score(wide.tp,wide.true_matches)
    # Missing-address slice matches D002 definition.
    dbc=duckdb.connect(str(DB),read_only=True);dbc.register("ids",ids[["source1_entity_id"]]);missing=dbc.execute("""select i.source1_entity_id,coalesce(bool_or(coalesce(a.business_address,'')=''),false)missing from ids i left join edges e on e.s1=i.source1_entity_id left join(select entity_id,business_address from train_s2 union all select entity_id,business_address from train_s3)a on a.entity_id=e.target_id group by 1""").fetchdf();dbc.close();wide["missing_address"]=wide.entity_id.map(missing.set_index("source1_entity_id").missing).fillna(False)
    def add_slice(name,z,tp="tp",true="true_matches"):
        rows.append({"slice":name,"n_s1":len(z),"candidate_pairs":int(z.candidates.sum()),"mean_candidates":float(z.candidates.mean()),"edge_recall":float(z[tp].sum()/z[true].sum()) if z[true].sum() else 1.,"oracle_macro_f05":float(score(z[tp],z[true]).mean())})
    add_slice("overall",wide)
    for c,z in wide.groupby("country"):add_slice(c,z)
    add_slice("S2",wide,"tp_s2","s2_matches");add_slice("S3",wide,"tp_s3","s3_matches")
    add_slice("true_candidate_address_missing",wide[wide.missing_address]);add_slice("true_candidate_address_present",wide[~wide.missing_address])
    for k in [0,1,2,3,4]:add_slice(f"cardinality_{k}",wide[wide.true_matches.eq(k)])
    add_slice("cardinality_5+",wide[wide.true_matches.ge(5)])
    counts=wide.candidates.sort_values(ascending=False).to_numpy();total=counts.sum()
    heavy={"top_1pct_pair_fraction":float(counts[:math.ceil(len(counts)*.01)].sum()/total),"top_0_1pct_pair_fraction":float(counts[:math.ceil(len(counts)*.001)].sum()/total),"top_0_01pct_pair_fraction":float(counts[:math.ceil(len(counts)*.0001)].sum()/total)}
    for threshold in [500,1000,5000,10000,50000]:heavy[f"s1_gt_{threshold}"]=int((wide.candidates>threshold).sum())
    overall=rows[0];result={"state":"COMPLETE","candidate_key":manifest["content_key"],"s1":len(wide),"total_pairs":int(wide.candidates.sum()),"mean_candidates":float(wide.candidates.mean()),"median_candidates":float(wide.candidates.median()),"p95_candidates":float(wide.candidates.quantile(.95)),"p99_candidates":float(wide.candidates.quantile(.99)),"p999_candidates":float(wide.candidates.quantile(.999)),"max_candidates":int(wide.candidates.max()),"edge_recall":overall["edge_recall"],"oracle_macro_f05":overall["oracle_macro_f05"],"heavy_tail":heavy,"gate_pass":overall["oracle_macro_f05"]>=.995}
    pd.DataFrame(rows).to_csv(out/"slices.csv",index=False);atomic_parquet(wide,out/"per_s1.parquet");atomic_json(result_path,result)
    complete_manifest(out/"manifest.json",identity,[result_path,out/"slices.csv",out/"per_s1.parquet"],{"gate_pass":result["gate_pass"],"count":len(wide)})
    log("FULL_VALIDATION_GATE",**result);return result


def run_population(name:str,info:dict,gate_only=False):
    fkeys,rkeys=register_indexes(name,info);raw,raw_manifest=retrieve(name,info,fkeys,rkeys);base,manifest=assemble(name,info,raw,raw_manifest)
    if name=="clean_validation":return base,manifest,validation_gate(info,base,manifest)
    return base,manifest,None


def main():
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=["validation","materialize-all","population"]);parser.add_argument("--population",choices=list(POP_COUNTRIES));args=parser.parse_args()
    pops=populations();base,manifest,gate=run_population("clean_validation",pops["clean_validation"])
    if not gate["gate_pass"]:raise SystemExit("STOP: full clean B2 oracle below 0.9950")
    if args.stage=="validation":return
    targets=[args.population] if args.stage=="population" else ["development_train","test"]
    for name in targets:
        if name=="clean_validation":continue
        run_population(name,pops[name])


if __name__=="__main__":main()
