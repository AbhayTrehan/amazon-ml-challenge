#!/usr/bin/env python3
"""Final F20_R3 candidate, feature, LightGBM, and XLM-R preparation pipeline.

This program deliberately does not train XLM-R.  Every large stage is resumable;
published cache manifests are content addressed and list checksums for all files.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import gc
import hashlib
import json
import math
import multiprocessing as mp
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import duckdb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from sparse_dot_topn import sp_matmul_topn

ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "analysis/shared_cache_v1"
REPORT = SHARED / "reports/f20_r3_lgbm"
MANIFESTS = SHARED / "manifests"
CANDIDATES = SHARED / "candidates"
FEATURE_ROOT = SHARED / "features"
DB = ROOT / "analysis/data_audit_v1/artifacts/audit.duckdb"
E001 = ROOT / "analysis/experiments_v1/e001_classical_baseline"
E008 = ROOT / "analysis/experiments_v1/e008_final_robust_matcher"
P002 = ROOT / "analysis/production/p002_final"
MODEL_DIR = REPORT / "model"
SCORE_DIR = REPORT / "scores"
XLMR_DIR = REPORT / "xlmr"
THREADS = 20
REVERSE_THREADS = 10
REVERSE_LANES_PER_STREAM = 2
REVERSE_PROCESSES = 4
FORWARD_K = 20
REVERSE_K = 3
ROW_SHARD = 3_000_000
QUERY_SHARD = 20_000
AUX_SHARD = 50_000
COUNTRY_CODE = {"India": 0, "US": 1, "France": 2}
CODE_COUNTRY = {v: k for k, v in COUNTRY_CODE.items()}
COUNTRIES = {"development_train": ["India", "US"], "clean_validation": ["India", "US"], "test": ["France", "India", "US"]}
BLOCKER_ORACLE = 0.9953454862
SHORTLIST_GATE = BLOCKER_ORACLE - 0.0005
VENV_PYTHON = "/home/satvik/venvs/amazonml2026/bin/python"
PROC = psutil.Process()

sys.path.insert(0, str(SHARED))
import run_b2_full as b2  # noqa: E402
from cache_utils import atomic_json, complete_manifest, content_key, sha256_file, validate_complete  # noqa: E402
sys.path.insert(0, str(E001 / "scripts"))
import prepare_baseline as e001  # noqa: E402
sys.path.insert(0, str(P002 / "src"))
from feature_builder import add_context, build_features  # noqa: E402

BASE49 = json.loads((E001 / "model/model_config.json").read_text())["features"]
RC_FEATURES = [
    "reverse_name_rank", "reverse_address_rank", "reverse_name_score", "reverse_address_score",
    "best_forward_rank", "best_reverse_rank", "forward_hit", "reverse_hit", "mutual_hit",
    "retrieval_channel_count", "forward_name_available", "forward_address_available",
    "reverse_name_available", "reverse_address_available",
]
ALL_FEATURES = BASE49 + RC_FEATURES


def log(event: str, **kw) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {event} " + json.dumps(kw, sort_keys=True, default=float), flush=True)


def atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    df.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)


def stable_hash(x: str, salt: str) -> int:
    return int.from_bytes(hashlib.blake2b((salt + "\0" + str(x)).encode(), digest_size=8).digest(), "big")


def rss_gb() -> float:
    return PROC.memory_info().rss / 2**30


def preflight() -> dict:
    REPORT.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(ROOT)
    vm = psutil.virtual_memory()
    # Empirical projections use the complete validation B2 candidate cache and
    # E008 optimized BASE49 feature cache.  A conservative 400M-pair ceiling
    # covers all three F20_R3 populations.
    old_cand = CANDIDATES / "22dbcf8707c8739f3025264446e9734e2db0e3fc4024b023d8da27d4bf382129"
    cand_bytes = sum(p.stat().st_size for p in old_cand.glob("shards/*/*.parquet"))
    cand_rows = 34_430_509
    feat_dir = E008 / "cache/features/train"
    feat_bytes = sum(p.stat().st_size for p in feat_dir.rglob("*.parquet"))
    feat_rows = 7_684_083
    cand_bpr = cand_bytes / cand_rows
    feat_bpr = feat_bytes / feat_rows
    projected_pairs = 400_000_000
    projected = int(projected_pairs * (cand_bpr + feat_bpr) + 80 * 2**30)
    required_with_margin = math.ceil(projected * 1.2)
    cache_sizes = {}
    for p in [SHARED, ROOT / "analysis/production", ROOT / "analysis/experiments_v1"]:
        cache_sizes[str(p.relative_to(ROOT))] = sum(x.stat().st_size for x in p.rglob("*") if x.is_file())
    out = {
        "free_disk_bytes": usage.free, "free_ram_bytes": vm.available, "logical_cpu_threads": os.cpu_count(),
        "cache_sizes_bytes": cache_sizes, "representative_candidate_bytes_per_pair": cand_bpr,
        "representative_base49_feature_bytes_per_pair": feat_bpr,
        "projected_pair_ceiling": projected_pairs, "projected_total_bytes_including_80GiB_artifacts": projected,
        "required_with_20pct_margin_bytes": required_with_margin, "pass": usage.free >= required_with_margin,
        "git_head": None, "git_limitation": ".git is absent/empty; no Git HEAD can be recorded",
    }
    atomic_json(REPORT / "preflight.json", out)
    if not out["pass"]:
        raise SystemExit(f"STOP: disk preflight failed: free={usage.free}, required={required_with_margin}")
    log("PREFLIGHT_PASS", **out)
    return out


def populations() -> dict:
    p = b2.populations()
    expected = {"development_train": 1_775_537, "clean_validation": 431_284, "test": 1_732_544}
    for name, n in expected.items():
        if len(p[name]["frame"]) != n or not p[name]["frame"].source1_entity_id.is_unique:
            raise RuntimeError(f"population invariant failed {name}")
    a = set(p["development_train"]["frame"].source1_entity_id)
    v = set(p["clean_validation"]["frame"].source1_entity_id)
    if a & v:
        raise RuntimeError("development/clean-validation owner overlap")
    return p


def schema_manifest() -> dict:
    parity_path=P002/"diagnostics/feature_parity.json";parity=json.loads(parity_path.read_text())
    if not parity["sample_50000"]["ok"] or not parity["full_r007"]["ok"] or not parity["context"]["ok"]:
        raise SystemExit("STOP: existing optimized BASE49 parity proof failed")
    atomic_json(REPORT/"base49_parity.json",{
        "state":"PASS","evidence_path":str(parity_path),"evidence_sha256":sha256_file(parity_path),
        "optimized_code_sha256":sha256_file(P002/"src/feature_builder.py"),
        "full_rows":parity["full_r007"]["rows"],"feature_order_exact":parity["full_r007"]["feature_order_exact"],
        "discrete_mismatches":parity["full_r007"]["discrete_mismatches"],
        "numeric_global_max_abs":parity["full_r007"]["numeric_global_max_abs"],
        "lightgbm_max_abs":parity["full_r007"]["lightgbm_max_abs"],
        "shortlist_identity_exact":parity["full_r007"]["shortlist_identity_exact"],
    })
    identity = {
        "kind": "f20_r3_feature_schema", "base49_names": BASE49,
        "base49_code_sha256": sha256_file(P002 / "src/feature_builder.py"),
        "base49_schema_sha256": sha256_file(E001 / "model/model_config.json"),
        "retrieval_context_namespace": "RETRIEVAL_CONTEXT_V2", "retrieval_context_names": RC_FEATURES,
        "retrieval_context_semantics": {
            "rank_sentinel": 99, "score_sentinel": 0.0,
            "availability": "one explicit flag per retrieval direction/channel; applies to both its rank and score",
            "channel_count": "number of available direction/channel hits among forward/reverse x name/address",
            "label_free": True,
        },
    }
    key = content_key(identity)
    path = MANIFESTS / "f20_r3_feature_schema.json"
    atomic_json(path, {"state": "COMPLETE", "content_key": key, "identity": identity,
                       "base49_count": len(BASE49), "retrieval_context_count": len(RC_FEATURES), "total_count": len(ALL_FEATURES)})
    return json.loads(path.read_text())


def reverse_stream_worker(args: tuple) -> list[dict]:
    """One exact reverse stream in an isolated process.

    Process isolation matters here: sparse_dot_topn's native worker pool is
    process-global, so concurrent Python threads shared one 20-thread pool and
    left half the logical CPUs idle.
    """
    population, rkeys, base_text, country, channel, source, lane = args
    base=Path(base_text);stream_runtime=[];ridx=b2.INDEXES/rkeys[f"{country}_{channel}"]
    corpus=pd.read_parquet(ridx/"corpus.parquet");s1rows=corpus.qrow.to_numpy(np.int32);matrix=sparse.load_npz(ridx/"matrix.npz").tocsr();idf=np.load(ridx/"idf.npy")
    idx=b2.forward_index(population,country,source);aux=b2.normalized_aux(population,country,source,idx);apos=aux.apos.to_numpy(np.int32)
    for part,start in enumerate(range(0,len(aux),AUX_SHARD)):
        if part % REVERSE_LANES_PER_STREAM != lane:continue
        out=base/"reverse"/country/f"s{source}_{channel}_{part:05d}.parquet";meta=out.with_suffix(".meta.json")
        if out.exists()and meta.exists():stream_runtime.append(json.loads(meta.read_text()));continue
        block=aux.iloc[start:start+AUX_SHARD];t=time.perf_counter();xq=b2.qmatrix(block,channel,idf)
        top=sp_matmul_topn(xq,matrix.T,top_n=REVERSE_K,threshold=1e-8,sort=True,n_threads=REVERSE_THREADS).tocsr();qpos=np.repeat(np.arange(top.shape[0]),np.diff(top.indptr));ranks=np.concatenate([np.arange(1,n+1,dtype=np.int16)for n in np.diff(top.indptr)])
        frame=pd.DataFrame({"country_code":np.int8(COUNTRY_CODE[country]),"qrow":s1rows[top.indices],"source":np.int8(source),"apos":apos[start:start+len(block)][qpos],"channel":np.int8(channel=="address"),"rank":ranks,"score":top.data.astype(np.float32)})
        atomic_parquet(frame,out);row={"direction":"reverse","country":country,"source":source,"channel":channel,"part":part,"queries":len(block),"rows":len(frame),"wall_seconds":time.perf_counter()-t,"parallel_processes":REVERSE_PROCESSES,"lanes_per_stream":REVERSE_LANES_PER_STREAM,"lane":lane,"threads_per_process":REVERSE_THREADS};atomic_json(meta,row);stream_runtime.append(row);log("REVERSE_READY",population=population,**row)
        del block,xq,top,qpos,ranks,frame;gc.collect()
    del matrix,corpus,aux;gc.collect();return stream_runtime


def retrieve_new(population: str, info: dict, fkeys: dict, rkeys: dict) -> tuple[Path, dict]:
    identity = {
        "kind": "f20_r3_raw_retrieval", "population_key": info["manifest"]["content_key"],
        "forward_index_keys": fkeys, "reverse_index_keys": rkeys,
        "definition": "forward K20 plus reverse top3; literal string country equality",
        "forward": {"k": 20, "posting_budget": 20000, "coarse_m": 800},
        "reverse": {"k": 3, "method": "exact sparse_dot_topn", "threshold": 1e-8},
    }
    key = content_key(identity); base = CANDIDATES / key; raw_manifest_path = base / "raw.manifest.json"
    cached = validate_complete(raw_manifest_path)
    if cached:
        log("RAW_CACHE_HIT", population=population, key=key); return base, cached
    fn = b2.load_kernel(); runtime = []; started_all = time.perf_counter()
    for country in COUNTRIES[population]:
        q = pd.read_parquet(b2.query_map(population, country, info)); qids = q.qrow.to_numpy(np.int32)
        for source in [2, 3]:
            idx = b2.forward_index(population, country, source); aux = b2.normalized_aux(population, country, source, idx); apos = aux.apos.to_numpy(np.int32)
            for channel in ["name", "address"]:
                expected=[(base/"forward"/country/f"s{source}_{channel}_{part:05d}.parquet")for part,_ in enumerate(range(0,len(q),QUERY_SHARD))]
                if all(out.exists()and out.with_suffix(".meta.json").exists()for out in expected):continue
                xa = sparse.load_npz(idx / f"{channel}.npz").tocsr(); idf = np.load(idx / f"{channel}_idf.npy")
                for part, start in enumerate(range(0, len(q), QUERY_SHARD)):
                    out = base / "forward" / country / f"s{source}_{channel}_{part:05d}.parquet"; meta = out.with_suffix(".meta.json")
                    if out.exists() and meta.exists(): continue
                    block = q.iloc[start:start+QUERY_SHARD]; t = time.perf_counter(); xq = b2.qmatrix(block, channel, idf); oi, ov, ksec = b2.selective(fn, xq, xa); rr, kk = np.where(oi >= 0)
                    frame = pd.DataFrame({"country_code": np.int8(COUNTRY_CODE[country]), "qrow": qids[start:start+len(block)][rr], "source": np.int8(source), "apos": apos[oi[rr, kk]], "channel": np.int8(channel == "address"), "rank": (kk+1).astype(np.int16), "score": ov[rr, kk].astype(np.float32)})
                    atomic_parquet(frame, out); row = {"direction":"forward","country":country,"source":source,"channel":channel,"part":part,"queries":len(block),"rows":len(frame),"kernel_seconds":ksec,"wall_seconds":time.perf_counter()-t}; atomic_json(meta,row); runtime.append(row); log("FORWARD_READY", population=population, **row)
                    del block, xq, oi, ov, frame; gc.collect()
                del xa; gc.collect()
    # Reverse streams are independent. Separate processes avoid the native
    # library's process-global worker-pool sharing observed with Python threads.
    tasks=[(population,rkeys,str(base),country,channel,source,lane)for country in COUNTRIES[population]for channel in ["name","address"]for source in [2,3]for lane in range(REVERSE_LANES_PER_STREAM)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=REVERSE_PROCESSES,mp_context=mp.get_context("spawn"))as pool:
        futures=[pool.submit(reverse_stream_worker,task)for task in tasks]
        for future in futures:runtime.extend(future.result())
    rp=base/"raw_runtime.json";atomic_json(rp,{"stages":runtime,"wall_seconds":time.perf_counter()-started_all})
    artifacts=sorted(base.glob("forward/*/*.parquet"))+sorted(base.glob("forward/*/*.meta.json"))+sorted(base.glob("reverse/*/*.parquet"))+sorted(base.glob("reverse/*/*.meta.json"))+[rp]
    return base,complete_manifest(raw_manifest_path,identity,artifacts,{"population":population,"wall_seconds":time.perf_counter()-started_all})


def assemble(population: str, info: dict, raw: Path | None, raw_manifest: dict | None) -> tuple[Path, dict]:
    if population == "clean_validation":
        dependency = json.loads((REPORT.parent / "reverse_depth_full_validation/manifest.json").read_text())["content_key"]
    else:
        dependency = raw_manifest["content_key"]
    identity={"kind":"f20_r3_candidates","population":population,"population_key":info["manifest"]["content_key"],"retrieval_dependency":dependency,"definition":"F20_R3","row_shard_target":ROW_SHARD,"country_equality":"literal string"}
    key=content_key(identity);base=CANDIDATES/key;mp=base/"manifest.json";cached=validate_complete(mp)
    if cached: log("CANDIDATE_CACHE_HIT",population=population,key=key);return base,cached
    base.mkdir(parents=True,exist_ok=True);dbp=base/"assemble.duckdb"
    if dbp.exists():dbp.unlink()
    con=duckdb.connect(str(dbp));con.execute("pragma threads=20");con.execute("pragma memory_limit='100GB'");t=time.perf_counter()
    if population == "clean_validation":
        old=str(CANDIDATES/"22dbcf8707c8739f3025264446e9734e2db0e3fc4024b023d8da27d4bf382129"/"shards/*/*.parquet")
        evaldb=str(REPORT.parent/"reverse_depth_full_validation/evaluation.duckdb")
        con.execute(f"attach '{evaldb}' as rev(read_only)")
        con.execute(f"""create table f as select country_code,qrow,source,apos,name_rank,name_retrieval_score,address_rank,address_retrieval_score from read_parquet('{old}') where origin_mask&1=1""")
        con.execute("""create table r as select country_code,qrow,source,apos,min(rank)filter(channel=0) reverse_name_rank,max(score)filter(channel=0) reverse_name_score,min(rank)filter(channel=1) reverse_address_rank,max(score)filter(channel=1) reverse_address_score from rev.normalized_reverse where rank<=3 group by all""")
    else:
        fg=str(raw/"forward/*/*.parquet");rg=str(raw/"reverse/*/*.parquet")
        con.execute(f"""create table f as select country_code,qrow,source,apos,min(rank)filter(channel=0)name_rank,max(score)filter(channel=0)name_retrieval_score,min(rank)filter(channel=1)address_rank,max(score)filter(channel=1)address_retrieval_score from read_parquet('{fg}',union_by_name=true)group by all""")
        con.execute(f"""create table r as select country_code,qrow,source,apos,min(rank)filter(channel=0)reverse_name_rank,max(score)filter(channel=0)reverse_name_score,min(rank)filter(channel=1)reverse_address_rank,max(score)filter(channel=1)reverse_address_score from read_parquet('{rg}',union_by_name=true)where rank<=3 group by all""")
    con.execute("""create table candidates as select coalesce(f.country_code,r.country_code)::tinyint country_code,coalesce(f.qrow,r.qrow)::integer qrow,coalesce(f.source,r.source)::tinyint as "source",coalesce(f.apos,r.apos)::integer apos,f.name_rank::smallint name_rank,f.name_retrieval_score::float name_retrieval_score,f.address_rank::smallint address_rank,f.address_retrieval_score::float address_retrieval_score,r.reverse_name_rank::smallint reverse_name_rank,r.reverse_name_score::float reverse_name_score,r.reverse_address_rank::smallint reverse_address_rank,r.reverse_address_score::float reverse_address_score,((f.qrow is not null)::tinyint+2*(r.qrow is not null)::tinyint)::tinyint origin_mask from f full outer join r using(country_code,qrow,source,apos)""")
    dup=con.execute("select count(*)-count(distinct(country_code,qrow,source,apos))from candidates").fetchone()[0]
    owners=con.execute("select count(distinct(country_code,qrow))from candidates").fetchone()[0]
    if dup or owners!=len(info["frame"]):raise RuntimeError(f"candidate invariant failure duplicates={dup} owners={owners}")
    if population=="clean_validation":
        expected=con.execute("select count(*)from rev.all_pairs where is_old_r1 or reverse_rank<=3").fetchone()[0]
        missing=con.execute("""select count(*)from((select country_code,qrow,source,apos from rev.all_pairs where is_old_r1 or reverse_rank<=3)except(select country_code,qrow,source,apos from candidates))""").fetchone()[0]
        extra=con.execute("""select count(*)from((select country_code,qrow,source,apos from candidates)except(select country_code,qrow,source,apos from rev.all_pairs where is_old_r1 or reverse_rank<=3))""").fetchone()[0]
        if expected!=39_172_900 or missing or extra:raise SystemExit(f"STOP: F20_R3 reproduction failed expected={expected} missing={missing} extra={extra}")
    con.execute(f"""create table groups as with g as(select country_code,qrow,count(*)n from candidates group by all),z as(select *,sum(n)over(order by country_code,qrow rows unbounded preceding)cum from g)select *,floor((cum-n)/{ROW_SHARD})::integer shard_id from z""")
    outtmp=base/f"shards.tmp-{os.getpid()}";out=base/"shards"
    if outtmp.exists():shutil.rmtree(outtmp)
    con.execute("pragma threads=1");con.execute(f"""copy(select c.*,g.shard_id from candidates c join groups g using(country_code,qrow)order by shard_id,country_code,qrow,source,apos)to '{outtmp}'(format parquet,compression zstd,partition_by(shard_id),overwrite_or_ignore)""")
    split_groups=con.execute("select count(*)from(select country_code,qrow,count(distinct shard_id)n from groups group by all having n<>1)").fetchone()[0]
    if split_groups:raise RuntimeError(f"group-preserving sharding failed: {split_groups}")
    if out.exists():shutil.rmtree(out)
    outtmp.replace(out)
    stats=con.execute("""with x as(select country_code,qrow,count(*)n from candidates group by all)select count(*)s1,sum(n)pairs,avg(n)mean,median(n)median,quantile_cont(n,.95)p95,quantile_cont(n,.99)p99,quantile_cont(n,.999)p999,max(n)maximum from x""").fetchdf().iloc[0].to_dict();con.close()
    sp=base/"candidate_stats.json";atomic_json(sp,stats);files=sorted(out.glob("*/*.parquet"));manifest=complete_manifest(mp,identity,files+[sp],{"population":population,"query_count":len(info["frame"]),"row_count":int(stats["pairs"]),"physical_shards":len(files),"seconds":time.perf_counter()-t,"schema":"compact integer pair IDs plus forward/reverse provenance"})
    log("CANDIDATES_READY",population=population,key=key,**stats);return base,manifest


def candidate_artifacts(population: str, info: dict) -> tuple[Path, dict]:
    fkeys,rkeys=b2.register_indexes(population,info)
    if population=="clean_validation": return assemble(population,info,None,None)
    raw,rm=retrieve_new(population,info,fkeys,rkeys);return assemble(population,info,raw,rm)


def retrieval_context(feat: pd.DataFrame) -> pd.DataFrame:
    fn=feat.forward_name_available.to_numpy(dtype=bool);fa=feat.forward_address_available.to_numpy(dtype=bool);rn=feat.reverse_name_rank.notna().to_numpy();ra=feat.reverse_address_rank.notna().to_numpy()
    feat["reverse_name_available"]=rn.astype(np.int8);feat["reverse_address_available"]=ra.astype(np.int8)
    # build_features/add_context has already replaced forward NaNs with BASE49 sentinels.
    feat["reverse_name_rank"]=feat.reverse_name_rank.fillna(99).astype(np.int16);feat["reverse_address_rank"]=feat.reverse_address_rank.fillna(99).astype(np.int16)
    feat["reverse_name_score"]=feat.reverse_name_score.fillna(0).astype(np.float32);feat["reverse_address_score"]=feat.reverse_address_score.fillna(0).astype(np.float32)
    feat["best_forward_rank"]=np.minimum(feat.name_rank,feat.address_rank).astype(np.int16);feat["best_reverse_rank"]=np.minimum(feat.reverse_name_rank,feat.reverse_address_rank).astype(np.int16)
    feat["forward_hit"]=(fn|fa).astype(np.int8);feat["reverse_hit"]=(rn|ra).astype(np.int8);feat["mutual_hit"]=((fn|fa)&(rn|ra)).astype(np.int8);feat["retrieval_channel_count"]=(fn.astype(np.int8)+fa.astype(np.int8)+rn.astype(np.int8)+ra.astype(np.int8)).astype(np.int8)
    return feat


def feature_cache(population:str,info:dict,cbase:Path,cmanifest:dict,schema:dict,base49_only:bool=False)->tuple[Path,dict]:
    selected_features=BASE49 if base49_only else ALL_FEATURES
    identity={"kind":"f20_r3_base49_features" if base49_only else "f20_r3_features","population":population,"candidate_key":cmanifest["content_key"],"base49_code_schema_key":schema["content_key"],"dtype":"historical BASE49 dtypes" if base49_only else "historical BASE49 dtypes; float32 continuous context"}
    if not base49_only:identity["retrieval_context_v2"]=content_key(schema["identity"]["retrieval_context_semantics"])
    key=content_key(identity);base=FEATURE_ROOT/key;mp=base/"manifest.json";cached=validate_complete(mp)
    if cached:log("FEATURE_CACHE_HIT",population=population,key=key);return base,cached
    base.mkdir(parents=True,exist_ok=True);files=sorted((cbase/"shards").glob("*/*.parquet"));stats=[];started_all=time.perf_counter();peak=rss_gb()
    hist=joblib.load(E008/"model/lightgbm/model.joblib")
    data_cache={}
    con=duckdb.connect(str(DB),read_only=True)
    for ordinal,fp in enumerate(files):
        op=base/"shards"/f"part-{ordinal:05d}.parquet";meta=op.with_suffix(".meta.json")
        if op.exists() and meta.exists():stats.append(json.loads(meta.read_text()));continue
        cand=pd.read_parquet(fp);parts=[];t=time.perf_counter()
        for (cc,source),d in cand.groupby(["country_code","source"],sort=True):
            country=CODE_COUNTRY[int(cc)];source=int(source);cachekey=(country,source)
            if cachekey not in data_cache:
                q=pd.read_parquet(b2.query_map(population,country,info));idx=b2.forward_index(population,country,source);aux=b2.normalized_aux(population,country,source,idx)
                data_cache[cachekey]=(q,aux,idx,np.load(idx/"name_idf.npy"),np.load(idx/"address_idf.npy"),sparse.load_npz(idx/"name.npz").tocsr(),sparse.load_npz(idx/"address.npz").tocsr())
            q,aux,idx,nidf,aidf,nmat,amat=data_cache[cachekey];d=d.reset_index(drop=True);qu=np.unique(d.qrow.to_numpy());au=np.unique(d.apos.to_numpy());qs=q.iloc[qu].reset_index(drop=True);aa=aux.iloc[au].reset_index(drop=True)
            d["qpos"]=np.searchsorted(qu,d.qrow.to_numpy()).astype(np.int32);d["apos_global"]=d.apos;d["apos"]=np.searchsorted(au,d.apos.to_numpy()).astype(np.int32);d["source"]=np.int8(source)
            qnm=b2.qmatrix(qs,"name",nidf);qam=b2.qmatrix(qs,"address",aidf);anm=nmat[au];aam=amat[au]
            d["name_char_tfidf"]=e001.row_cosines(qnm,anm,d.qpos.to_numpy(),d.apos.to_numpy());d["address_char_tfidf"]=e001.row_cosines(qam,aam,d.qpos.to_numpy(),d.apos.to_numpy())
            nf=aux.nname.value_counts(dropna=False);af=aux.naddr.value_counts(dropna=False)
            d["s1_normalized_name_fanout"]=qs.nname.map(nf).fillna(0).clip(0,65535).astype(np.uint16).to_numpy()[d.qpos]
            d["s1_normalized_address_fanout"]=qs.naddr.map(af).fillna(0).clip(0,65535).astype(np.uint16).to_numpy()[d.qpos]
            d["candidate_normalized_name_fanout"]=aa.candidate_normalized_name_fanout.to_numpy()[d.apos];d["candidate_normalized_address_fanout"]=aa.candidate_normalized_address_fanout.to_numpy()[d.apos]
            qn=qs.nname.fillna("").to_numpy()[d.qpos];qa=qs.naddr.fillna("").to_numpy()[d.qpos];an=aa.nname.fillna("").to_numpy()[d.apos];ad=aa.naddr.fillna("").to_numpy()[d.apos]
            d["nname_exact"]=((qn!="")&(qn==an)).astype(np.int8);d["naddr_exact"]=((qa!="")&(qa==ad)).astype(np.int8)
            d["forward_name_available"]=d.name_rank.notna().astype(np.int8);d["forward_address_available"]=d.address_rank.notna().astype(np.int8)
            d["country"]=country;d["_priority"]=np.int8(0);d=add_context(d);d=build_features(d,qs,aa,workers=THREADS);d=retrieval_context(d)
            if population!="test":
                qids=pd.DataFrame({"s1":qs.entity_id.to_numpy()[d.qrow.map(pd.Series(np.arange(len(qs)),index=qu)).to_numpy()]}) if False else None
                truth=con.execute("select target_id from edges where 1=0").fetchdf() if False else None
                # Vectorized truth identity via compact global maps.
                pairs=pd.DataFrame({"s1":q.entity_id.to_numpy()[d.qrow.to_numpy()],"target_id":aux.entity_id.to_numpy()[d.apos_global.to_numpy()]})
                con.register("pairs",pairs);lab=con.execute("select exists(select 1 from edges e where e.s1=p.s1 and e.target_id=p.target_id and e.target_source=?) as is_match from pairs p",[source]).fetchnumpy()["is_match"];con.unregister("pairs");d["label"]=lab.astype(np.int8)
            else:d["label"]=np.int8(-1)
            d["historical_miner_score"]=hist.predict_proba(d[BASE49])[:,1].astype(np.float32)
            d["apos"]=d.pop("apos_global").astype(np.int32)
            keep=["country_code","qrow","source","apos","label","historical_miner_score"]+selected_features
            parts.append(d[keep]);del d,qs,aa,qnm,qam,anm,aam;gc.collect()
        feat=pd.concat(parts,ignore_index=True);atomic_parquet(feat,op);peak=max(peak,rss_gb());row={"part":ordinal,"rows":len(feat),"positives":int((feat.label==1).sum()),"seconds":time.perf_counter()-t,"bytes":op.stat().st_size,"peak_rss_gb":peak};stats.append(row);processed=sum(x["rows"] for x in stats);elapsed=time.perf_counter()-started_all;throughput=processed/elapsed;row.update({"processed":processed,"total":int(cmanifest["metadata"]["row_count"]),"elapsed":elapsed,"pairs_per_second":throughput,"eta_seconds":max(0,int(cmanifest["metadata"]["row_count"])-processed)/throughput});atomic_json(meta,row);log("FEATURE_SHARD_READY",population=population,**row);del cand,feat,parts;gc.collect()
    con.close();art=sorted((base/"shards").glob("*"));rows=sum(x["rows"] for x in stats);seconds=sum(x["seconds"] for x in stats);manifest=complete_manifest(mp,identity,art,{"population":population,"row_count":rows,"seconds":seconds,"wall_seconds":time.perf_counter()-started_all,"pairs_per_second":rows/seconds,"peak_rss_gb":peak,"bytes_per_pair":sum(x["bytes"] for x in stats)/rows})
    return base,manifest


def make_internal_split(info:dict)->Path:
    out=REPORT/"internal_lgbm_early_stopping_s1.parquet"
    if out.exists():return out
    d=info["frame"][["source1_entity_id"]].copy();d["hash"]=d.source1_entity_id.map(lambda x:stable_hash(x,"f20-r3-lgbm-internal-50k-20260925"));d=d.nsmallest(50_000,"hash").drop(columns="hash").sort_values("source1_entity_id");atomic_parquet(d,out);return out


def train_lgbm(info:dict,fbase:Path,fmanifest:dict)->dict:
    MODEL_DIR.mkdir(parents=True,exist_ok=True);report_path=MODEL_DIR/"training.json"
    if (MODEL_DIR/"COMPLETE").exists():return json.loads(report_path.read_text())
    internal=make_internal_split(info);glob=str(fbase/"shards/*.parquet");dbp=MODEL_DIR/"training.duckdb"
    if dbp.exists():dbp.unlink()
    con=duckdb.connect(str(dbp));con.execute("pragma threads=20");con.execute("pragma memory_limit='100GB'");t0=time.perf_counter()
    # Strongest historical BASE49 miner selects all reachable positives plus top-5 false rows/S1.
    maps=[]
    for country in COUNTRIES["development_train"]:
        q=b2.query_map("development_train",country,info);maps.append(f"select {COUNTRY_CODE[country]}::tinyint country_code,qrow,entity_id s1 from read_parquet('{q}')")
    con.execute("create table qmap as "+" union all ".join(maps))
    con.execute(f"""create table allrows as select f.*,q.s1 from read_parquet('{glob}',union_by_name=true)f join qmap q using(country_code,qrow)""")
    con.execute("""create table chosen as select *,null::bigint neg_rank from allrows where label=1 union all by name select * from(select *,row_number()over(partition by s1 order by historical_miner_score desc,source,apos)neg_rank from allrows where label=0)where neg_rank<=5""")
    con.execute(f"create table internal as select source1_entity_id s1 from read_parquet('{internal}')")
    con.execute("""create table weighted as with c as(select *,count(*)filter(where label=1)over(partition by s1)np,count(*)filter(where label=0)over(partition by s1)nn from chosen)select *,case when label=1 then 1.0/np else 1.0/nn end::float sample_weight,exists(select 1 from internal i where i.s1=c.s1) is_internal from c""")
    stats=con.execute("select count(*) rows,count(*)filter(where label=1)positives,count(*)filter(where label=0)negatives,count(distinct s1)s1,sum(label) reachable_positives from weighted").fetchdf().iloc[0].to_dict()
    total_true=con.execute("select sum(true_matches)from read_parquet(?)",[str(info["path"])]).fetchone()[0]
    check=con.execute("select max(abs(pm-1)),max(abs(nm-1))from(select s1,sum(sample_weight)filter(where label=1)pm,sum(sample_weight)filter(where label=0)nm from weighted group by s1)").fetchone()
    if check[0]>2e-6 or check[1]>2e-6:raise RuntimeError(f"S1 weights invalid {check}")
    train=con.execute("select "+",".join(ALL_FEATURES)+",label,sample_weight from weighted where not is_internal").fetchdf();valid=con.execute("select "+",".join(ALL_FEATURES)+",label,sample_weight from weighted where is_internal").fetchdf();con.close()
    cfg=json.loads((E001/"model/model_config.json").read_text())["config"].copy();cfg.update({"n_estimators":5000,"n_jobs":THREADS})
    model=lgb.LGBMClassifier(**cfg);fit_start=time.perf_counter();model.fit(train[ALL_FEATURES],train.label,sample_weight=train.sample_weight,eval_set=[(valid[ALL_FEATURES],valid.label)],eval_sample_weight=[valid.sample_weight],eval_metric=["binary_logloss","auc","average_precision"],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(25)]);fit_seconds=time.perf_counter()-fit_start;best=int(model.best_iteration_)
    pred=model.predict_proba(valid[ALL_FEATURES])[:,1];metrics={"binary_logloss":float(-(valid.sample_weight*(valid.label*np.log(np.clip(pred,1e-12,1))+(1-valid.label)*np.log(np.clip(1-pred,1e-12,1)))).sum()/valid.sample_weight.sum()),"auc":float(roc_auc_score(valid.label,pred,sample_weight=valid.sample_weight)),"ap":float(average_precision_score(valid.label,pred,sample_weight=valid.sample_weight))}
    projected_retrain=fit_seconds*len(pd.concat([train,valid],ignore_index=False))/len(train)*best/max(best,1)
    retrained=False
    if projected_retrain<=1800:
        allrows=pd.concat([train,valid],ignore_index=True);final=lgb.LGBMClassifier(**{**cfg,"n_estimators":best});rt=time.perf_counter();final.fit(allrows[ALL_FEATURES],allrows.label,sample_weight=allrows.sample_weight);retrain_seconds=time.perf_counter()-rt;model=final;retrained=True;del allrows
    else:retrain_seconds=0.0
    joblib.dump(model,MODEL_DIR/"model.joblib");model.booster_.save_model(str(MODEL_DIR/"model.txt"));imp=pd.DataFrame({"feature":ALL_FEATURES,"gain":model.booster_.feature_importance("gain"),"split":model.booster_.feature_importance("split")}).sort_values("gain",ascending=False);imp.to_csv(MODEL_DIR/"feature_importance.csv",index=False)
    report={"candidate_feature_key":fmanifest["content_key"],"historical_miner_model":str(E008/"model/lightgbm/model.joblib"),"total_true_positive_edges":int(total_true),"reachable_positives":int(stats["reachable_positives"]),"missed_positives":int(total_true-stats["reachable_positives"]),"positive_coverage":float(stats["reachable_positives"]/total_true),"selected_negatives":int(stats["negatives"]),"mean_selected_negatives_per_s1":float(stats["negatives"]/len(info["frame"])),"training_rows_total":int(stats["rows"]),"training_positives":int(stats["positives"]),"training_negatives":int(stats["negatives"]),"internal_s1":50000,"fit_rows":len(train),"internal_rows":len(valid),"best_iteration":best,"internal_metrics":metrics,"early_stop_seconds":fit_seconds,"projected_full_retrain_seconds":projected_retrain,"retrained_all_dev":retrained,"retrain_seconds":retrain_seconds,"peak_rss_gb":rss_gb(),"total_seconds":time.perf_counter()-t0,"model_sha256":sha256_file(MODEL_DIR/"model.txt"),"sample_weight_rule":"per S1, positives total 1.0 when present; selected negatives total 1.0 when present; singleton negatives total 1.0"}
    atomic_json(report_path,report);(MODEL_DIR/"COMPLETE").write_text(sha256_file(report_path)+"\n");return report


def score_features(population:str,fbase:Path)->Path:
    out=SCORE_DIR/population;out.mkdir(parents=True,exist_ok=True);model=joblib.load(MODEL_DIR/"model.joblib")
    for fp in sorted((fbase/"shards").glob("*.parquet")):
        op=out/fp.name;meta=op.with_suffix(".meta.json")
        if op.exists()and meta.exists():continue
        t=time.perf_counter();d=pd.read_parquet(fp);d["score"]=model.predict_proba(d[ALL_FEATURES])[:,1].astype(np.float32);keep=["country_code","qrow","source","apos","label","candidate_address_missing","score"];atomic_parquet(d[keep],op);atomic_json(meta,{"rows":len(d),"seconds":time.perf_counter()-t});log("SCORED",population=population,file=fp.name,rows=len(d));del d;gc.collect()
    return out


def oracle_score(tp,true):
    tp=np.asarray(tp,float);true=np.asarray(true,float);den=1.25*tp+.25*(true-tp);z=np.divide(1.25*tp,den,out=np.zeros_like(tp),where=den>0);z[(true==0)&(tp==0)]=1;return z


def evaluate_shortlists(info:dict,score_dir:Path)->dict:
    out=REPORT/"validation";out.mkdir(parents=True,exist_ok=True);rp=out/"results.json"
    if rp.exists()and(out/"COMPLETE").exists():return json.loads(rp.read_text())
    glob=str(score_dir/"*.parquet");con=duckdb.connect();con.execute("pragma threads=20");con.execute("pragma memory_limit='100GB'")
    maps=[]
    for country in COUNTRIES["clean_validation"]:
        q=b2.query_map("clean_validation",country,info);maps.append(f"select {COUNTRY_CODE[country]}::tinyint country_code,qrow,entity_id s1 from read_parquet('{q}')")
    con.execute("create table qmap as "+" union all ".join(maps));con.execute(f"""create table ranked as select f.*,q.s1,row_number()over(partition by q.s1 order by score desc,source,apos)rank from read_parquet('{glob}',union_by_name=true)f join qmap q using(country_code,qrow)""")
    owners=info["frame"].copy();con.register("owners0",owners);con.execute("create table owners as select * from owners0")
    dbc=duckdb.connect(str(DB),read_only=True);dbc.register("ids",owners[["source1_entity_id"]]);missing=dbc.execute("""select i.source1_entity_id,coalesce(bool_or(coalesce(a.business_address,'')=''),false)missing from ids i left join edges e on e.s1=i.source1_entity_id left join(select entity_id,business_address from train_s2 union all select entity_id,business_address from train_s3)a on a.entity_id=e.target_id group by 1""").fetchdf();dbc.close();con.register("missing0",missing);con.execute("create table missing as select * from missing0")
    ks=[6,8,10,12,16,24,32];rows=[];slice_rows=[]
    for k in ks:
        per=con.execute(f"""with z as(select s1,count(*)candidates,count(*)filter(where label=1)tp,count(*)filter(where label=1 and source=2)tp_s2,count(*)filter(where label=1 and source=3)tp_s3 from ranked where rank<={k} group by s1)select o.*,coalesce(z.candidates,0)candidates,coalesce(z.tp,0)tp,coalesce(z.tp_s2,0)tp_s2,coalesce(z.tp_s3,0)tp_s3,m.missing from owners o left join z on o.source1_entity_id=z.s1 left join missing m using(source1_entity_id)""").fetchdf();per["oracle"]=oracle_score(per.tp,per.true_matches)
        row={"k":k,"candidate_pairs":int(per.candidates.sum()),"mean_candidates":float(per.candidates.mean()),"edge_recall":float(per.tp.sum()/per.true_matches.sum()),"oracle_f05":float(per.oracle.mean())};row["loss_vs_f20_r3"]=BLOCKER_ORACLE-row["oracle_f05"];rows.append(row)
        def add(name,z,tp="tp",true="true_matches"):
            slice_rows.append({"k":k,"slice":name,"n_s1":len(z),"edge_recall":float(z[tp].sum()/z[true].sum())if z[true].sum()else 1.0,"oracle_f05":float(oracle_score(z[tp],z[true]).mean())})
        for c,z in per.groupby("country"):add(c,z)
        add("S2",per,"tp_s2","s2_matches");add("S3",per,"tp_s3","s3_matches");add("true_address_missing",per[per.missing]);add("true_address_present",per[~per.missing])
        for n in [0,1,2,3,4]:add(f"cardinality_{n}",per[per.true_matches.eq(n)])
        add("cardinality_5+",per[per.true_matches.ge(5)])
    table=pd.DataFrame(rows);table.to_csv(out/"shortlist_results.csv",index=False);pd.DataFrame(slice_rows).to_csv(out/"shortlist_slices.csv",index=False)
    pos=con.execute("select score from ranked where label=1").fetchnumpy()["score"];hard=con.execute("select score from ranked where label=0 qualify row_number()over(partition by s1 order by score desc)<=5").fetchnumpy()["score"];pair_diag={"positive_quantiles":{str(q):float(np.quantile(pos,q))for q in [.01,.1,.5,.9,.99]},"hard_false_quantiles":{str(q):float(np.quantile(hard,q))for q in [.01,.1,.5,.9,.99]},"auroc":float(roc_auc_score(np.r_[np.ones(len(pos)),np.zeros(len(hard))],np.r_[pos,hard])),"ap":float(average_precision_score(np.r_[np.ones(len(pos)),np.zeros(len(hard))],np.r_[pos,hard]))};con.close()
    qual=table[table.oracle_f05>=SHORTLIST_GATE];freeze=None if qual.empty else int(qual.iloc[0].k);result={"shortlists":rows,"freeze_k":freeze,"freeze_name":None if freeze is None else f"L{freeze}","freeze_oracle":None if freeze is None else float(qual.iloc[0].oracle_f05),"freeze_loss":None if freeze is None else float(qual.iloc[0].loss_vs_f20_r3),"pair_diagnostics":pair_diag}
    atomic_json(rp,result);(out/"COMPLETE").write_text(sha256_file(rp)+"\n");return result


def xlmr_manifest(info:dict,train_score_dir:Path)->dict:
    XLMR_DIR.mkdir(parents=True,exist_ok=True);out=XLMR_DIR/"training_pairs.parquet";rp=XLMR_DIR/"manifest_stats.json"
    if out.exists()and(XLMR_DIR/"MANIFEST_COMPLETE").exists():return json.loads(rp.read_text())
    dbp=XLMR_DIR/"manifest.duckdb";
    if dbp.exists():dbp.unlink()
    con=duckdb.connect(str(dbp));con.execute("pragma threads=20");con.execute("pragma memory_limit='100GB'");con.execute(f"attach '{DB}' as audit(read_only)")
    maps=[];amaps=[]
    for country in COUNTRIES["development_train"]:
        q=b2.query_map("development_train",country,info);maps.append(f"select {COUNTRY_CODE[country]}::tinyint country_code,qrow,entity_id s1 from read_parquet('{q}')")
        for source in [2,3]:
            a=b2.forward_index("development_train",country,source)/"aux.parquet";amaps.append(f"select {COUNTRY_CODE[country]}::tinyint country_code,{source}::tinyint source,apos,entity_id candidate_id,coalesce(business_address,'')='' candidate_address_missing from read_parquet('{a}')")
    con.execute("create table qmap as "+" union all ".join(maps));con.execute("create table amap as "+" union all ".join(amaps));glob=str(train_score_dir/"*.parquet")
    con.execute(f"""create table negatives as with z as(select q.s1,a.candidate_id,r.source,r.score,row_number()over(partition by q.s1 order by r.score desc,r.source,a.candidate_id)hard_negative_rank,q.country_code,a.candidate_address_missing from read_parquet('{glob}',union_by_name=true)r join qmap q using(country_code,qrow)join amap a using(country_code,source,apos)where r.label=0)select * from z where hard_negative_rank<=3""")
    con.execute("create table reachable as select distinct q.s1,a.candidate_id,r.source from read_parquet(?)r join qmap q using(country_code,qrow)join amap a using(country_code,source,apos)where r.label=1",[glob])
    dev=str(info["path"])
    con.execute(f"""create table positives as select e.s1,e.target_id candidate_id,e.target_source::tinyint source,null::float score,null::bigint hard_negative_rank,case q.country when 'India' then 0 else 1 end::tinyint country_code,coalesce(a.business_address,'')='' candidate_address_missing,(r.s1 is not null)blocker_reachable from audit.edges e join read_parquet('{dev}')q on q.source1_entity_id=e.s1 join(select 2 source,* from audit.train_s2 union all by name select 3 source,* from audit.train_s3)a on a.source=e.target_source and a.entity_id=e.target_id left join reachable r on r.s1=e.s1 and r.candidate_id=e.target_id and r.source=e.target_source""")
    con.execute("""create table manifest as select s1 s1_id,candidate_id,case source when 2 then 'S2'else 'S3'end source,1::tinyint label,null::smallint hard_negative_rank,blocker_reachable,coalesce(q.business_address,'')='' s1_address_missing,candidate_address_missing,case country_code when 0 then 'India'else 'US'end country from positives p join audit.train_s1 q on q.entity_id=p.s1 union all select s1,candidate_id,case source when 2 then 'S2'else 'S3'end,0::tinyint,hard_negative_rank::smallint,true,coalesce(q.business_address,'')='',candidate_address_missing,case country_code when 0 then 'India'else 'US'end from negatives n join audit.train_s1 q on q.entity_id=n.s1""")
    tmp=out.with_name(out.name+f".tmp-{os.getpid()}");con.execute(f"copy(select * from manifest order by s1_id,label desc,source,candidate_id)to '{tmp}'(format parquet,compression zstd)");tmp.replace(out)
    totals=con.execute("select count(distinct s1_id)unique_s1,count(*)filter(where label=1)positives,count(*)filter(where label=0)negatives,count(*)total_rows,count(*)filter(where label=1 and not blocker_reachable)blocker_missed_positives from manifest").fetchdf().iloc[0].to_dict();slices=con.execute("select country,source,candidate_address_missing,label,count(*)rows from manifest group by all order by all").fetchdf().to_dict("records");con.close();result={**{k:int(v)for k,v in totals.items()},"slices":slices,"sha256":sha256_file(out)};atomic_json(rp,result);(XLMR_DIR/"MANIFEST_COMPLETE").write_text(sha256_file(rp)+"\n");return result


def serialization_and_probe(xstats:dict)->tuple[dict,dict]:
    # Kept in a separate subprocess so CUDA visibility can be pinned by UUID.
    script=REPORT/"run_xlmr_probe.py"
    if not script.exists():raise RuntimeError(f"missing probe helper {script}")
    subprocess.run([VENV_PYTHON,str(script)],cwd=ROOT,check=True)
    return json.loads((XLMR_DIR/"serialization.json").read_text()),json.loads((XLMR_DIR/"throughput.json").read_text())


def write_report(pre,arts,features,train,short,xstats,ser,gpu):
    REPORT.mkdir(parents=True,exist_ok=True);rows=short["shortlists"]
    cand={k:int(v[1]["metadata"]["row_count"])for k,v in arts.items()};fm={k:v[1] for k,v in features.items()}
    lines=["# EXECUTIVE_RESULT","",f"BLOCKER = F20_R3",f"BLOCKER_ORACLE = {BLOCKER_ORACLE:.10f}","",f"DEV_TRAIN_S1 = 1775537",f"DEV_TRAIN_CANDIDATES = {cand['development_train']}","",f"TEST_S1 = 1732544",f"TEST_CANDIDATES = {cand['test']}","",f"BASE49_FEATURES = 49",f"RETRIEVAL_CONTEXT_FEATURES = {len(RC_FEATURES)}",f"TOTAL_FEATURES = {len(ALL_FEATURES)}","",f"LIGHTGBM_TRAIN_ROWS = {train['training_rows_total']}",f"LIGHTGBM_POSITIVES = {train['training_positives']}",f"LIGHTGBM_NEGATIVES = {train['training_negatives']}",f"LIGHTGBM_BEST_ITERATION = {train['best_iteration']}","",f"SHORTLIST_CANDIDATE_TO_FREEZE = {short['freeze_name']}",f"SHORTLIST_ORACLE = {short['freeze_oracle']}",f"SHORTLIST_LOSS_VS_BLOCKER = {short['freeze_loss']}","",f"XLMR_TRAIN_POSITIVES = {xstats['positives']}",f"XLMR_TRAIN_NEGATIVES = {xstats['negatives']}",f"XLMR_TOTAL_ROWS = {xstats['total_rows']}","",f"XLMR_MAX_LENGTH_RECOMMENDATION = {ser['recommendation']}","",f"XLMR_MEASURED_TRAIN_EXAMPLES_PER_SEC = {gpu['train_examples_per_second']}",f"XLMR_PROJECTED_ONE_EPOCH_TIME = {gpu['projected_one_epoch_seconds']} seconds","","# CANDIDATE_MATERIALIZATION","",f"Candidate content keys: `{ {k:v[1]['content_key'] for k,v in arts.items()} }`.","","# FEATURE_CACHE","",f"Feature content keys: `{ {k:v[1]['content_key'] for k,v in features.items()} }`."]
    lines += ["", "# RETRIEVAL_CONTEXT_FEATURES", "", ", ".join(RC_FEATURES)+". No labels or country one-hot features are present.","","# LIGHTGBM_TRAINING","",json.dumps(train,indent=2),"","# CLEAN_VALIDATION_SCORING","",json.dumps(short["pair_diagnostics"],indent=2),"","# SHORTLIST_RESULTS","","| K | Mean cand/S1 | Edge recall | Oracle F0.5 | Loss vs F20_R3 |","|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['k']} | {r['mean_candidates']:.6f} | {r['edge_recall']:.9f} | {r['oracle_f05']:.10f} | {r['loss_vs_f20_r3']:.10f} |" for r in rows]
    lines += ["","# SHORTLIST_SLICES","",f"See `{REPORT/'validation/shortlist_slices.csv'}`.","","# XLMR_TRAINING_MANIFEST","",json.dumps(xstats,indent=2),"","# SERIALIZATION","",f"Token length quantiles: `{ser['token_length_quantiles']}`.","","| Max length | Truncation rate |","|---:|---:|"]
    lines += [f"| {k} | {v:.6%} |" for k,v in ser["truncation_rates"].items()]
    lines += ["","# XLMR_THROUGHPUT","",json.dumps(gpu,indent=2),"","# DISK_AND_RUNTIME","",json.dumps(pre,indent=2),"","# CACHE_INTEGRITY","",f"All published cache manifests are content-addressed and enumerate SHA-256 checksums. Re-running completed stages validates manifests before reuse. `.git` is absent/empty, so a Git HEAD cannot be recorded.","","# NEXT_DECISION","",("Launch the final serious XLM-R development training with the frozen shortlist and manifest; this phase did not launch training." if short["freeze_name"] else "No L<=32 met the oracle gate; do not freeze or train XLM-R. Inspect source-starvation diagnostics first.")]
    tmp=REPORT/"REPORT.md.tmp";tmp.write_text("\n".join(lines)+"\n");tmp.replace(REPORT/"REPORT.md")


def main():
    ap=argparse.ArgumentParser();ap.add_argument("stage",choices=["preflight","candidates","features","train","score","xlmr","probe","all"],default="all",nargs="?");a=ap.parse_args()
    pre=preflight()
    if a.stage=="preflight":return
    pops=populations();schema=schema_manifest();arts={};features={}
    for name in ["clean_validation","development_train","test"]:
        arts[name]=candidate_artifacts(name,pops[name])
    if a.stage=="candidates":return
    # Train-critical features are intentionally completed before test features.
    for name in ["clean_validation","development_train"]:
        features[name]=feature_cache(name,pops[name],*arts[name],schema)
    if a.stage=="features":
        features["test"]=feature_cache("test",pops["test"],*arts["test"],schema);return
    train=train_lgbm(pops["development_train"],*features["development_train"]);train_scores=score_features("development_train",features["development_train"][0]);val_scores=score_features("clean_validation",features["clean_validation"][0]);short=evaluate_shortlists(pops["clean_validation"],val_scores)
    if short["freeze_name"] is None:
        raise SystemExit("STOP: no global L<=32 meets shortlist oracle gate")
    features["test"]=feature_cache("test",pops["test"],*arts["test"],schema)
    xstats=xlmr_manifest(pops["development_train"],train_scores);ser,gpu=serialization_and_probe(xstats);write_report(pre,arts,features,train,short,xstats,ser,gpu)
    (REPORT/"COMPLETE").write_text(sha256_file(REPORT/"REPORT.md")+"\n")


if __name__=="__main__":main()
