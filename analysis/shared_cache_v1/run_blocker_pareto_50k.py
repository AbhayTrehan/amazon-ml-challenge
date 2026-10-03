#!/usr/bin/env python3
"""Freeze clean validation and run the CPU-only 50k lexical blocker benchmark."""
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
HERE = ROOT / "analysis/shared_cache_v1"
MANIFESTS = HERE / "manifests"
INDEXES = HERE / "indexes"
CANDIDATES = HERE / "candidates"
REPORT = HERE / "reports/blocker_pareto_50k"
D002 = ROOT / "analysis/experiments_v1/d002_clean_public_gap"
E001 = ROOT / "analysis/experiments_v1/e001_classical_baseline"
R006 = ROOT / "analysis/experiments_v1/r006a_selective_postings"
DB = ROOT / "analysis/data_audit_v1/artifacts/audit.duckdb"
SPLIT = ROOT / "analysis/data_audit_v1/artifacts/proposed_validation_split.tsv"
OLD = E001 / "selected_ids/evaluation_s1.tsv"
ASSIGNMENT = E001 / "cache/auxiliary_fold_assignment.parquet"
D002_IDS = D002 / "clean_eval_ids.tsv"
FORWARD_INDEX = R006 / "artifacts/indexes"
PYTHON = "/home/satvik/venvs/amazonml2026/bin/python"
SALT = "amazon-ml-2026:blocker-pareto-50k:v1"
N_BENCHMARK = 50_000
COUNTRIES = ["India", "US"]
SOURCES = [2, 3]
CHANNELS = ["name", "address"]
THREADS = 20
POSTING_BUDGET = 20_000
COARSE_M = 800
FORWARD_K = 50
REVERSE_K = 3
SHARD = 50_000
EXPECTED = {
    "clean_count": 431_284,
    "clean_file_sha256": "5084362ccd992845ddf65508293552b2919b8a4f1c13190e82e7bf026ee5e74f",
    "split_sha256": "7d19600aa61d4bede7caf982e19a179e218107fd4e089d06bfa9dfcefe5cc9b9",
    "old_sha256": "6462d75744cdd6f7f91f85c3fb74d75623b539baa7549210ad3eea2f84bcc230",
}
PROC = psutil.Process()

sys.path.insert(0, str(HERE))
from cache_utils import atomic_json, complete_manifest, content_key, sha256_file, validate_complete  # noqa: E402
sys.path.insert(0, str(E001 / "scripts"))
import prepare_baseline as e001  # noqa: E402
sys.path.insert(0, str(R006))
import run_r006a  # noqa: E402


def log(message: str, **values) -> None:
    suffix = " " + json.dumps(values, sort_keys=True, default=float) if values else ""
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {message}{suffix}", flush=True)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def atomic_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    frame.to_csv(temporary, sep="\t", index=False)
    temporary.replace(path)


def code_version() -> dict:
    git_dir = ROOT / ".git"
    return {
        "git_head": None,
        "git_status": "UNAVAILABLE_EMPTY_GIT_DIRECTORY" if git_dir.is_dir() and not any(git_dir.iterdir()) else "UNAVAILABLE",
        "runner_sha256": sha256_file(Path(__file__)),
        "cache_utils_sha256": sha256_file(HERE / "cache_utils.py"),
        "r006_python_sha256": sha256_file(R006 / "run_r006a.py"),
        "r006_kernel_sha256": sha256_file(R006 / "selective_postings.cpp"),
    }


def logical_id_hash(ids: pd.Series) -> str:
    payload = "".join(f"{value}\n" for value in ids.astype(str)).encode()
    return hashlib.sha256(payload).hexdigest()


def freeze_populations() -> tuple[pd.DataFrame, dict, pd.DataFrame, dict]:
    for directory in [MANIFESTS, INDEXES, CANDIDATES, REPORT]:
        directory.mkdir(parents=True, exist_ok=True)
    clean = pd.read_csv(D002_IDS, sep="\t", dtype={"source1_entity_id": str})
    split = pd.read_csv(SPLIT, sep="\t", dtype={"source1_entity_id": str})
    old = pd.read_csv(OLD, sep="\t", dtype={"source1_entity_id": str})
    failures = []
    if sha256_file(D002_IDS) != EXPECTED["clean_file_sha256"]: failures.append("D002 clean ID artifact hash")
    if sha256_file(SPLIT) != EXPECTED["split_sha256"]: failures.append("canonical split hash")
    if sha256_file(OLD) != EXPECTED["old_sha256"]: failures.append("historical 10k hash")
    if len(clean) != EXPECTED["clean_count"] or not clean.source1_entity_id.is_unique: failures.append("clean count/uniqueness")
    split_fold = split.set_index("source1_entity_id").fold
    if not clean.source1_entity_id.isin(split_fold.index).all() or not split_fold.loc[clean.source1_entity_id].eq(0).all(): failures.append("fold0 membership")
    if set(clean.source1_entity_id) & set(old.source1_entity_id): failures.append("historical 10k exclusion")
    reconstructed = split[split.fold.eq(0) & ~split.source1_entity_id.isin(old.source1_entity_id)].sort_values("source1_entity_id", kind="mergesort")
    if reconstructed.source1_entity_id.tolist() != clean.source1_entity_id.tolist(): failures.append("D002 construction replay")
    if failures:
        raise RuntimeError("AUTHORITATIVE_POPULATION_CONTRADICTION: " + ", ".join(failures))

    clean_out = MANIFESTS / "clean_validation_s1.tsv"
    if not clean_out.exists() or sha256_file(clean_out) != EXPECTED["clean_file_sha256"]:
        temporary = clean_out.with_name(clean_out.name + f".tmp-{os.getpid()}")
        shutil.copyfile(D002_IDS, temporary)
        temporary.replace(clean_out)
    clean_identity = {
        "kind": "clean_validation_s1",
        "authoritative_artifact_sha256": EXPECTED["clean_file_sha256"],
        "upstream_split_sha256": EXPECTED["split_sha256"],
        "exclusion_sha256": EXPECTED["old_sha256"],
        "rule": "authoritative D002 clean_eval_ids.tsv; fold=0 minus exact historical E001 10k",
    }
    clean_manifest = complete_manifest(
        MANIFESTS / "clean_validation_s1.manifest.json", clean_identity, [clean_out],
        {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "count": len(clean),
         "unique_count": clean.source1_entity_id.nunique(), "logical_ids_sha256": logical_id_hash(clean.source1_entity_id),
         "code_version": code_version(), "schema": {c: str(t) for c, t in clean.dtypes.items()}},
    )

    ranked = clean.assign(_sample_hash=clean.source1_entity_id.map(
        lambda value: hashlib.sha256(f"{SALT}\0{value}".encode()).hexdigest()))
    sample = ranked.sort_values(["_sample_hash", "source1_entity_id"], kind="mergesort").head(N_BENCHMARK).drop(columns="_sample_hash")
    sample = sample.sort_values("source1_entity_id", kind="mergesort").reset_index(drop=True)
    sample_out = MANIFESTS / "blocker_benchmark_50k_s1.tsv"
    atomic_tsv(sample, sample_out)
    sample_identity = {
        "kind": "blocker_benchmark_s1", "parent_content_key": clean_manifest["content_key"],
        "algorithm": "smallest SHA256(salt + NUL + source1_entity_id), then ID sort", "salt": SALT, "count": N_BENCHMARK,
    }
    sample_manifest = complete_manifest(
        MANIFESTS / "blocker_benchmark_50k_s1.manifest.json", sample_identity, [sample_out],
        {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "count": len(sample),
         "unique_count": sample.source1_entity_id.nunique(), "logical_ids_sha256": logical_id_hash(sample.source1_entity_id),
         "schema": {c: str(t) for c, t in sample.dtypes.items()}},
    )
    if len(sample) != N_BENCHMARK or not sample.source1_entity_id.is_unique:
        raise RuntimeError("benchmark sample contract failed")
    log("POPULATIONS_FROZEN", clean=len(clean), sample=len(sample), sample_key=sample_manifest["content_key"])
    return clean, clean_manifest, sample, sample_manifest


def query_frame(ids: pd.DataFrame, country: str) -> pd.DataFrame:
    take = ids.loc[ids.country.eq(country), ["source1_entity_id"]]
    con = duckdb.connect(str(DB), read_only=True); con.register("ids", take)
    frame = con.execute("""select s.entity_id,s.business_name,coalesce(s.business_address,'') business_address,
      s.country from train_s1 s join ids i on i.source1_entity_id=s.entity_id order by s.entity_id""").fetchdf()
    con.close()
    if frame.entity_id.tolist() != sorted(take.source1_entity_id.tolist()):
        raise RuntimeError(f"query recovery mismatch: {country}")
    return frame


def qmatrix(frame: pd.DataFrame, channel: str, idf: np.ndarray):
    column = "business_name" if channel == "name" else "business_address"
    transform = TfidfTransformer(); transform.idf_ = idf
    return transform.transform(e001.vec_for(channel).transform(frame[column].fillna(""))).astype(np.float32).tocsr()


def register_forward_indexes() -> tuple[dict, float]:
    started = time.perf_counter(); keys = {}
    for country in COUNTRIES:
        for source in SOURCES:
            base = FORWARD_INDEX / country / f"s{source}"
            dependencies = {
                "kind": "forward_index_reference", "candidate_corpus": "train auxiliary fold0",
                "candidate_assignment_sha256": sha256_file(ASSIGNMENT), "source": source, "country": country,
                "normalization_view": "E001 HashingVectorizer char_wb configs",
                "vectorizer": repr(e001.vec_for("name")) + "|" + repr(e001.vec_for("address")),
                "files": {p.name: sha256_file(p) for p in sorted(base.iterdir()) if p.is_file()},
            }
            manifest_path = INDEXES / "forward" / country / f"s{source}.manifest.json"
            artifacts = [base / "aux.parquet", base / "name.npz", base / "name_idf.npy", base / "address.npz", base / "address_idf.npy"]
            manifest = complete_manifest(manifest_path, dependencies, artifacts,
                {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "storage": "in_place_reference",
                 "row_count": len(pd.read_parquet(base / "aux.parquet", columns=["entity_id"])),
                 "schema": {"aux": ["entity_id", "business_name", "business_address"], "matrix": "CSR float32"}})
            keys[f"{country}_s{source}"] = manifest["content_key"]
    return keys, time.perf_counter() - started


def build_reverse_indexes(sample: pd.DataFrame, sample_manifest: dict) -> tuple[dict, float, int]:
    started = time.perf_counter(); keys = {}; hits = 0
    for country in COUNTRIES:
        query = query_frame(sample, country)
        for channel in CHANNELS:
            identity = {"kind": "reverse_s1_index", "benchmark_population_key": sample_manifest["content_key"],
                        "country": country, "channel": channel, "normalization_view": "E001 HashingVectorizer char_wb configs",
                        "vectorizer": repr(e001.vec_for(channel))}
            key = content_key(identity); base = INDEXES / key; manifest_path = base / "manifest.json"
            manifest = validate_complete(manifest_path)
            if manifest:
                hits += 1; keys[f"{country}_{channel}"] = key; log("REVERSE_INDEX_CACHE_HIT", country=country, channel=channel); continue
            base.mkdir(parents=True, exist_ok=True)
            column = "business_name" if channel == "name" else "business_address"
            counts = e001.vec_for(channel).transform(query[column].fillna(""))
            transform = TfidfTransformer(); matrix = transform.fit_transform(counts).astype(np.float32).tocsr()
            corpus = base / "corpus.parquet"; matrix_path = base / "matrix.npz"; idf_path = base / "idf.npy"
            atomic_parquet(query[["entity_id", "country"]], corpus)
            sparse.save_npz(matrix_path.with_name(matrix_path.name + f".tmp-{os.getpid()}"), matrix, compressed=False)
            temporary_matrix = matrix_path.with_name(matrix_path.name + f".tmp-{os.getpid()}.npz")
            generated_matrix = matrix_path.with_name(matrix_path.name + f".tmp-{os.getpid()}") + Path(".npz") if False else None
            # scipy appends .npz when needed; normalize its temporary filename.
            possible = Path(str(matrix_path.with_name(matrix_path.name + f".tmp-{os.getpid()}")) + ".npz")
            if possible.exists(): possible.replace(matrix_path)
            temporary_idf = idf_path.with_name(idf_path.name + f".tmp-{os.getpid()}")
            with temporary_idf.open("wb") as handle: np.save(handle, transform.idf_.astype(np.float32))
            temporary_idf.replace(idf_path)
            complete_manifest(manifest_path, identity, [corpus, matrix_path, idf_path],
                {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "row_count": len(query),
                 "schema": {"corpus": ["entity_id", "country"], "matrix": "CSR float32"}, "measured_build_seconds": time.perf_counter()-started})
            keys[f"{country}_{channel}"] = key; log("REVERSE_INDEX_READY", country=country, channel=channel, rows=len(query))
            del counts, matrix; gc.collect()
    return keys, time.perf_counter() - started, hits


def load_topk_kernel():
    library = ctypes.CDLL(str(run_r006a.compile_kernel()))
    function = library.selective_postings_topk
    ip = ctypes.POINTER(ctypes.c_int32); fp = ctypes.POINTER(ctypes.c_float)
    i64p = ctypes.POINTER(ctypes.c_int64); dp = ctypes.POINTER(ctypes.c_double)
    function.argtypes = [ip, ip, fp, ctypes.c_int, ip, ip, fp, ip, ip, fp, ctypes.c_int,
                         ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ip, fp, i64p, i64p, ip, ip, dp]
    function.restype = ctypes.c_int
    return function


def selective_topk(function, xq, xa, k: int):
    query = xq.tocsr(); rows = xa.tocsr(); columns = xa.tocsc()
    arrays = [run_r006a.i32(query.indptr), run_r006a.i32(query.indices), run_r006a.f32(query.data),
              run_r006a.i32(columns.indptr), run_r006a.i32(columns.indices), run_r006a.f32(columns.data),
              run_r006a.i32(rows.indptr), run_r006a.i32(rows.indices), run_r006a.f32(rows.data)]
    n = query.shape[0]; indices = np.full((n, k), -1, np.int32); scores = np.zeros((n, k), np.float32)
    visits = np.zeros(n, np.int64); full = np.zeros(n, np.int64); counts = np.zeros(n, np.int32)
    fallback = np.zeros(n, np.int32); timing = np.zeros(5, np.float64)
    ip = ctypes.POINTER(ctypes.c_int32); fp = ctypes.POINTER(ctypes.c_float); i64p = ctypes.POINTER(ctypes.c_int64)
    started = time.perf_counter()
    result = function(arrays[0].ctypes.data_as(ip), arrays[1].ctypes.data_as(ip), arrays[2].ctypes.data_as(fp), n,
        arrays[3].ctypes.data_as(ip), arrays[4].ctypes.data_as(ip), arrays[5].ctypes.data_as(fp),
        arrays[6].ctypes.data_as(ip), arrays[7].ctypes.data_as(ip), arrays[8].ctypes.data_as(fp), rows.shape[0],
        POSTING_BUDGET, COARSE_M, THREADS, k, indices.ctypes.data_as(ip), scores.ctypes.data_as(fp),
        visits.ctypes.data_as(i64p), full.ctypes.data_as(i64p), counts.ctypes.data_as(ip), fallback.ctypes.data_as(ip),
        timing.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    if result: raise RuntimeError(f"selective top-k kernel failed: {result}")
    return indices, scores, time.perf_counter() - started


def forward_pool(sample: pd.DataFrame, sample_manifest: dict, index_keys: dict) -> tuple[Path, dict, bool]:
    identity = {"kind": "forward_retrieval_pool", "query_population_key": sample_manifest["content_key"],
                "index_keys": index_keys, "blocker": {"posting_budget": POSTING_BUDGET, "coarse_m": COARSE_M,
                "threads": THREADS, "top_k": FORWARD_K, "exact_rerank": True},
                "retrieval_code_sha256": sha256_file(R006 / "selective_postings.cpp")}
    key = content_key(identity); base = CANDIDATES / key; manifest_path = base / "manifest.json"
    cached = validate_complete(manifest_path)
    if cached:
        log("FORWARD_POOL_CACHE_HIT", key=key); return base / "pool.parquet", cached, True
    base.mkdir(parents=True, exist_ok=True); function = load_topk_kernel(); pieces = []; runtimes = []; started_all = time.perf_counter()
    for country in COUNTRIES:
        query = query_frame(sample, country); query_ids = query.entity_id.to_numpy()
        for source in SOURCES:
            index_base = FORWARD_INDEX / country / f"s{source}"
            aux = pd.read_parquet(index_base / "aux.parquet", columns=["entity_id"]); aux_ids = aux.entity_id.to_numpy()
            for channel in CHANNELS:
                matrix = sparse.load_npz(index_base / f"{channel}.npz").tocsr(); idf = np.load(index_base / f"{channel}_idf.npy")
                transform_start = time.perf_counter(); queries = qmatrix(query, channel, idf); transform_seconds = time.perf_counter()-transform_start
                indices, scores, kernel_seconds = selective_topk(function, queries, matrix, FORWARD_K)
                rr, kk = np.where(indices >= 0)
                frame = pd.DataFrame({"s1": query_ids[rr], "target_id": aux_ids[indices[rr, kk]], "source": np.int8(source),
                    "country": country, "channel": channel, "rank": (kk+1).astype(np.int16), "score": scores[rr, kk].astype(np.float32)})
                pieces.append(frame); runtimes.append({"country": country, "source": source, "channel": channel,
                    "queries": len(query), "rows": len(frame), "transform_seconds": transform_seconds, "retrieval_seconds": kernel_seconds})
                log("FORWARD_STREAM_READY", country=country, source=source, channel=channel, queries=len(query), rows=len(frame), seconds=kernel_seconds)
                del matrix, queries, indices, scores, frame; gc.collect()
    pool = pd.concat(pieces, ignore_index=True); pool_path = base / "pool.parquet"; atomic_parquet(pool, pool_path)
    runtime_path = base / "runtime.json"; atomic_json(runtime_path, {"streams": runtimes, "wall_seconds": time.perf_counter()-started_all})
    manifest = complete_manifest(manifest_path, identity, [pool_path, runtime_path],
        {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "row_count": len(pool),
         "query_count": len(sample), "schema": {c: str(t) for c,t in pool.dtypes.items()}, "measured_runtime_seconds": time.perf_counter()-started_all})
    return pool_path, manifest, False


def reverse_pool(sample: pd.DataFrame, sample_manifest: dict, reverse_keys: dict) -> tuple[Path, dict, bool]:
    identity = {"kind": "reverse_retrieval_pool", "query_auxiliary": "train auxiliary fold0 only",
                "owner_population_key": sample_manifest["content_key"], "index_keys": reverse_keys,
                "retrieval": {"method": "exact sparse_dot_topn", "top_k": REVERSE_K, "threshold": 1e-8, "threads": THREADS},
                "retrieval_code_version": "sparse_dot_topn_exact_v1"}
    key = content_key(identity); base = CANDIDATES / key; manifest_path = base / "manifest.json"
    cached = validate_complete(manifest_path)
    if cached:
        log("REVERSE_POOL_CACHE_HIT", key=key); return base / "pool.parquet", cached, True
    base.mkdir(parents=True, exist_ok=True); pieces = []; runtimes = []; started_all = time.perf_counter()
    for country in COUNTRIES:
        for channel in CHANNELS:
            index_base = INDEXES / reverse_keys[f"{country}_{channel}"]
            corpus = pd.read_parquet(index_base / "corpus.parquet"); corpus_ids = corpus.entity_id.to_numpy()
            matrix = sparse.load_npz(index_base / "matrix.npz").tocsr(); idf = np.load(index_base / "idf.npy")
            for source in SOURCES:
                aux_path = FORWARD_INDEX / country / f"s{source}" / "aux.parquet"
                aux = pd.read_parquet(aux_path); aux_ids = aux.entity_id.to_numpy()
                for part, start in enumerate(range(0, len(aux), SHARD)):
                    shard_path = base / "parts" / country / f"s{source}_{channel}_{part:04d}.parquet"
                    if shard_path.exists():
                        piece = pd.read_parquet(shard_path); pieces.append(piece); log("REVERSE_SHARD_CACHE_HIT", path=str(shard_path)); continue
                    block = aux.iloc[start:start+SHARD]; transform_start = time.perf_counter(); queries = qmatrix(block, channel, idf); transform_seconds = time.perf_counter()-transform_start
                    retrieve_start = time.perf_counter(); top = sp_matmul_topn(queries, matrix.T, top_n=REVERSE_K, threshold=1e-8, sort=True, n_threads=THREADS).tocsr(); retrieval_seconds = time.perf_counter()-retrieve_start
                    qpos = np.repeat(np.arange(top.shape[0]), np.diff(top.indptr)); ranks = np.concatenate([np.arange(1, top.indptr[i+1]-top.indptr[i]+1) for i in range(top.shape[0])]).astype(np.int16)
                    frame = pd.DataFrame({"s1": corpus_ids[top.indices], "target_id": aux_ids[start:start+len(block)][qpos],
                        "source": np.int8(source), "country": country, "channel": channel, "rank": ranks, "score": top.data.astype(np.float32)})
                    atomic_parquet(frame, shard_path); pieces.append(frame); runtimes.append({"country": country, "source": source, "channel": channel,
                        "part": part, "queries": len(block), "rows": len(frame), "transform_seconds": transform_seconds, "retrieval_seconds": retrieval_seconds})
                    log("REVERSE_SHARD_READY", country=country, source=source, channel=channel, part=part, queries=len(block), seconds=retrieval_seconds)
                    del block, queries, top, qpos, ranks, frame; gc.collect()
                del aux
            del matrix; gc.collect()
    pool = pd.concat(pieces, ignore_index=True); pool_path = base / "pool.parquet"; atomic_parquet(pool, pool_path)
    runtime_path = base / "runtime.json"; atomic_json(runtime_path, {"shards": runtimes, "wall_seconds": time.perf_counter()-started_all})
    manifest = complete_manifest(manifest_path, identity, [pool_path, runtime_path],
        {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "row_count": len(pool),
         "query_count": int(sum(len(pd.read_parquet(FORWARD_INDEX/c/f"s{s}"/"aux.parquet",columns=["entity_id"])) for c in COUNTRIES for s in SOURCES)),
         "schema": {c: str(t) for c,t in pool.dtypes.items()}, "measured_runtime_seconds": time.perf_counter()-started_all})
    return pool_path, manifest, False


def f05(tp, truth):
    tp=np.asarray(tp,dtype=float); truth=np.asarray(truth,dtype=float); den=1.25*tp+.25*(truth-tp)
    out=np.divide(1.25*tp,den,out=np.zeros_like(tp),where=den>0); out[(truth==0)&(tp==0)]=1.; return out


def evaluate(sample: pd.DataFrame, clean: pd.DataFrame, sample_manifest: dict, forward_path: Path, forward_manifest: dict,
             reverse_path: Path, reverse_manifest: dict, index_seconds: float, cache_hits: dict) -> dict:
    started = time.perf_counter(); database = REPORT / "benchmark.duckdb"
    if database.exists(): database.unlink()
    con = duckdb.connect(str(database)); con.execute("pragma threads=20"); con.execute("pragma memory_limit='100GB'")
    con.execute(f"attach '{DB}' as audit (read_only)")
    con.execute("create table ids as select * from read_csv(?, delim='\\t', header=true)", [str(MANIFESTS/"blocker_benchmark_50k_s1.tsv")])
    con.execute("create table forward as select * from read_parquet(?)", [str(forward_path)])
    con.execute("create table reverse as select * from read_parquet(?)", [str(reverse_path)])
    con.execute("create table truth as select e.s1,e.target_id,e.target_source::tinyint as \"source\" from read_parquet(?) e join ids i on e.s1=i.source1_entity_id", [str(ROOT/"analysis/data_audit_v1/artifacts/positive_edges.parquet")])
    if con.execute("select count(*) from truth").fetchone()[0] != int(sample.true_matches.sum()): raise RuntimeError("truth count mismatch")
    # Deterministic rank-based ambiguity selection; invalid margins sort first.
    con.execute("""create table margins as with z as(select s1,source,channel,
      max(case when rank=1 then score end)-max(case when rank=2 then score end) margin,
      count_if(rank<=2) score_count from forward group by 1,2,3)
      select *,case when score_count<2 or not isfinite(margin) then -1e300 else margin end effective_margin from z""")
    stream_total = con.execute("select count(*) from margins").fetchone()[0]
    expected_streams = len(sample)*4
    if stream_total != expected_streams: raise RuntimeError(f"margin stream count {stream_total} != {expected_streams}")
    for pct in [10,20,30]:
        con.execute(f"""create table escalate_{pct} as select s1,source,channel from(
          select *,row_number()over(partition by source,channel order by effective_margin,s1) rn,
          count(*)over(partition by source,channel)stream_n from margins)where rn<=ceil(stream_n*{pct}/100.0)""")
    definitions = {
        "B0": "select distinct s1,target_id,source from forward where rank<=10",
        "B1": "select distinct s1,target_id,source from forward where rank<=20",
        "B2": "select distinct s1,target_id,source from (select s1,target_id,source from forward where rank<=20 union all select s1,target_id,source from reverse where rank<=1)",
        "B3": "select distinct s1,target_id,source from (select s1,target_id,source from forward where rank<=20 union all select s1,target_id,source from reverse where rank<=3)",
    }
    for pct in [10,20,30]:
        definitions[f"A{pct}"] = f"""select distinct s1,target_id,source from(
          select s1,target_id,source from forward where rank<=20
          union all select s1,target_id,source from reverse where rank<=1
          union all select f.s1,f.target_id,f.source from forward f join escalate_{pct} e using(s1,source,channel) where f.rank<=50)"""
    derivation = {}
    for config, query in definitions.items():
        t=time.perf_counter(); con.execute(f"create table c_{config} as {query}"); derivation[config]=time.perf_counter()-t
    metrics=[]; slices=[]
    for config, query in definitions.items():
        invalid = con.execute(f"select count(*) from c_{config} c left join ids i on c.s1=i.source1_entity_id where i.source1_entity_id is null").fetchone()[0]
        duplicates = con.execute(f"select count(*)-count(distinct (s1,target_id,source)) from c_{config}").fetchone()[0]
        wrong_source = con.execute(f"select count(*) from c_{config} where source not in (2,3)").fetchone()[0]
        country_cross = con.execute(f"""select count(*) from c_{config} c join ids i on c.s1=i.source1_entity_id join(
          select entity_id,country from audit.train_s2 union all select entity_id,country from audit.train_s3)a on a.entity_id=c.target_id where i.country<>a.country""").fetchone()[0]
        if any([invalid,duplicates,wrong_source,country_cross]): raise RuntimeError(f"candidate invariant failed {config}: {invalid,duplicates,wrong_source,country_cross}")
        per=con.execute(f"""with ca as(select s1,count(*) candidates from c_{config} group by 1), hit as(
          select c.s1,count(*)tp from c_{config} c join truth t using(s1,target_id,source)group by 1)
          select i.*,coalesce(ca.candidates,0)candidates,coalesce(hit.tp,0)tp from ids i left join ca on i.source1_entity_id=ca.s1 left join hit on i.source1_entity_id=hit.s1 order by 1""").fetchdf()
        scores=f05(per.tp,per.true_matches); total=int(per.candidates.sum()); tp=int(per.tp.sum())
        base_tp = 0 if config=="B1" else con.execute(f"select count(*) from ((select * from c_{config} intersect select * from truth) except (select * from c_B1 intersect select * from truth))").fetchone()[0]
        row={"config":config,"mean_candidates_per_s1":total/len(sample),"median_candidates":float(per.candidates.median()),
             "p95_candidates":float(per.candidates.quantile(.95)),"p99_candidates":float(per.candidates.quantile(.99)),"max_candidates":int(per.candidates.max()),
             "total_candidate_pairs":total,"true_edges_retained":tp,"true_edge_recall":tp/int(sample.true_matches.sum()),
             "oracle_macro_f05":float(scores.mean()),"unique_tp_vs_B1":int(base_tp),"derivation_seconds":derivation[config]}
        metrics.append(row)
        slice_masks={"overall":np.ones(len(per),bool),"US":per.country.eq("US").to_numpy(),"India":per.country.eq("India").to_numpy()}
        for card in [0,1,2,3,4]: slice_masks[f"cardinality_{card}"]=per.true_matches.eq(card).to_numpy()
        slice_masks["cardinality_5+"]=per.true_matches.ge(5).to_numpy()
        if config=="B0":
            meta_con=duckdb.connect(str(DB),read_only=True); meta_con.register("sid",sample[["source1_entity_id"]])
            missing=meta_con.execute("""select i.source1_entity_id,coalesce(bool_or(coalesce(a.business_address,'')=''),false) missing from sid i left join edges e on e.s1=i.source1_entity_id left join(select entity_id,business_address from train_s2 union all select entity_id,business_address from train_s3)a on a.entity_id=e.target_id group by 1""").fetchdf(); meta_con.close()
            missing_map=missing.set_index("source1_entity_id").missing; sample_missing=per.source1_entity_id.map(missing_map).fillna(False).to_numpy()
        slice_masks["true_candidate_address_missing"]=sample_missing
        slice_masks["true_candidate_address_present"]=~sample_missing
        for name,mask in slice_masks.items():
            sub=per.loc[mask]; subscores=scores[mask]; slices.append({"config":config,"slice":name,"n_s1":len(sub),
                "true_edges":int(sub.true_matches.sum()),"true_edges_retained":int(sub.tp.sum()),
                "edge_recall":float(sub.tp.sum()/sub.true_matches.sum()) if sub.true_matches.sum() else 1.,
                "oracle_macro_f05":float(subscores.mean()),"mean_candidates_per_s1":float(sub.candidates.mean())})
        for source,col in [(2,"s2_matches"),(3,"s3_matches")]:
            src=con.execute(f"""with h as(select c.s1,count(*)tp from c_{config} c join truth t using(s1,target_id,source)where c.source={source} group by 1)
                select i.source1_entity_id,i.{col} truth,coalesce(h.tp,0)tp from ids i left join h on i.source1_entity_id=h.s1 order by 1""").fetchdf()
            slices.append({"config":config,"slice":f"S{source}","n_s1":len(src),"true_edges":int(src.truth.sum()),
                "true_edges_retained":int(src.tp.sum()),"edge_recall":float(src.tp.sum()/src.truth.sum()),
                "oracle_macro_f05":float(f05(src.tp,src.truth).mean()),"mean_candidates_per_s1":float("nan")})
        log("CONFIG_SCORED", config=config, mean=row["mean_candidates_per_s1"], oracle=row["oracle_macro_f05"], recall=row["true_edge_recall"])
    metric_frame=pd.DataFrame(metrics); baseline=metric_frame.set_index("config").loc["B1"]
    metric_frame["delta_oracle_vs_B1"]=metric_frame.oracle_macro_f05-baseline.oracle_macro_f05
    metric_frame["delta_candidates_per_s1_vs_B1"]=metric_frame.mean_candidates_per_s1-baseline.mean_candidates_per_s1
    # Exact parity with the already-realized D002 production blocker on this sample.
    d002_glob=str(D002/"cache/lightgbm/*/s*/*.parquet")
    parity=con.execute(f"""with old as(select distinct s1,target_id,source from read_parquet('{d002_glob}',union_by_name=true)join ids on s1=source1_entity_id),
      missing as(select * from old except select * from c_B0),extra as(select * from c_B0 except select * from old)
      select (select count(*)from old)old_pairs,(select count(*)from c_B0)new_pairs,(select count(*)from missing)missing,(select count(*)from extra)extra""").fetchdf().iloc[0].to_dict()
    if parity["missing"] or parity["extra"]: raise RuntimeError(f"B0 exact parity failed: {parity}")
    reverse_rows=[]
    for source in SOURCES:
        for channel in CHANNELS:
            for depth in [1,3]:
                vals=con.execute(f"""with r as(select distinct s1,target_id,source from reverse where source={source}and channel='{channel}'and rank<={depth}),
                 novel as(select * from r except select * from c_B1), true_novel as(select * from novel intersect select * from truth)
                 select (select count(*)from novel),(select count(*)from true_novel)""").fetchone()
                mutual=con.execute(f"""select count(*) from(select distinct r.s1,r.target_id,r.source from reverse r join forward f using(s1,target_id,source,channel)
                  where r.source={source}and r.channel='{channel}'and r.rank=1 and f.rank=1)""").fetchone()[0]
                reverse_rows.append({"source":f"S{source}","channel":channel,"reverse_depth":depth,
                    "unique_pairs_over_B1":vals[0],"unique_true_edges_over_B1":vals[1],"mutual_rank1_pairs":mutual if depth==1 else np.nan})
    reverse_frame=pd.DataFrame(reverse_rows)
    adaptive=[]
    for pct in [10,20,30]:
        row=metric_frame.set_index("config").loc[f"A{pct}"]
        escalated=con.execute(f"select count(*) from escalate_{pct}").fetchone()[0]
        adaptive.append({"config":f"A{pct}","escalated_stream_queries":escalated,"escalated_fraction":escalated/expected_streams,
            "candidate_increase_vs_B2":row.total_candidate_pairs-metric_frame.set_index("config").loc["B2"].total_candidate_pairs,
            "unique_tp_vs_B1":row.unique_tp_vs_B1,"oracle_gain_vs_B1":row.delta_oracle_vs_B1})
    adaptive_frame=pd.DataFrame(adaptive)
    # Decision gate and Pareto frontier.
    eligible=metric_frame[metric_frame.oracle_macro_f05.ge(.9930)].sort_values(["mean_candidates_per_s1","oracle_macro_f05"],ascending=[True,False])
    freeze=None if eligible.empty else eligible.iloc[0]["config"]
    frontier=[]
    for _,row in metric_frame.sort_values("mean_candidates_per_s1").iterrows():
        if not frontier or row.oracle_macro_f05>max(x["oracle_macro_f05"] for x in frontier): frontier.append(row.to_dict())
    best=max(frontier,key=lambda x:x["oracle_macro_f05"])
    if freeze is None:
        miss=con.execute(f"""select t.*,i.country,i.true_matches from truth t join ids i on t.s1=i.source1_entity_id left join c_{best['config']} c using(s1,target_id,source)where c.s1 is null order by t.s1,t.source,t.target_id""").fetchdf()
        atomic_parquet(miss,REPORT/"blocker_misses_best_pareto.parquet")
    metric_frame.to_csv(REPORT/"config_metrics.csv",index=False); pd.DataFrame(slices).to_csv(REPORT/"slice_metrics.csv",index=False)
    reverse_frame.to_csv(REPORT/"reverse_channel_contributions.csv",index=False); adaptive_frame.to_csv(REPORT/"adaptive_contributions.csv",index=False)
    metric_frame.to_parquet(REPORT/"config_metrics.parquet",index=False); pd.DataFrame(slices).to_parquet(REPORT/"slice_metrics.parquet",index=False)
    # Runtime and scale projections from measured query rates.
    fr=json.loads((forward_path.parent/"runtime.json").read_text()); rr=json.loads((reverse_path.parent/"runtime.json").read_text())
    f_queries=sum(x["queries"] for x in fr["streams"]); f_seconds=sum(x["transform_seconds"]+x["retrieval_seconds"] for x in fr["streams"])
    r_queries=sum(x["queries"] for x in rr["shards"]); r_seconds=sum(x["transform_seconds"]+x["retrieval_seconds"] for x in rr["shards"])
    split=pd.read_csv(SPLIT,sep="\t"); train_n=int(split.fold.ne(0).sum()); val_n=len(clean); test_n=1_732_544
    dbc=duckdb.connect(str(DB),read_only=True)
    train_aux=int(dbc.execute(f"select count(*) from read_parquet('{ASSIGNMENT}')where fold<>0").fetchone()[0]); val_aux=int(dbc.execute(f"select count(*) from read_parquet('{ASSIGNMENT}')where fold=0").fetchone()[0]); test_aux=sum(dbc.execute(f"select count(*) from test_s{s}").fetchone()[0] for s in SOURCES); dbc.close()
    projections=[]
    for population,n_s1,n_aux in [("development_train",train_n,train_aux),("clean_validation",val_n,val_aux),("test",test_n,test_aux)]:
        projected_seconds=f_seconds/f_queries*(4*n_s1)+r_seconds/r_queries*(2*n_aux)
        for _,row in metric_frame.iterrows(): projections.append({"population":population,"config":row.config,"s1_count":n_s1,
            "projected_candidate_pairs":int(round(row.mean_candidates_per_s1*n_s1)),"projected_retrieval_seconds":projected_seconds,
            "basis":"measured forward stream-query and reverse auxiliary-query rates; excludes future index build"})
    projection_frame=pd.DataFrame(projections); projection_frame.to_csv(REPORT/"scale_projection.csv",index=False)
    elapsed=time.perf_counter()-started; peak=max(PROC.memory_info().rss,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)/2**30
    summary={"state":"COMPLETE","population_key":sample_manifest["content_key"],"forward_pool_key":forward_manifest["content_key"],
      "reverse_pool_key":reverse_manifest["content_key"],"b0_parity":parity,"decision":{"threshold":.993,"blocker_candidate_to_freeze":freeze,
      "best_pareto_config":best["config"],"pareto_frontier":[x["config"] for x in frontier]},"timing":{"index_load_build_seconds":index_seconds,
      "forward_wall_seconds":fr["wall_seconds"],"reverse_wall_seconds":rr["wall_seconds"],"config_derivation_scoring_seconds":elapsed,
      "derivation_by_config_seconds":derivation},"cache_hits":cache_hits,"peak_rss_gb":peak,"disk_bytes":sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()),
      "oracle_tests":{"perfect_coverage":float(f05([3],[3])[0]),"empty_truth":float(f05([0],[0])[0]),"one_of_two":float(f05([1],[2])[0]),"oracle_contains_false_candidates":False},
      "reverse_no_label_leakage":"retrieval ranks use text TF-IDF only; labels are joined after all candidate sets are materialized"}
    atomic_json(REPORT/"benchmark_manifest.json",summary)
    write_report(clean, sample, metric_frame, pd.DataFrame(slices), reverse_frame, adaptive_frame, projection_frame, summary, clean_manifest=None)
    artifacts=[REPORT/"REPORT.md",REPORT/"config_metrics.csv",REPORT/"config_metrics.parquet",REPORT/"slice_metrics.csv",REPORT/"slice_metrics.parquet",
      REPORT/"reverse_channel_contributions.csv",REPORT/"adaptive_contributions.csv",REPORT/"scale_projection.csv",REPORT/"benchmark_manifest.json"]
    if freeze is None: artifacts.append(REPORT/"blocker_misses_best_pareto.parquet")
    identity={"kind":"blocker_pareto_50k_report","population_key":sample_manifest["content_key"],"forward_pool_key":forward_manifest["content_key"],
              "reverse_pool_key":reverse_manifest["content_key"],"configurations":list(definitions),"decision_threshold":.993}
    identity["evaluation_code_sha256"] = sha256_file(Path(__file__))
    complete_manifest(REPORT/"COMPLETE.manifest.json",identity,artifacts,{"created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"metrics_rows":len(metric_frame)})
    con.close(); return summary


def write_report(clean,sample,metrics,slices,reverse,adaptive,projection,summary,clean_manifest=None):
    def table(frame, columns):
        head="| "+" | ".join(columns)+" |\n|"+"|".join(["---"]*len(columns))+"|\n"; rows=[]
        for _,r in frame.iterrows(): rows.append("| "+" | ".join(str(r[c]) for c in columns)+" |")
        return head+"\n".join(rows)
    display=metrics.copy(); display["Mean cand/S1"]=display.mean_candidates_per_s1.map(lambda x:f"{x:.3f}"); display["Edge recall"]=display.true_edge_recall.map(lambda x:f"{x:.6f}")
    display["Oracle F0.5"]=display.oracle_macro_f05.map(lambda x:f"{x:.6f}"); display["Δoracle vs K20"]=display.delta_oracle_vs_B1.map(lambda x:f"{x:+.6f}")
    display["Unique TP vs K20"]=display.unique_tp_vs_B1.astype(int); display["Runtime"]=display.derivation_seconds.map(lambda x:f"{x:.2f}s derive")
    display.rename(columns={"config":"Config"},inplace=True)
    gate=(f"BLOCKER_CANDIDATE_TO_FREEZE = {summary['decision']['blocker_candidate_to_freeze']}" if summary['decision']['blocker_candidate_to_freeze'] else
          f"NO_CONFIG_REACHED_0.993\n\nPareto-best config: {summary['decision']['best_pareto_config']}; see `blocker_misses_best_pareto.parquet`.")
    important=slices[slices.slice.isin(["US","India","S2","S3","true_candidate_address_missing","true_candidate_address_present","cardinality_0","cardinality_1","cardinality_2","cardinality_3","cardinality_4","cardinality_5+"])].copy()
    important["oracle_macro_f05"]=important.oracle_macro_f05.map(lambda x:f"{x:.6f}"); important["edge_recall"]=important.edge_recall.map(lambda x:f"{x:.6f}")
    report=f"""# EXECUTIVE_RESULT

{gate}

{table(display,["Config","Mean cand/S1","Edge recall","Oracle F0.5","Δoracle vs K20","Unique TP vs K20","Runtime"])}

# VALIDATION_FREEZE

- Clean validation: {len(clean):,} S1; file SHA-256 `{sha256_file(MANIFESTS/'clean_validation_s1.tsv')}`; logical-ID SHA-256 `{logical_id_hash(clean.source1_entity_id)}`.
- 50k benchmark: {len(sample):,} S1; file SHA-256 `{sha256_file(MANIFESTS/'blocker_benchmark_50k_s1.tsv')}`; logical-ID SHA-256 `{logical_id_hash(sample.source1_entity_id)}`; salt `{SALT}`.
- Composition: {sample.country.value_counts().to_dict()}; truth cardinality {sample.true_matches.map(lambda x:str(x) if x<5 else '5+').value_counts().sort_index().to_dict()}; singleton share {(sample.true_matches==0).mean():.6f}.
- Git provenance: no HEAD/status is available because the repository `.git` directory is empty.

# CACHE_IMPLEMENTATION

- Added canonical-JSON content keys, separate incidental metadata, checksummed manifests, and atomic COMPLETE publication under `manifests/`, `indexes/`, `candidates/`, and `reports/`.
- Reused the 1.6 GB R006B fold-0 forward indexes in place through checksummed reference manifests; built only compact benchmark-S1 reverse indexes.
- Cache hits during this run: `{summary['cache_hits']}`. Re-running this command validates the COMPLETE manifest and all hashes before returning a cache hit.

# BLOCKER_RESULTS

Overall results are in `config_metrics.csv/parquet`. Important slices:

{table(important,["config","slice","n_s1","edge_recall","oracle_macro_f05"])}

Reverse channel contributions are in `reverse_channel_contributions.csv`; adaptive contributions are in `adaptive_contributions.csv`. B0 exactly equals D002/R006B on this population: `{summary['b0_parity']}`.

# SCALE_PROJECTION

- Measured: index load/build {summary['timing']['index_load_build_seconds']:.2f}s; forward top50 {summary['timing']['forward_wall_seconds']:.2f}s; reverse top3 {summary['timing']['reverse_wall_seconds']:.2f}s; derivation/scoring {summary['timing']['config_derivation_scoring_seconds']:.2f}s.
- Peak process RSS: {summary['peak_rss_gb']:.2f} GiB. Shared-cache disk after completion: {summary['disk_bytes']/2**30:.2f} GiB.
- Per-population/config candidate and runtime projections are in `scale_projection.csv`; they scale measured forward stream-query and reverse auxiliary-query rates, and are projections rather than measured full runs.

# DECISION_GATE

{gate}

# REPOSITORY_CHANGES

- Added `analysis/shared_cache_v1/cache_utils.py`, `run_blocker_pareto_50k.py`, focused tests, manifests, indexes, candidate pools, and this report.
- Extended the existing R006 C++ kernel with a backward-compatible configurable-top-K entry point; the original `selective_postings` ABI remains K10.
- Tests: population/fold/exclusion checks, official oracle edge cases, candidate ID/country/duplicate/population invariants, no-label reverse construction, exact B0 parity, and cache replay.
- Reproduce: `{PYTHON} analysis/shared_cache_v1/run_blocker_pareto_50k.py all`.
- `git diff --check`: unavailable because `.git` is empty; whitespace checks are run directly by the focused test command.
"""
    (REPORT/"REPORT.md").write_text(report)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("stage",choices=["all","prepare"]); args=parser.parse_args()
    complete=validate_complete(REPORT/"COMPLETE.manifest.json") if args.stage=="all" else None
    if complete and complete["identity"].get("evaluation_code_sha256") == sha256_file(Path(__file__)):
        log("BENCHMARK_CACHE_HIT", content_key=complete["content_key"]); return
    total=time.perf_counter(); clean,clean_manifest,sample,sample_manifest=freeze_populations()
    if args.stage=="prepare": return
    forward_keys,index_ref_seconds=register_forward_indexes(); reverse_keys,index_build_seconds,index_hits=build_reverse_indexes(sample,sample_manifest)
    forward_path,forward_manifest,forward_hit=forward_pool(sample,sample_manifest,forward_keys)
    reverse_path,reverse_manifest,reverse_hit=reverse_pool(sample,sample_manifest,reverse_keys)
    summary=evaluate(sample,clean,sample_manifest,forward_path,forward_manifest,reverse_path,reverse_manifest,index_ref_seconds+index_build_seconds,
                     {"reverse_indexes":index_hits,"forward_pool":forward_hit,"reverse_pool":reverse_hit})
    log("BENCHMARK_COMPLETE", seconds=time.perf_counter()-total, decision=summary["decision"])


if __name__=="__main__": main()
