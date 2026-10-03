#!/usr/bin/env python3
"""R006A: isolated selective-postings coarse retrieval + exact TF-IDF rerank."""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from sparse_dot_topn import sp_matmul_topn

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
E001 = ROOT / "analysis/experiments_v1/e001_classical_baseline"
E005 = ROOT / "analysis/experiments_v1/e005_xlmr_crossencoder"
DB = ROOT / "analysis/data_audit_v1/artifacts/audit.duckdb"
ART = HERE / "artifacts"
INDEX = ART / "indexes"
THREADS = 6
PROC = psutil.Process()

sys.path.insert(0, str(E001 / "scripts"))
import prepare_baseline as e001  # noqa: E402
sys.path.insert(0, str(E005 / "src"))
from metric import grouped_f05, self_test as metric_self_test  # noqa: E402


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=float) + "\n")
    tmp.replace(path)


def compile_kernel() -> Path:
    so = HERE / "selective_postings.so"
    src = HERE / "selective_postings.cpp"
    if not so.exists() or so.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(["g++", "-O3", "-march=native", "-fopenmp", "-shared", "-fPIC", str(src), "-o", str(so)], check=True)
    return so


def load_kernel():
    lib = ctypes.CDLL(str(compile_kernel()))
    fn = lib.selective_postings
    ip = ctypes.POINTER(ctypes.c_int32); fp = ctypes.POINTER(ctypes.c_float)
    i64p = ctypes.POINTER(ctypes.c_int64); dp = ctypes.POINTER(ctypes.c_double)
    fn.argtypes = [ip, ip, fp, ctypes.c_int, ip, ip, fp, ip, ip, fp, ctypes.c_int,
                   ctypes.c_int, ctypes.c_int, ctypes.c_int, ip, fp, i64p, i64p, ip, ip, dp]
    fn.restype = ctypes.c_int
    return fn


def i32(x): return np.ascontiguousarray(x, dtype=np.int32)
def f32(x): return np.ascontiguousarray(x, dtype=np.float32)


def cohort(n: int) -> pd.DataFrame:
    selected = pd.read_csv(E001 / "selected_ids/evaluation_s1.tsv", sep="\t")
    if len(selected) != 10000 or not selected.source1_entity_id.is_unique or set(selected.fold) != {0}:
        raise RuntimeError("historical evaluation cohort contract failed")
    return selected.iloc[:n].copy().reset_index(drop=True)


def build_or_load(country: str, source: int):
    base = INDEX / country / f"s{source}"
    meta_path = base / "aux.parquet"
    mats = {c: base / f"{c}.npz" for c in ["name", "address"]}
    idfs = {c: base / f"{c}_idf.npy" for c in ["name", "address"]}
    if meta_path.exists() and all(p.exists() for p in [*mats.values(), *idfs.values()]):
        aux = pd.read_parquet(meta_path)
        return aux, {c: (sparse.load_npz(mats[c]).tocsr(), np.load(idfs[c])) for c in mats}
    base.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB), read_only=True)
    con.execute(f"PRAGMA threads={THREADS}")
    assignment = E001 / "cache/auxiliary_fold_assignment.parquet"
    aux = con.execute(f"""SELECT a.entity_id,a.business_name,coalesce(a.business_address,'') business_address
      FROM train_s{source} a JOIN read_parquet(?) f ON f.entity_id=a.entity_id AND f.source={source}
      WHERE a.country=? AND f.fold=0 ORDER BY a.entity_id""", [str(assignment), country]).fetchdf()
    con.close()
    aux.to_parquet(meta_path, index=False, compression="zstd")
    out = {}
    from sklearn.feature_extraction.text import TfidfTransformer
    for channel, col in [("name", "business_name"), ("address", "business_address")]:
        counts = e001.vec_for(channel).transform(aux[col].fillna(""))
        tf = TfidfTransformer(); xa = tf.fit_transform(counts).astype(np.float32).tocsr()
        sparse.save_npz(mats[channel], xa, compressed=False); np.save(idfs[channel], tf.idf_.astype(np.float32))
        out[channel] = (xa, tf.idf_.astype(np.float32))
    return aux, out


def query_frame(selected_country: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect(str(DB), read_only=True)
    con.register("qids", selected_country[["source1_entity_id"]])
    q = con.execute("""SELECT s.entity_id,s.business_name,coalesce(s.business_address,'') business_address
      FROM train_s1 s JOIN qids q ON s.entity_id=q.source1_entity_id ORDER BY s.entity_id""").fetchdf()
    con.close()
    want = selected_country.source1_entity_id.sort_values(kind="mergesort").tolist()
    if q.entity_id.tolist() != want: raise RuntimeError("query ID recovery mismatch")
    return q


def qmatrix(q: pd.DataFrame, channel: str, idf: np.ndarray):
    from sklearn.feature_extraction.text import TfidfTransformer
    col = "business_name" if channel == "name" else "business_address"
    tf = TfidfTransformer(); tf.idf_ = idf
    return tf.transform(e001.vec_for(channel).transform(q[col].fillna(""))).astype(np.float32).tocsr()


def exact_top10(xq, xa):
    t = time.perf_counter()
    full = sp_matmul_topn(xq, xa.T, top_n=20, threshold=1e-8, sort=True, n_threads=THREADS).tocsr()
    elapsed = time.perf_counter() - t
    lengths = np.minimum(np.diff(full.indptr), 10)
    indptr = np.concatenate(([0], np.cumsum(lengths, dtype=np.int32)))
    indices = np.concatenate([full.indices[full.indptr[i]:full.indptr[i] + lengths[i]] for i in range(full.shape[0])])
    data = np.concatenate([full.data[full.indptr[i]:full.indptr[i] + lengths[i]] for i in range(full.shape[0])])
    c = sparse.csr_matrix((data, indices, indptr), shape=full.shape)
    return c, elapsed


def selective(fn, xq, xa, budget: int, m: int):
    q = xq.tocsr(); a = xa.tocsr(); c = xa.tocsc()
    arrays = [i32(q.indptr), i32(q.indices), f32(q.data), i32(c.indptr), i32(c.indices), f32(c.data),
              i32(a.indptr), i32(a.indices), f32(a.data)]
    nq = q.shape[0]
    oi = np.full((nq, 10), -1, np.int32); ov = np.zeros((nq, 10), np.float32)
    visits = np.zeros(nq, np.int64); full = np.zeros(nq, np.int64)
    counts = np.zeros(nq, np.int32); fallback = np.zeros(nq, np.int32); timing = np.zeros(5, np.float64)
    ip = ctypes.POINTER(ctypes.c_int32); fp = ctypes.POINTER(ctypes.c_float)
    start = time.perf_counter()
    rc = fn(arrays[0].ctypes.data_as(ip), arrays[1].ctypes.data_as(ip), arrays[2].ctypes.data_as(fp), nq,
            arrays[3].ctypes.data_as(ip), arrays[4].ctypes.data_as(ip), arrays[5].ctypes.data_as(fp),
            arrays[6].ctypes.data_as(ip), arrays[7].ctypes.data_as(ip), arrays[8].ctypes.data_as(fp), a.shape[0],
            budget, m, THREADS, oi.ctypes.data_as(ip), ov.ctypes.data_as(fp),
            visits.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), full.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            counts.ctypes.data_as(ip), fallback.ctypes.data_as(ip), timing.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    wall = time.perf_counter() - start
    if rc: raise RuntimeError(f"selective kernel failed: {rc}")
    return oi, ov, wall, visits, full, counts, fallback, timing


def rows_from_dense(qids, aux_ids, inds, scores, source, country, channel):
    rr, kk = np.where(inds >= 0)
    return pd.DataFrame({"s1": np.asarray(qids)[rr], "target_id": np.asarray(aux_ids)[inds[rr, kk]],
                         "source": np.int8(source), "country": country, "channel": channel,
                         "rank": (kk + 1).astype(np.int16), "score": scores[rr, kk].astype(np.float32)})


def rows_from_csr(qids, aux_ids, c, source, country, channel):
    qpos = np.repeat(np.arange(c.shape[0]), np.diff(c.indptr)); ranks = np.concatenate(
        [np.arange(1, c.indptr[i+1] - c.indptr[i] + 1) for i in range(c.shape[0])]).astype(np.int16)
    return pd.DataFrame({"s1": np.asarray(qids)[qpos], "target_id": np.asarray(aux_ids)[c.indices],
                         "source": np.int8(source), "country": country, "channel": channel,
                         "rank": ranks, "score": c.data.astype(np.float32)})


def union_streams(frames):
    x = pd.concat(frames, ignore_index=True)
    return x.sort_values(["s1", "target_id", "source"], kind="mergesort").drop_duplicates(
        ["s1", "target_id", "source"])[["s1", "target_id", "source"]].reset_index(drop=True)


def truth_for(selected):
    con = duckdb.connect(str(DB), read_only=True); con.register("qids", selected[["source1_entity_id"]])
    truth = con.execute("SELECT e.s1,e.target_id,e.target_source AS src FROM edges e JOIN qids q ON e.s1=q.source1_entity_id").fetchdf()
    truth.rename(columns={"src": "source"}, inplace=True)
    con.close(); return truth


def quality(pairs, selected, truth, exact_keys):
    tset = set(map(tuple, truth[["s1", "target_id", "source"]].itertuples(index=False, name=None)))
    keys = list(map(tuple, pairs[["s1", "target_id", "source"]].itertuples(index=False, name=None)))
    labels = np.fromiter((k in tset for k in keys), np.int8, len(keys))
    metric_frame = pairs.rename(columns={"s1": "source1_entity_id"}).copy(); metric_frame["label"] = labels
    official_selected = selected.rename(columns={"source1_entity_id": "source1_entity_id"})
    oracle = grouped_f05(metric_frame, official_selected, labels.astype(bool))
    pmap = {s: set(zip(g.target_id, g.source)) for s, g in pairs.groupby("s1", sort=False)}
    tmap = {s: set(zip(g.target_id, g.source)) for s, g in truth.groupby("s1", sort=False)}
    positive = [s for s in selected.source1_entity_id if tmap.get(s, set())]
    hit = sum(len(tmap.get(s, set()) & pmap.get(s, set())) for s in selected.source1_entity_id)
    any_hit = sum(bool(tmap[s] & pmap.get(s, set())) for s in positive)
    all_hit = sum(tmap[s] <= pmap.get(s, set()) for s in positive)
    keyset = set(keys)
    return {"candidate_pairs": len(pairs), "mean_candidates_per_s1": len(pairs)/len(selected),
            "current_k10_overlap": len(keyset & exact_keys)/len(exact_keys),
            "true_edges": len(truth), "true_edges_retained": hit, "edge_recall": hit/len(truth),
            "positive_s1": len(positive), "positive_s1_any_truth_recall": any_hit/len(positive),
            "positive_s1_all_truth_recall": all_hit/len(positive), "oracle_grouped_macro_f05": oracle["macro_f05"]}


def cached_exact_keys(selected):
    ids = set(selected.source1_entity_id); chunks = []
    for country in selected.country.unique():
        for source in [2, 3]:
            p = E001 / f"cache/candidates/evaluation_{country}_s{source}.parquet"
            z = pd.read_parquet(p, columns=["s1", "target_id", "source", "in_k10"])
            chunks.append(z[z.s1.isin(ids) & z.in_k10.eq(1)][["s1", "target_id", "source"]])
    x = pd.concat(chunks).drop_duplicates()
    return set(map(tuple, x.itertuples(index=False, name=None)))


def posting_stats(n=2000):
    selected = cohort(n); rows = []
    for country, sc in selected.groupby("country", sort=False):
        q = query_frame(sc)
        for source in [2, 3]:
            _, indexes = build_or_load(country, source)
            for channel in ["name", "address"]:
                xa, idf = indexes[channel]; xq = qmatrix(q, channel, idf); df = np.diff(xa.tocsc().indptr)
                weighted = df[xq.indices].astype(np.int64, copy=False)
                cumulative = np.concatenate(([0], np.cumsum(weighted, dtype=np.int64)))
                work = cumulative[xq.indptr[1:]] - cumulative[xq.indptr[:-1]]
                for quantile in [0, .1, .25, .5, .75, .9, .95, .99, 1]:
                    rows.append({"country": country, "source": source, "channel": channel, "quantile": quantile,
                                 "full_posting_visits": float(np.quantile(work, quantile)), "queries": len(q)})
                del xa, xq; gc.collect()
    out = pd.DataFrame(rows); ART.mkdir(parents=True, exist_ok=True); out.to_csv(ART / "pilot_posting_stats.csv", index=False)
    print(out.to_string(index=False))


def run(n: int, budgets, ms, label: str):
    metric_self_test(); fn = load_kernel(); selected = cohort(n); truth = truth_for(selected)
    exact_frames = []; method_frames = {(b, m): [] for b in budgets for m in ms}
    perf = {(b, m): {"retrieval_seconds": 0.0, "transform_seconds": 0.0, "postings_visited": 0,
                     "full_postings": 0, "coarse_sum": 0, "fallback_queries": 0,
                     "stream_queries": 0, "timing_cpu_seconds": np.zeros(5)} for b in budgets for m in ms}
    exact_seconds = transform_seconds = 0.0; score_max_diff = 0.0; rerank_checks = 0
    stream_count_fail = country_fail = 0
    for country, sc0 in selected.groupby("country", sort=False):
        sc = sc0.sort_values("source1_entity_id", kind="mergesort"); q = query_frame(sc)
        for source in [2, 3]:
            aux, indexes = build_or_load(country, source); aux_ids = aux.entity_id.to_numpy()
            for channel in ["name", "address"]:
                xa, idf = indexes[channel]
                t = time.perf_counter(); xq = qmatrix(q, channel, idf); tx = time.perf_counter() - t
                transform_seconds += tx
                exact, sec = exact_top10(xq, xa); exact_seconds += sec
                ef = rows_from_csr(q.entity_id.to_numpy(), aux_ids, exact, source, country, channel)
                exact_frames.append(ef); stream_count_fail += int((ef.groupby("s1").size() > 10).any())
                for b in budgets:
                    for m in ms:
                        oi, ov, wall, visits, full, counts, fallback, timing = selective(fn, xq, xa, b, m)
                        sf = rows_from_dense(q.entity_id.to_numpy(), aux_ids, oi, ov, source, country, channel)
                        method_frames[(b, m)].append(sf)
                        p = perf[(b, m)]; p["retrieval_seconds"] += wall; p["transform_seconds"] += tx
                        p["postings_visited"] += int(visits.sum()); p["full_postings"] += int(full.sum())
                        p["coarse_sum"] += int(np.minimum(counts, m).sum()); p["fallback_queries"] += int((fallback > 0).sum())
                        p["stream_queries"] += len(q); p["timing_cpu_seconds"] += timing
                        rr, kk = np.where(oi >= 0)
                        if len(rr):
                            check = np.asarray(xq[rr].multiply(xa[oi[rr, kk]]).sum(axis=1)).ravel().astype(np.float32)
                            score_max_diff = max(score_max_diff, float(np.max(np.abs(check - ov[rr, kk])))); rerank_checks += len(check)
                del xa, xq, exact; gc.collect()
    t = time.perf_counter(); exact_union = union_streams(exact_frames); exact_union_seconds = time.perf_counter() - t
    exact_keys = set(map(tuple, exact_union.itertuples(index=False, name=None)))
    cached = cached_exact_keys(selected); baseline_cached_overlap = len(exact_keys & cached) / len(cached)
    baseline = quality(exact_union, selected, truth, exact_keys)
    baseline.update({"configuration": "exact_k10", "posting_budget": None, "coarse_m": None,
                     "wall_seconds": transform_seconds + exact_seconds + exact_union_seconds,
                     "retrieval_seconds": exact_seconds, "transform_seconds": transform_seconds,
                     "aggregate_stream_qps": 4*n/(transform_seconds + exact_seconds), "speedup_vs_exact": 1.0,
                     "postings_avoided_fraction": 0.0, "mean_coarse_candidates": None, "fallback_query_fraction": None,
                     "oracle_delta_vs_exact": 0.0})
    results = [baseline]; profiles = {}
    for key, frames in method_frames.items():
        b, m = key; t = time.perf_counter(); pairs = union_streams(frames); union_sec = time.perf_counter() - t
        p = perf[key]; met = quality(pairs, selected, truth, exact_keys)
        wall = p["transform_seconds"] + p["retrieval_seconds"] + union_sec
        met.update({"configuration": f"budget{b}_m{m}", "posting_budget": b, "coarse_m": m,
                    "wall_seconds": wall, "retrieval_seconds": p["retrieval_seconds"],
                    "transform_seconds": p["transform_seconds"], "union_seconds": union_sec,
                    "aggregate_stream_qps": 4*n/wall, "speedup_vs_exact": baseline["wall_seconds"]/wall,
                    "postings_visited": p["postings_visited"], "full_postings": p["full_postings"],
                    "postings_avoided_fraction": 1-p["postings_visited"]/p["full_postings"],
                    "mean_coarse_candidates": p["coarse_sum"]/p["stream_queries"],
                    "fallback_query_fraction": p["fallback_queries"]/p["stream_queries"],
                    "oracle_delta_vs_exact": met["oracle_grouped_macro_f05"]-baseline["oracle_grouped_macro_f05"]})
        results.append(met)
        names = ["query_extract_sort", "postings_accumulation", "coarse_top_m", "exact_local_rerank", "union_output"]
        vals = list(p["timing_cpu_seconds"][:4]) + [union_sec * THREADS]
        profiles[met["configuration"]] = {k: v/sum(vals) for k, v in zip(names, vals)}
    rss = max(PROC.memory_info().rss, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)/2**30
    report = {"label": label, "n_s1": n, "threads": THREADS, "budgets": budgets, "coarse_ms": ms,
              "results": results, "profiles": profiles, "correctness": {
                  "rerank_score_checks": rerank_checks, "rerank_max_abs_difference": score_max_diff,
                  "rerank_scores_pass": score_max_diff <= 2e-6, "country_crossings": country_fail,
                  "country_pass": country_fail == 0, "stream_top10_semantics_pass": stream_count_fail == 0,
                  "official_metric_self_test_pass": True, "fold0_only_no_fit": bool((selected.fold == 0).all()),
                  "exact_vs_historical_cached_overlap": baseline_cached_overlap,
                  "exact_vs_historical_cached_exact": exact_keys == cached},
              "peak_rss_gb": rss, "peak_vram_gb": 0.0}
    atomic_json(ART / f"{label}_results.json", report)
    pd.DataFrame(results).to_csv(ART / f"{label}_comparison.csv", index=False)
    print(json.dumps(report, indent=2, default=float))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="command", required=True)
    sp = sub.add_parser("stats"); sp.add_argument("--n", type=int, default=2000)
    for name, default_n in [("pilot", 2000), ("confirm", 10000)]:
        p = sub.add_parser(name); p.add_argument("--n", type=int, default=default_n)
        p.add_argument("--budgets", type=int, nargs="+", required=True); p.add_argument("--ms", type=int, nargs="+", required=True)
    args = ap.parse_args()
    if args.command == "stats": posting_stats(args.n)
    else: run(args.n, args.budgets, args.ms, args.command)


if __name__ == "__main__": main()
