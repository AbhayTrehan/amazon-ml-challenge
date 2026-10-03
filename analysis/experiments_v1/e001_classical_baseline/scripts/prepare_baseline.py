#!/usr/bin/env python3
"""Prepare leak-free samples, cached K20/K10 candidates, features, and ceilings.

This script intentionally stops before fitting when neither LightGBM nor XGBoost is
available. It never writes to the supplied data or data_audit_v1 artifacts.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import resource
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import psutil
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sparse_dot_topn import sp_matmul_topn

ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "analysis/experiments_v1/e001_classical_baseline"
AUDIT = ROOT / "analysis/data_audit_v1"
DB = AUDIT / "artifacts/audit.duckdb"
SPLIT = AUDIT / "artifacts/proposed_validation_split.tsv"
SEED = 20260925
ROLE_FOLDS = {"train": [1, 2, 3], "calibration": [4], "evaluation": [0]}
ROLE_N = {"train": 30000, "calibration": 10000, "evaluation": 10000}
LEGAL = {"llc", "llp", "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "pvt", "private", "plc", "lp", "sarl", "sas", "sa", "gmbh", "bv", "pte", "firm", "enterprises", "enterprise"}
PROC = psutil.Process()
START = time.time()
PEAK = PROC.memory_info().rss


def mark_peak():
    global PEAK
    PEAK = max(PEAK, PROC.memory_info().rss)


def stable_hash(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")


def allocate_largest_remainder(counts: pd.Series, total: int) -> pd.Series:
    raw = counts / counts.sum() * total
    alloc = np.floor(raw).astype(int)
    remaining = total - int(alloc.sum())
    order = (raw - alloc).sort_values(ascending=False, kind="mergesort").index
    alloc.loc[order[:remaining]] += 1
    return alloc


def sample_ids(force=False):
    outdir = EXP / "selected_ids"
    outdir.mkdir(parents=True, exist_ok=True)
    split = pd.read_csv(SPLIT, sep="\t")
    split["cardinality_bucket"] = split.true_matches.map(lambda n: str(n) if n < 5 else "5+")
    split["singleton_vs_positive"] = np.where(split.true_matches.eq(0), "singleton", "positive")
    summaries = []
    selected_all = []
    for role, folds in ROLE_FOLDS.items():
        path = outdir / f"{role}_s1.tsv"
        if path.exists() and not force:
            chosen = pd.read_csv(path, sep="\t")
        else:
            pool = split[split.fold.isin(folds)].copy()
            keys = ["fold", "country", "singleton_vs_positive", "cardinality_bucket"]
            counts = pool.groupby(keys, observed=True).size()
            quotas = allocate_largest_remainder(counts, ROLE_N[role])
            pool["_hash"] = pool.source1_entity_id.map(stable_hash)
            chunks = []
            for key, quota in quotas.items():
                mask = np.ones(len(pool), dtype=bool)
                for col, val in zip(keys, key):
                    mask &= pool[col].eq(val).to_numpy()
                chunks.append(pool.loc[mask].nsmallest(int(quota), "_hash"))
            chosen = pd.concat(chunks, ignore_index=True).drop(columns="_hash")
            chosen = chosen.sort_values("source1_entity_id", kind="mergesort")
            assert len(chosen) == ROLE_N[role] and chosen.source1_entity_id.is_unique
            chosen.to_csv(path, sep="\t", index=False)
        chosen["role"] = role
        selected_all.append(chosen)
        dist = chosen.groupby(["role", "fold", "country", "cardinality_bucket"], observed=True).size().rename("s1_count").reset_index()
        summaries.append(dist)
    pd.concat(summaries, ignore_index=True).to_csv(outdir / "sampling_distribution.csv", index=False)
    pd.concat(selected_all, ignore_index=True).to_parquet(outdir / "selected_s1.parquet", index=False)
    return pd.concat(selected_all, ignore_index=True)


def build_aux_assignment(selected: pd.DataFrame, force=False):
    path = EXP / "cache/auxiliary_fold_assignment.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return path
    con = duckdb.connect(str(DB), read_only=True)
    con.execute("PRAGMA threads=6")
    con.execute("PRAGMA memory_limit='3GB'")
    split = pd.read_csv(SPLIT, sep="\t", usecols=["source1_entity_id", "fold"])
    con.register("split_map", split)
    # All matched leaves inherit their unique owner's fold. Unmatched leaves use a
    # deterministic hash fold so no physical auxiliary row crosses roles.
    q = f"""
      COPY (
        WITH owned AS (
          SELECT e.target_id, min(s.fold)::UTINYINT AS fold
          FROM edges e JOIN split_map s ON e.s1=s.source1_entity_id GROUP BY 1
        ), aux AS (
          SELECT 2::UTINYINT AS src, entity_id FROM train_s2
          UNION ALL SELECT 3::UTINYINT AS src, entity_id FROM train_s3
        )
        SELECT a.src AS source, a.entity_id,
               coalesce(o.fold, hash(a.entity_id, {SEED}) % 5)::UTINYINT AS fold
        FROM aux a LEFT JOIN owned o ON a.entity_id=o.target_id
      ) TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con.execute(q)
    con.close()
    return path


def vec_for(channel):
    if channel == "name":
        return HashingVectorizer(n_features=2**18, analyzer="char_wb", ngram_range=(3, 5), alternate_sign=False, norm=None, dtype=np.float32)
    return HashingVectorizer(n_features=2**17, analyzer="char_wb", ngram_range=(3, 3), alternate_sign=False, norm=None, dtype=np.float32)


def make_matrix(aux_text, query_text, channel):
    vec = vec_for(channel)
    xa = vec.transform(aux_text)
    xq = vec.transform(query_text)
    tf = TfidfTransformer()
    xa = tf.fit_transform(xa).astype(np.float32)
    xq = tf.transform(xq).astype(np.float32)
    return xa, xq


def retrieve(xq, xa, channel):
    c = sp_matmul_topn(xq, xa.T, top_n=20, threshold=1e-8, sort=True, n_threads=6)
    qpos = np.repeat(np.arange(c.shape[0], dtype=np.int32), np.diff(c.indptr))
    ranks = np.concatenate([np.arange(1, c.indptr[i+1]-c.indptr[i]+1, dtype=np.int16) for i in range(c.shape[0])])
    return pd.DataFrame({"qpos": qpos, "apos": c.indices.astype(np.int32), f"{channel}_rank": ranks, f"{channel}_retrieval_score": c.data.astype(np.float32)})


def row_cosines(xq, xa, qpos, apos, batch=50000):
    out = np.empty(len(qpos), dtype=np.float32)
    for start in range(0, len(qpos), batch):
        stop = min(len(qpos), start + batch)
        out[start:stop] = np.asarray(xq[qpos[start:stop]].multiply(xa[apos[start:stop]]).sum(axis=1)).ravel()
    return out


def toks(s):
    return str(s).split() if s else []


def suffix(s):
    x = toks(s)
    while x and x[-1] in LEGAL:
        x.pop()
    return " ".join(x)


def accent(s):
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if unicodedata.category(c) != "Mn")


def jac(a, b):
    x, y = set(toks(a)), set(toks(b))
    return len(x & y) / len(x | y) if x or y else 1.0


def num_stats(a, b):
    x = {z for z in toks(a) if z.isdigit()}
    y = {z for z in toks(b) if z.isdigit()}
    inter = len(x & y)
    union = len(x | y)
    return (inter / union if union else 1.0, inter, len(x), len(y), int(bool(x) and bool(y) and inter == 0))


def ratio_len(a, b):
    la, lb = len(str(a)), len(str(b))
    return min(la, lb) / max(la, lb) if max(la, lb) else 1.0


def add_context(cand):
    cand["retrieval_channels"] = cand.name_rank.notna().astype(np.int8) + cand.address_rank.notna().astype(np.int8)
    cand["retrieved_by_name"] = cand.name_rank.notna().astype(np.int8)
    cand["retrieved_by_address"] = cand.address_rank.notna().astype(np.int8)
    cand["name_rank"] = cand.name_rank.fillna(99).astype(np.int16)
    cand["address_rank"] = cand.address_rank.fillna(99).astype(np.int16)
    cand["name_retrieval_score"] = cand.name_retrieval_score.fillna(0).astype(np.float32)
    cand["address_retrieval_score"] = cand.address_retrieval_score.fillna(0).astype(np.float32)
    cand["reciprocal_name_rank"] = np.where(cand.name_rank < 99, 1 / cand.name_rank, 0).astype(np.float32)
    cand["reciprocal_address_rank"] = np.where(cand.address_rank < 99, 1 / cand.address_rank, 0).astype(np.float32)
    cand["fused_max_score"] = cand[["name_char_tfidf", "address_char_tfidf"]].max(axis=1).astype(np.float32)
    cand["fused_sum_score"] = cand[["name_char_tfidf", "address_char_tfidf"]].sum(axis=1).astype(np.float32)
    grp = cand.groupby("qpos", sort=False)
    cand["candidate_count_k20_source"] = grp.qpos.transform("size").astype(np.int16)
    for col, prefix in [("name_char_tfidf", "name"), ("address_char_tfidf", "address"), ("fused_max_score", "fused")]:
        top = grp[col].transform("max")
        # second distinct/row score; ties intentionally yield margin zero.
        second = grp[col].transform(lambda s: s.nlargest(2).iloc[-1] if len(s) > 1 else 0.0)
        cand[f"query_{prefix}_top12_margin"] = (top - second).astype(np.float32)
    topf = grp.fused_max_score.transform("max")
    cand["near_tie_count_002"] = (cand.fused_max_score.ge(topf - 0.02)).groupby(cand.qpos).transform("sum").astype(np.int16)
    cand["in_k10"] = ((cand.name_rank <= 10) | (cand.address_rank <= 10)).astype(np.int8)
    return cand


def hard_negative_mask(cand):
    if cand.role.iloc[0] != "train":
        return np.ones(len(cand), dtype=bool)
    cand["_priority"] = (
        cand.nname_exact.astype(np.float32) * 10
        + cand.retrieval_channels * 2
        + cand.fused_max_score
        + 0.25 * cand.fused_sum_score
    )
    pos = cand[cand.label.eq(1)].index
    neg = cand[cand.label.eq(0)].sort_values(
        ["qpos", "_priority", "target_id"], ascending=[True, False, True], kind="mergesort"
    )
    neg = neg[neg.groupby("qpos", sort=False).cumcount() < 15].index
    return cand.index.isin(pos.union(neg))


def build_features(cand, q, aux):
    qn = q.nname.to_numpy()[cand.qpos]
    qa = q.naddr.fillna("").to_numpy()[cand.qpos]
    an = aux.nname.to_numpy()[cand.apos]
    aa = aux.naddr.fillna("").to_numpy()[cand.apos]
    qr_n = q.business_name.fillna("").to_numpy()[cand.qpos]
    qr_a = q.business_address.fillna("").to_numpy()[cand.qpos]
    ar_n = aux.business_name.fillna("").to_numpy()[cand.apos]
    ar_a = aux.business_address.fillna("").to_numpy()[cand.apos]
    out = cand.copy()
    out["legal_suffix_name_exact"] = np.fromiter((bool(suffix(a)) and suffix(a) == suffix(b) for a,b in zip(qn,an)), np.int8, len(out))
    out["accent_name_exact"] = np.fromiter((bool(accent(a)) and accent(a) == accent(b) for a,b in zip(qn,an)), np.int8, len(out))
    out["name_token_jaccard"] = np.fromiter((jac(a,b) for a,b in zip(qn,an)), np.float32, len(out))
    out["address_token_jaccard"] = np.fromiter((jac(a,b) for a,b in zip(qa,aa)), np.float32, len(out))
    out["name_length_ratio"] = np.fromiter((ratio_len(a,b) for a,b in zip(qn,an)), np.float32, len(out))
    out["address_length_ratio"] = np.fromiter((ratio_len(a,b) for a,b in zip(qa,aa)), np.float32, len(out))
    out["name_length_diff"] = np.fromiter((abs(len(a)-len(b)) for a,b in zip(qn,an)), np.int16, len(out))
    out["address_length_diff"] = np.fromiter((abs(len(a)-len(b)) for a,b in zip(qa,aa)), np.int16, len(out))
    out["name_token_count_diff"] = np.fromiter((abs(len(toks(a))-len(toks(b))) for a,b in zip(qn,an)), np.int16, len(out))
    out["address_token_count_diff"] = np.fromiter((abs(len(toks(a))-len(toks(b))) for a,b in zip(qa,aa)), np.int16, len(out))
    out["name_rf_ratio"] = np.fromiter((fuzz.ratio(a,b)/100 for a,b in zip(qn,an)), np.float32, len(out))
    out["name_rf_partial_ratio"] = np.fromiter((fuzz.partial_ratio(a,b)/100 for a,b in zip(qn,an)), np.float32, len(out))
    out["name_rf_token_set"] = np.fromiter((fuzz.token_set_ratio(a,b)/100 for a,b in zip(qn,an)), np.float32, len(out))
    out["name_rf_token_sort"] = np.fromiter((fuzz.token_sort_ratio(a,b)/100 for a,b in zip(qn,an)), np.float32, len(out))
    out["name_jaro_winkler"] = np.fromiter((JaroWinkler.normalized_similarity(a,b) for a,b in zip(qn,an)), np.float32, len(out))
    out["address_rf_ratio"] = np.fromiter((fuzz.ratio(a,b)/100 for a,b in zip(qa,aa)), np.float32, len(out))
    out["address_rf_token_set"] = np.fromiter((fuzz.token_set_ratio(a,b)/100 for a,b in zip(qa,aa)), np.float32, len(out))
    ns = [num_stats(a,b) for a,b in zip(qa,aa)]
    out["numeric_token_jaccard"] = np.asarray([x[0] for x in ns], np.float32)
    out["numeric_exact_overlap_count"] = np.asarray([x[1] for x in ns], np.int16)
    out["s1_numeric_token_count"] = np.asarray([x[2] for x in ns], np.int16)
    out["candidate_numeric_token_count"] = np.asarray([x[3] for x in ns], np.int16)
    out["obvious_numeric_conflict"] = np.asarray([x[4] for x in ns], np.int8)
    out["s1_address_missing"] = np.fromiter((not bool(x) for x in qr_a), np.int8, len(out))
    out["candidate_address_missing"] = np.fromiter((not bool(x) for x in ar_a), np.int8, len(out))
    out["source3"] = out.source.eq(3).astype(np.int8)
    out.drop(columns=["qpos", "apos", "_priority"], errors="ignore", inplace=True)
    return out


def process_shard(role, country, source, selected, assignment, force=False):
    cpath = EXP / f"cache/candidates/{role}_{country}_s{source}.parquet"
    fpath = EXP / f"cache/features/{role}_{country}_s{source}.parquet"
    cpath.parent.mkdir(parents=True, exist_ok=True)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    if cpath.exists() and fpath.exists() and not force:
        print("cached", role, country, source, flush=True)
        return
    con = duckdb.connect(str(DB), read_only=True)
    con.execute("PRAGMA threads=6")
    con.execute("PRAGMA memory_limit='3GB'")
    ids = selected[(selected.role == role) & (selected.country == country)][["source1_entity_id"]]
    con.register("qids", ids)
    q = con.execute("""SELECT s.entity_id, s.business_name, coalesce(s.business_address,'') business_address,
                              s.nname, coalesce(s.naddr,'') naddr
                       FROM train_s1 s JOIN qids q ON s.entity_id=q.source1_entity_id
                       ORDER BY s.entity_id""").fetchdf()
    folds = ",".join(map(str, ROLE_FOLDS[role]))
    aux = con.execute(f"""SELECT a.entity_id, a.business_name, coalesce(a.business_address,'') business_address,
                                  a.nname, coalesce(a.naddr,'') naddr
                           FROM train_s{source} a
                           JOIN read_parquet('{assignment}') f ON f.entity_id=a.entity_id AND f.source={source}
                           WHERE a.country=? AND f.fold IN ({folds}) ORDER BY a.entity_id""", [country]).fetchdf()
    truth = con.execute("SELECT e.s1, e.target_id FROM edges e JOIN qids q ON e.s1=q.source1_entity_id WHERE e.target_source=?", [source]).fetchdf()
    truth_set = set(zip(truth.s1, truth.target_id))
    con.close()
    print(f"shard {role} {country} S{source}: q={len(q):,} aux={len(aux):,}", flush=True)
    qnamefreq = aux.nname.value_counts(dropna=False)
    qaddrfreq = aux.naddr.value_counts(dropna=False)
    q_name_fan = q.nname.map(qnamefreq).fillna(0).clip(0, 65535).astype(np.uint16).to_numpy()
    q_addr_fan = q.naddr.map(qaddrfreq).fillna(0).clip(0, 65535).astype(np.uint16).to_numpy()
    a_name_fan = aux.nname.map(qnamefreq).fillna(0).clip(0, 65535).astype(np.uint16).to_numpy()
    a_addr_fan = aux.naddr.map(qaddrfreq).fillna(0).clip(0, 65535).astype(np.uint16).to_numpy()

    matrices = {}
    rets = []
    for channel, col in [("name", "business_name"), ("address", "business_address")]:
        t = time.time()
        xa, xq = make_matrix(aux[col].fillna(""), q[col].fillna(""), channel)
        matrices[channel] = (xa, xq)
        rets.append(retrieve(xq, xa, channel))
        mark_peak()
        print(f"  {channel}: nnz={xa.nnz:,}, seconds={time.time()-t:.1f}, rss_gb={PROC.memory_info().rss/2**30:.2f}", flush=True)
    cand = rets[0].merge(rets[1], on=["qpos", "apos"], how="outer")
    cand["s1"] = q.entity_id.to_numpy()[cand.qpos]
    cand["target_id"] = aux.entity_id.to_numpy()[cand.apos]
    cand["source"] = np.int8(source)
    cand["country"] = country
    cand["role"] = role
    for channel in ["name", "address"]:
        xa, xq = matrices[channel]
        cand[f"{channel}_char_tfidf"] = row_cosines(xq, xa, cand.qpos.to_numpy(), cand.apos.to_numpy())
    cand["s1_normalized_name_fanout"] = q_name_fan[cand.qpos]
    cand["s1_normalized_address_fanout"] = q_addr_fan[cand.qpos]
    cand["candidate_normalized_name_fanout"] = a_name_fan[cand.apos]
    cand["candidate_normalized_address_fanout"] = a_addr_fan[cand.apos]
    qn = q.nname.to_numpy()[cand.qpos]
    qa = q.naddr.to_numpy()[cand.qpos]
    an = aux.nname.to_numpy()[cand.apos]
    aa = aux.naddr.to_numpy()[cand.apos]
    cand["nname_exact"] = ((qn != "") & (qn == an)).astype(np.int8)
    cand["naddr_exact"] = ((qa != "") & (qa == aa)).astype(np.int8)
    cand["label"] = np.fromiter(((s,t) in truth_set for s,t in zip(cand.s1,cand.target_id)), np.int8, len(cand))
    cand = add_context(cand)
    keepcols = [c for c in cand.columns if c not in ["_priority"]]
    cand[keepcols].to_parquet(cpath, index=False, compression="zstd")
    keep = hard_negative_mask(cand)
    feat = build_features(cand.loc[keep].copy(), q, aux)
    feat.to_parquet(fpath, index=False, compression="zstd")
    print(f"  union={len(cand):,}, features={len(feat):,}, positives={int(feat.label.sum()):,}", flush=True)
    del matrices, rets, cand, feat, q, aux
    gc.collect(); mark_peak()


def f05(truth, pred):
    truth, pred = set(truth), set(pred)
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(truth & pred)
    return 1.25 * tp / (1.25 * tp + .25 * (len(truth)-tp) + (len(pred)-tp))


def candidate_stats(selected):
    con = duckdb.connect(str(DB), read_only=True)
    rows = []
    for role in ["calibration", "evaluation"]:
        ids = selected[selected.role.eq(role)][["source1_entity_id", "true_matches"]]
        con.register("qids", ids)
        truth = con.execute("SELECT q.source1_entity_id s1,e.target_id FROM qids q LEFT JOIN edges e ON q.source1_entity_id=e.s1").fetchdf()
        tmap = {s: set(g.target_id.dropna()) for s,g in truth.groupby("s1", sort=False)}
        files = sorted((EXP / "cache/candidates").glob(f"{role}_*_s*.parquet"))
        c = pd.concat([pd.read_parquet(p, columns=["s1", "target_id", "in_k10"]) for p in files], ignore_index=True)
        for k in [10, 20]:
            z = c if k == 20 else c[c.in_k10.eq(1)]
            pmap = {s: set(g.target_id) for s,g in z.groupby("s1", sort=False)}
            counts = z.groupby("s1").size().reindex(ids.source1_entity_id, fill_value=0)
            edge_total = edge_hit = pos = anyhit = allhit = 0
            scores = []
            for s in ids.source1_entity_id:
                tr = tmap.get(s, set()); pr = pmap.get(s, set())
                edge_total += len(tr); edge_hit += len(tr & pr)
                if tr:
                    pos += 1; anyhit += bool(tr & pr); allhit += tr <= pr
                scores.append(f05(tr, tr & pr))
            rows.append({
                "role": role, "candidate_set": f"K{k}", "s1_count": len(ids), "total_candidate_pairs": len(z),
                "mean_candidates_per_s1": counts.mean(), "p50": counts.quantile(.5), "p90": counts.quantile(.9),
                "p95": counts.quantile(.95), "p99": counts.quantile(.99), "max": counts.max(),
                "true_edges": edge_total, "true_edges_surviving": edge_hit, "edge_recall": edge_hit/edge_total,
                "positive_s1": pos, "positive_s1_any_truth": anyhit, "positive_s1_any_truth_recall": anyhit/pos,
                "positive_s1_all_truth": allhit, "positive_s1_all_truth_recall": allhit/pos,
                "oracle_grouped_macro_f05": np.mean(scores)
            })
    con.close()
    pd.DataFrame(rows).to_csv(EXP / "candidate_stats.csv", index=False)
    return pd.DataFrame(rows)


def write_schema_and_manifest(selected, stats):
    files = sorted((EXP / "cache/features").glob("*.parquet"))
    schema = pd.read_parquet(files[0]).dtypes.astype(str).to_dict()
    non_features = {"s1", "target_id", "country", "role", "label", "in_k10"}
    payload = {"columns": [{"name": k, "dtype": v, "model_feature": k not in non_features} for k,v in schema.items()], "label": "label", "notes": "K10 is derived from cached K20 retrieval; training feature rows contain all surviving positives plus at most 15 ranked hard negatives per S1 per source (30 total)."}
    (EXP / "feature_schema.json").write_text(json.dumps(payload, indent=2))
    train_counts = []
    for p in files:
        x = pd.read_parquet(p, columns=["role", "label"])
        train_counts.append({"file": p.name, "rows": len(x), "positives": int(x.label.sum()), "negatives": int((x.label==0).sum())})
    pd.DataFrame(train_counts).to_csv(EXP / "feature_artifact_counts.csv", index=False)
    deps = {m: bool(importlib.util.find_spec(m)) for m in ["lightgbm", "xgboost"]}
    runtime = {"status": "blocked_before_training" if not any(deps.values()) else "ready_for_training", "classifier_dependencies": deps, "elapsed_seconds": time.time()-START, "peak_rss_gb": PEAK/2**30, "ru_maxrss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20, "python": sys.version, "platform": platform.platform()}
    (EXP / "runtime.json").write_text(json.dumps(runtime, indent=2))
    (EXP / "model").mkdir(exist_ok=True)
    (EXP / "predictions").mkdir(exist_ok=True)
    blocker = EXP / "model/DEPENDENCY_BLOCKER.json"
    if not any(deps.values()):
        blocker.write_text(json.dumps({"required_preferred": "lightgbm", "allowed_fallback": "xgboost", "detected": deps, "action": "No model was trained, per experimental contract."}, indent=2))
    for name, cols in [("calibration_results.csv", ["experiment","candidate_set","status","macro_f05"]), ("evaluation_results.csv", ["experiment","candidate_set","status","macro_f05","oracle_macro_f05","oracle_actual_gap"]), ("slice_results.csv", ["experiment","slice","value","status","macro_f05"]), ("feature_importance.csv", ["feature","importance_gain","status"])]:
        pd.DataFrame(columns=cols).to_csv(EXP/name, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--sample-only", action="store_true")
    args = ap.parse_args()
    for d in ["selected_ids", "cache/candidates", "cache/features", "model", "predictions"]:
        (EXP / d).mkdir(parents=True, exist_ok=True)
    selected = sample_ids(args.force)
    assignment = build_aux_assignment(selected, args.force)
    if args.sample_only:
        return
    for role in ["train", "calibration", "evaluation"]:
        for country in ["India", "US"]:
            for source in [2, 3]:
                process_shard(role, country, source, selected, assignment, args.force)
    stats = candidate_stats(selected)
    write_schema_and_manifest(selected, stats)
    print(stats.to_string(index=False), flush=True)
    runtime = json.loads((EXP / "runtime.json").read_text())
    if runtime["status"] == "blocked_before_training":
        print("DEPENDENCY BLOCKER: neither LightGBM nor XGBoost is installed; stopping before training as required.", flush=True)


if __name__ == "__main__":
    main()
