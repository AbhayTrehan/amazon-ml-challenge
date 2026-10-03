#!/usr/bin/env python3
"""P002 frozen R006B + E005B resumable production stages."""
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
import subprocess
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

ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "analysis/production/p002_final"
P001 = ROOT / "analysis/production/p001_e005b"
INDEX_ROOT = P001 / "cache/indexes"  # read-only reuse; P001 is never modified
E001 = ROOT / "analysis/experiments_v1/e001_classical_baseline"
E005 = ROOT / "analysis/experiments_v1/e005_xlmr_crossencoder"
R006 = ROOT / "analysis/experiments_v1/r006a_selective_postings"
R007 = ROOT / "analysis/experiments_v1/r007_fast_blocker_e005b"
DB = ROOT / "analysis/data_audit_v1/artifacts/audit.duckdb"
TEST = ROOT / "student_resource/dataset/test"
MODEL_TXT = E001 / "model/lightgbm_model.txt"
MODEL_JOBLIB = E001 / "model/lightgbm_model.joblib"
MODEL_CONFIG = E001 / "model/model_config.json"
XL_CHECKPOINT = E005 / "model/best_checkpoint"
SHARD_SIZE = 20_000
THREADS = 20
COUNTRIES = ["France", "India", "US"]
EXPECTED_FILES = {
    "test_source1.tsv": (1_732_544, "3d4a32c54c2ca9c53fd7c2be105bf26f708f94c4d2f88eb370972a195665c2f5"),
    "test_source2.tsv": (4_887_273, "79d906c7497af2ace70aa277f6e334a652094909de99bd6c57b53420b6a7b2dd"),
    "test_source3.tsv": (5_082_316, "850942b11d2a4343486ed0834e28bce9f3b385f3fd497fd60ccf4ea3b8bda035"),
}

sys.path.insert(0, str(E001 / "scripts")); import prepare_baseline as e001  # noqa: E402
sys.path.insert(0, str(R006)); import run_r006a, run_r006b  # noqa: E402
sys.path.insert(0, str(E005 / "src")); from evaluate_e005 import mask_for  # noqa: E402
from metric import grouped_f05  # noqa: E402
from feature_builder import add_context, build_features  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""): h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=float) + "\n"); tmp.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False, compression="zstd"); tmp.replace(path)


def append_log(name: str, row) -> None:
    path = EXP / "logs" / name; path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f: f.write(json.dumps(row, default=float) + "\n")


def features() -> list[str]: return json.loads(MODEL_CONFIG.read_text())["features"]
def booster(): return lgb.Booster(model_file=str(MODEL_TXT))
def query_path(country): return INDEX_ROOT / country / "queries.parquet"
def aux_path(country, source): return INDEX_ROOT / country / f"s{source}" / "aux.parquet"
def matrix_path(country, source, channel): return INDEX_ROOT / country / f"s{source}" / f"{channel}.npz"
def idf_path(country, source, channel): return INDEX_ROOT / country / f"s{source}" / f"{channel}_idf.npy"


def tasks():
    for country in COUNTRIES:
        n = math.ceil(len(pd.read_parquet(query_path(country), columns=["entity_id"])) / SHARD_SIZE)
        for source in [2, 3]:
            for part in range(n): yield country, source, part


class FrozenSelectiveIndex:
    """R006B kernel wrapper retaining CSR/CSC arrays across query chunks."""
    def __init__(self, matrix):
        self.a = matrix.tocsr(); c = self.a.tocsc()
        self.c_indptr = run_r006a.i32(c.indptr); self.c_indices = run_r006a.i32(c.indices); self.c_data = run_r006a.f32(c.data)
        self.a_indptr = run_r006a.i32(self.a.indptr); self.a_indices = run_r006a.i32(self.a.indices); self.a_data = run_r006a.f32(self.a.data)
        self.fn, _ = run_r006b.load_functions()

    def query(self, xq):
        q = xq.tocsr(); qip = run_r006a.i32(q.indptr); qix = run_r006a.i32(q.indices); qv = run_r006a.f32(q.data)
        nq = q.shape[0]; oi = np.full((nq, 10), -1, np.int32); ov = np.zeros((nq, 10), np.float32)
        visits = np.zeros(nq, np.int64); full = np.zeros(nq, np.int64); counts = np.zeros(nq, np.int32); fallback = np.zeros(nq, np.int32); timing = np.zeros(5, np.float64)
        ip = ctypes.POINTER(ctypes.c_int32); fp = ctypes.POINTER(ctypes.c_float); i64p = ctypes.POINTER(ctypes.c_int64); dp = ctypes.POINTER(ctypes.c_double)
        start = time.perf_counter()
        rc = self.fn(qip.ctypes.data_as(ip), qix.ctypes.data_as(ip), qv.ctypes.data_as(fp), nq,
            self.c_indptr.ctypes.data_as(ip), self.c_indices.ctypes.data_as(ip), self.c_data.ctypes.data_as(fp),
            self.a_indptr.ctypes.data_as(ip), self.a_indices.ctypes.data_as(ip), self.a_data.ctypes.data_as(fp), self.a.shape[0],
            20_000, 800, 20, oi.ctypes.data_as(ip), ov.ctypes.data_as(fp), visits.ctypes.data_as(i64p),
            full.ctypes.data_as(i64p), counts.ctypes.data_as(ip), fallback.ctypes.data_as(ip), timing.ctypes.data_as(dp))
        if rc: raise RuntimeError(f"R006B kernel failed: {rc}")
        return oi, ov, time.perf_counter() - start


def qmatrix(q, channel, idf):
    tf = TfidfTransformer(); tf.idf_ = idf
    col = "business_name" if channel == "name" else "business_address"
    return tf.transform(e001.vec_for(channel).transform(q[col].fillna(""))).astype(np.float32).tocsr()


def stream(inds, scores, channel):
    rr, kk = np.where(inds >= 0)
    return pd.DataFrame({"qpos": rr.astype(np.int32), "apos": inds[rr, kk].astype(np.int32),
        f"{channel}_rank": (kk + 1).astype(np.int16), f"{channel}_retrieval_score": scores[rr, kk].astype(np.float32)})


def retrieve_partition(country: str, source: int):
    qfull = pd.read_parquet(query_path(country)); aux = pd.read_parquet(aux_path(country, source))
    indexes = {}; idfs = {}
    for channel in ["name", "address"]:
        mat = sparse.load_npz(matrix_path(country, source, channel)).tocsr(); indexes[channel] = FrozenSelectiveIndex(mat)
        idfs[channel] = np.load(idf_path(country, source, channel))
    for part, start in enumerate(range(0, len(qfull), SHARD_SIZE)):
        op = EXP / f"cache/blocker/{country}/s{source}/part-{part:05d}.parquet"
        mp = op.with_suffix(".meta.json")
        if op.exists() and mp.exists(): continue
        q = qfull.iloc[start:start + SHARD_SIZE].reset_index(drop=True); mats = {}; frames = []; kernel = 0.0; t0 = time.perf_counter()
        for channel in ["name", "address"]:
            xq = qmatrix(q, channel, idfs[channel]); mats[channel] = xq
            oi, ov, sec = indexes[channel].query(xq); kernel += sec; frames.append(stream(oi, ov, channel))
        cand = frames[0].merge(frames[1], on=["qpos", "apos"], how="outer")
        cand["s1"] = q.entity_id.to_numpy()[cand.qpos]; cand["target_id"] = aux.entity_id.to_numpy()[cand.apos]
        for channel in ["name", "address"]:
            cand[f"{channel}_char_tfidf"] = e001.row_cosines(mats[channel], indexes[channel].a, cand.qpos.to_numpy(), cand.apos.to_numpy())
        atomic_parquet(cand, op)
        meta = {"country": country, "source": source, "part": part, "s1": len(q), "pairs": len(cand),
            "kernel_seconds": kernel, "wall_seconds": time.perf_counter() - t0, "threads": 20, "budget": 20000, "m": 800}
        atomic_json(mp, meta); append_log("retrieval_parts.jsonl", meta); print("BLOCKER_READY", json.dumps(meta), flush=True)
        del q, mats, frames, cand; gc.collect()
    del qfull, aux, indexes; gc.collect()


def retrieve_full():
    start = time.time()
    for country in COUNTRIES:
        for source in [2, 3]: retrieve_partition(country, source)
    (EXP / "runtime/retrieval.done").write_text(f"{time.time()}\n")
    atomic_json(EXP / "diagnostics/retrieval_invocation.json", {"seconds": time.time() - start})


def enrich(cand, q, aux, source, country, nf=None, af=None):
    cand["source"] = np.int8(source); cand["country"] = country; cand["role"] = "test"; cand["label"] = np.int8(0)
    if nf is None: nf = aux.nname.value_counts(dropna=False)
    if af is None: af = aux.naddr.value_counts(dropna=False)
    cand["s1_normalized_name_fanout"] = q.nname.map(nf).fillna(0).clip(0,65535).astype(np.uint16).to_numpy()[cand.qpos]
    cand["s1_normalized_address_fanout"] = q.naddr.map(af).fillna(0).clip(0,65535).astype(np.uint16).to_numpy()[cand.qpos]
    cand["candidate_normalized_name_fanout"] = aux.candidate_normalized_name_fanout.to_numpy()[cand.apos]
    cand["candidate_normalized_address_fanout"] = aux.candidate_normalized_address_fanout.to_numpy()[cand.apos]
    qn=q.nname.to_numpy()[cand.qpos]; qa=q.naddr.to_numpy()[cand.qpos]; an=aux.nname.to_numpy()[cand.apos]; aa=aux.naddr.to_numpy()[cand.apos]
    cand["nname_exact"] = ((qn != "") & (qn == an)).astype(np.int8); cand["naddr_exact"] = ((qa != "") & (qa == aa)).astype(np.int8)
    return add_context(cand)


def feature_one(country, source, part, model, fs, workers=1, cache=None):
    ip = EXP / f"cache/blocker/{country}/s{source}/part-{part:05d}.parquet"
    op = EXP / f"cache/lightgbm/{country}/s{source}/part-{part:05d}.parquet"; mp = op.with_suffix(".meta.json")
    if op.exists() and mp.exists(): return
    if cache is None:
        qfull = pd.read_parquet(query_path(country)); aux = pd.read_parquet(aux_path(country, source)); nf=aux.nname.value_counts(dropna=False);af=aux.naddr.value_counts(dropna=False)
    else:
        qfull,aux,nf,af=cache
    q = qfull.iloc[part * SHARD_SIZE:(part + 1) * SHARD_SIZE].reset_index(drop=True); cand = pd.read_parquet(ip)
    t0 = time.perf_counter(); cand = enrich(cand, q, aux, source, country, nf, af); context_s = time.perf_counter() - t0
    t = time.perf_counter(); feat = build_features(cand, q, aux, workers=workers); feature_s = time.perf_counter() - t
    t = time.perf_counter(); scores = model.predict(feat[fs], num_threads=1).astype(np.float32); lgb_s = time.perf_counter() - t
    if not np.isfinite(scores).all(): raise RuntimeError("non-finite LightGBM score")
    cols = ["s1","target_id","source","country","name_rank","name_retrieval_score","address_rank","address_retrieval_score","retrieved_by_name","retrieved_by_address"]
    out = cand[cols].copy(); out["lightgbm_score"] = scores; atomic_parquet(out, op)
    meta = {"country": country, "source": source, "part": part, "pairs": len(out), "context_seconds": context_s,
        "feature_seconds": feature_s, "lightgbm_seconds": lgb_s, "wall_seconds": time.perf_counter() - t0}
    atomic_json(mp, meta); append_log("feature_parts.jsonl", meta); print("FEATURE_READY", json.dumps(meta), flush=True)


def feature_worker(name: str):
    model = booster(); fs = features(); idle = 0; current=None;cache=None
    while True:
        progressed = False
        for country, source, part in tasks():
            op = EXP / f"cache/lightgbm/{country}/s{source}/part-{part:05d}.parquet"
            if op.exists() and op.with_suffix(".meta.json").exists(): continue
            ip = EXP / f"cache/blocker/{country}/s{source}/part-{part:05d}.parquet"
            if not ip.exists(): continue
            lock = op.with_suffix(".lock"); lock.parent.mkdir(parents=True, exist_ok=True)
            try: fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
            except FileExistsError: continue
            try:
                if current!=(country,source):
                    cache=None;gc.collect();qfull=pd.read_parquet(query_path(country));aux=pd.read_parquet(aux_path(country,source));nf=aux.nname.value_counts(dropna=False);af=aux.naddr.value_counts(dropna=False);cache=(qfull,aux,nf,af);current=(country,source)
                feature_one(country, source, part, model, fs, cache=cache); progressed = True
            finally: lock.unlink(missing_ok=True)
        missing = sum(not (EXP / f"cache/lightgbm/{c}/s{s}/part-{p:05d}.parquet").exists() for c,s,p in tasks())
        if missing == 0 and (EXP / "runtime/retrieval.done").exists(): break
        if not progressed: time.sleep(2); idle += 1
    print("FEATURE_WORKER_DONE", name, flush=True)


def build_shortlist():
    start_all = time.time(); total = 0
    for country in COUNTRIES:
        qfull = pd.read_parquet(query_path(country)); aux = {s: pd.read_parquet(aux_path(country,s)).set_index("entity_id") for s in [2,3]}
        n = math.ceil(len(qfull) / SHARD_SIZE)
        for part in range(n):
            op = EXP / f"cache/shortlist/{country}/part-{part:05d}.parquet"; mp = op.with_suffix(".meta.json")
            if op.exists() and mp.exists(): continue
            frames = [pd.read_parquet(EXP / f"cache/lightgbm/{country}/s{s}/part-{part:05d}.parquet") for s in [2,3]]
            d = pd.concat(frames, ignore_index=True).sort_values(["s1","lightgbm_score","target_id"], ascending=[True,False,True], kind="mergesort")
            d["lightgbm_rank_within_s1"] = (d.groupby("s1", sort=False).cumcount() + 1).astype(np.int16)
            d = d[d.lightgbm_rank_within_s1.le(5) | d.lightgbm_score.ge(.05)].copy()
            q = qfull.iloc[part*SHARD_SIZE:(part+1)*SHARD_SIZE].set_index("entity_id")
            d["s1_business_name"] = d.s1.map(q.business_name); d["s1_business_address"] = d.s1.map(q.business_address).fillna("")
            d["candidate_business_name"] = ""; d["candidate_business_address"] = ""
            for s in [2,3]:
                take = d.source.eq(s); ids = d.loc[take,"target_id"]
                d.loc[take,"candidate_business_name"] = ids.map(aux[s].business_name).to_numpy()
                d.loc[take,"candidate_business_address"] = ids.map(aux[s].business_address).fillna("").to_numpy()
            d.rename(columns={"s1":"source1_entity_id","target_id":"candidate_entity_id"}, inplace=True)
            d["candidate_source"] = np.where(d.source.eq(2), "S2", "S3"); d["role"] = "test"
            cols = ["source1_entity_id","candidate_entity_id","candidate_source","country","role","s1_business_name","s1_business_address","candidate_business_name","candidate_business_address","lightgbm_score","lightgbm_rank_within_s1","name_retrieval_score","name_rank","address_retrieval_score","address_rank"]
            atomic_parquet(d[cols], op); meta={"country":country,"part":part,"rows":len(d)}; atomic_json(mp,meta); total += len(d); print("SHORTLIST_READY",json.dumps(meta),flush=True)
    atomic_json(EXP/"diagnostics/shortlist.json",{"rows":total,"seconds":time.time()-start_all,"candidate_pairs_semantics":"last candidate set supplied to frozen XLM-R matcher"})


def input_preflight():
    start=time.time(); report={"files":{},"models":{},"indexes":{},"ok":True}
    for name,(rows,digest) in EXPECTED_FILES.items():
        p=TEST/name; got=sha256(p); n=sum(1 for _ in p.open("rb"))-1; ok=(got==digest and n==rows)
        report["files"][name]={"rows":n,"sha256":got,"ok":ok}; report["ok"] &= ok
    report["models"]={"lightgbm_txt":str(MODEL_TXT),"lightgbm_sha256":sha256(MODEL_TXT),"lightgbm_joblib_sha256":sha256(MODEL_JOBLIB),
        "lightgbm_features":booster().num_feature(),"xlmr":str(XL_CHECKPOINT),"xlmr_sha256":sha256(XL_CHECKPOINT/"model.safetensors")}
    report["ok"] &= report["models"]["lightgbm_sha256"] == "226a35a7354a26f6edcab07893947115696bd1187d5e82aca70c0e0a3aa6f333"
    report["ok"] &= report["models"]["xlmr_sha256"] == "e7079b798b866b27116a0059c9c7cf1531817e42659ec0c9442654aee5d0c462"
    for country in COUNTRIES:
        for source in [2,3]:
            for channel in ["name","address"]:
                for p in [matrix_path(country,source,channel),idf_path(country,source,channel)]: report["indexes"][str(p)]={"bytes":p.stat().st_size,"sha256":sha256(p)}
    report["seconds"]=time.time()-start; atomic_json(EXP/"diagnostics/input_preflight.json",report)
    if not report["ok"]: raise RuntimeError("input preflight failed")
    print(json.dumps(report,indent=2))


def decode_outputs():
    start=time.time(); s=str(EXP/"cache/shortlist/*/*.parquet"); n=str(EXP/"cache/neural/*/*.parquet"); out=EXP/"output"; out.mkdir(parents=True,exist_ok=True)
    con=duckdb.connect();con.execute("PRAGMA threads=20");con.execute("PRAGMA memory_limit='100GB'");(EXP/"cache/ducktmp").mkdir(parents=True,exist_ok=True);con.execute(f"PRAGMA temp_directory='{EXP/'cache/ducktmp'}'")
    con.execute(f"""create temp table scored as select s.*,n.neural_probability,(.60*n.neural_probability+.40*s.lightgbm_score)::float final_score from read_parquet('{s}',union_by_name=true)s join read_parquet('{n}',union_by_name=true)n using(source1_entity_id,candidate_entity_id,candidate_source)""")
    counts=con.execute(f"select (select count(*)from read_parquet('{s}',union_by_name=true)),(select count(*)from scored),(select count(*)from read_parquet('{n}',union_by_name=true))").fetchone()
    if len(set(counts))!=1:raise RuntimeError(f"shortlist/neural merge cardinality failure: {counts}")
    con.execute("""create temp table owned as select *,row_number()over(partition by candidate_entity_id order by final_score desc,source1_entity_id)owner_rank,lead(final_score)over(partition by candidate_entity_id order by final_score desc,source1_entity_id)second_score from scored""")
    con.execute("""create temp table accepted as select * from owned where owner_rank=1 and final_score>=.51 and final_score-coalesce(second_score,0)>=.48""")
    # Preserve SQL NULL for empty groups: DuckDB serializes NULL as an empty TSV
    # field, whereas coalescing to an empty string serializes the literal `""`.
    con.execute(f"""copy(with q as(select *,row_number()over()rn from read_csv('{TEST/'test_source1.tsv'}',delim='\t',header=true,all_varchar=true,quote=''))select q.entity_id source1_entity_id,string_agg(a.candidate_entity_id,',' order by a.candidate_entity_id)matched_entity_ids from q left join accepted a on q.entity_id=a.source1_entity_id group by q.entity_id,q.rn order by q.rn)to '{out/'matching_results.tsv'}'(header,delimiter '\t')""")
    con.execute(f"""copy(with q as(select *,row_number()over()rn from read_csv('{TEST/'test_source1.tsv'}',delim='\t',header=true,all_varchar=true,quote=''))select q.entity_id source1_entity_id,string_agg(s.candidate_entity_id,',' order by s.candidate_entity_id)candidate_entity_ids from q left join scored s on q.entity_id=s.source1_entity_id group by q.entity_id,q.rn order by q.rn)to '{out/'candidate_pairs.tsv'}'(header,delimiter '\t')""")
    con.execute(f"copy(select * from accepted order by source1_entity_id,candidate_entity_id)to '{out/'accepted_pairs.parquet'}'(format parquet,compression zstd)")
    summary=con.execute("select (select count(*)from scored)shortlist_pairs,(select count(*)from accepted)predicted_edges,(select count(*)from owned where owner_rank=1)owner_winners").fetchdf().iloc[0].to_dict();con.close();summary["seconds"]=time.time()-start;atomic_json(EXP/"diagnostics/decode.json",summary);print(json.dumps(summary,indent=2,default=float))


def validate():
    start=time.time();out=EXP/"output";cmd=[sys.executable,str(ROOT/"student_resource/utils/validate_submission.py"),"--matching",str(out/"matching_results.tsv"),"--candidate",str(out/"candidate_pairs.tsv"),"--test-dir",str(TEST),"--check-ids"]
    p=subprocess.run(cmd,text=True,capture_output=True);text=p.stdout+p.stderr;(EXP/"logs/official_validator.log").write_text(text)
    if p.returncode or "PASS — no blocking issues found" not in text or "WARNING:" in text:raise RuntimeError("official validator failed")
    hashes={x:sha256(out/x) for x in ["matching_results.tsv","candidate_pairs.tsv"]};atomic_json(EXP/"diagnostics/validation.json",{"official_pass":True,"warnings":False,"hashes":hashes,"seconds":time.time()-start});print(text)


def main():
    for d in ["cache/blocker","cache/lightgbm","cache/shortlist","cache/neural","diagnostics","logs","output","runtime"]:(EXP/d).mkdir(parents=True,exist_ok=True)
    ap=argparse.ArgumentParser();ap.add_argument("stage",choices=["preflight","retrieve","features","shortlist","decode","validate"]);ap.add_argument("--worker",default="0");a=ap.parse_args()
    {"preflight":input_preflight,"retrieve":retrieve_full,"features":lambda:feature_worker(a.worker),"shortlist":build_shortlist,"decode":decode_outputs,"validate":validate}[a.stage]()


if __name__=="__main__":main()
