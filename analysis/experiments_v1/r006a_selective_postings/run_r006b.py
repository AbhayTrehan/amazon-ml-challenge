#!/usr/bin/env python3
"""R006B final rescue: fixed-budget M scaling and thread scaling only."""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_r006a as base  # noqa: E402

OUT = HERE / "artifacts/r006b_rescue"
BUDGET = 20000
TEST_S1 = 1_732_544
PROC = psutil.Process()


def cpu_seconds():
    z = PROC.cpu_times()
    return z.user + z.system


def load_functions():
    lib = ctypes.CDLL(str(base.compile_kernel()))
    main = lib.selective_postings
    ip = ctypes.POINTER(ctypes.c_int32); fp = ctypes.POINTER(ctypes.c_float)
    i64p = ctypes.POINTER(ctypes.c_int64); dp = ctypes.POINTER(ctypes.c_double)
    main.argtypes = [ip, ip, fp, ctypes.c_int, ip, ip, fp, ip, ip, fp, ctypes.c_int,
                     ctypes.c_int, ctypes.c_int, ctypes.c_int, ip, fp, i64p, i64p, ip, ip, dp]
    main.restype = ctypes.c_int
    diag = lib.selective_truth_diagnostic
    u8p = ctypes.POINTER(ctypes.c_uint8)
    diag.argtypes = [ip, ip, fp, ctypes.c_int, ip, ip, fp, ip, ip,
                     ctypes.c_int, ctypes.c_int, ctypes.c_int, u8p, u8p]
    diag.restype = ctypes.c_int
    return main, diag


def truth_csr(qids, aux_ids, truth, source):
    apos = pd.Series(np.arange(len(aux_ids), dtype=np.int32), index=pd.Index(aux_ids))
    by_q = {s: g.target_id.tolist() for s, g in truth[truth.source.eq(source)].groupby("s1", sort=False)}
    docs, keys, indptr = [], [], [0]
    for s1 in qids:
        targets = by_q.get(s1, [])
        for target in targets:
            if target not in apos.index: raise RuntimeError(f"truth target absent from fold-0 index: {target}")
            docs.append(int(apos[target])); keys.append((s1, target, source))
        indptr.append(len(docs))
    return np.asarray(indptr, np.int32), np.asarray(docs, np.int32), keys


def selective(main, xq, xa, m, threads):
    q=xq.tocsr(); a=xa.tocsr(); c=xa.tocsc()
    arrays=[base.i32(q.indptr),base.i32(q.indices),base.f32(q.data),base.i32(c.indptr),base.i32(c.indices),base.f32(c.data),
            base.i32(a.indptr),base.i32(a.indices),base.f32(a.data)]
    nq=q.shape[0]; oi=np.full((nq,10),-1,np.int32);ov=np.zeros((nq,10),np.float32)
    visits=np.zeros(nq,np.int64);full=np.zeros(nq,np.int64);counts=np.zeros(nq,np.int32);fallback=np.zeros(nq,np.int32);timing=np.zeros(5,np.float64)
    ip=ctypes.POINTER(ctypes.c_int32);fp=ctypes.POINTER(ctypes.c_float)
    t=time.perf_counter(); c0=cpu_seconds()
    rc=main(arrays[0].ctypes.data_as(ip),arrays[1].ctypes.data_as(ip),arrays[2].ctypes.data_as(fp),nq,
            arrays[3].ctypes.data_as(ip),arrays[4].ctypes.data_as(ip),arrays[5].ctypes.data_as(fp),
            arrays[6].ctypes.data_as(ip),arrays[7].ctypes.data_as(ip),arrays[8].ctypes.data_as(fp),a.shape[0],
            BUDGET,m,threads,oi.ctypes.data_as(ip),ov.ctypes.data_as(fp),
            visits.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),full.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            counts.ctypes.data_as(ip),fallback.ctypes.data_as(ip),timing.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    wall=time.perf_counter()-t; cpu=cpu_seconds()-c0
    if rc: raise RuntimeError(f"selective kernel failed: {rc}")
    return oi,ov,wall,cpu,visits,full,timing,arrays


def diagnose(diag, arrays, nq, tindptr, tdocs, m, threads):
    exposed=np.zeros(len(tdocs),np.uint8);coarse=np.zeros(len(tdocs),np.uint8)
    ip=ctypes.POINTER(ctypes.c_int32);fp=ctypes.POINTER(ctypes.c_float);u8p=ctypes.POINTER(ctypes.c_uint8)
    rc=diag(arrays[0].ctypes.data_as(ip),arrays[1].ctypes.data_as(ip),arrays[2].ctypes.data_as(fp),nq,
            arrays[3].ctypes.data_as(ip),arrays[4].ctypes.data_as(ip),arrays[5].ctypes.data_as(fp),
            tindptr.ctypes.data_as(ip),tdocs.ctypes.data_as(ip),BUDGET,m,threads,
            exposed.ctypes.data_as(u8p),coarse.ctypes.data_as(u8p))
    if rc: raise RuntimeError(f"diagnostic kernel failed: {rc}")
    return exposed,coarse


def run_config(main, diag, selected, truth, exact_keys, m, threads, diagnostics):
    frames=[]; exposed_keys=set();coarse_keys=set();transform_wall=kernel_wall=union_wall=0.0
    measured_cpu=0.0;timing=np.zeros(5);postings=full_postings=0;score_diff=0.0;checks=0
    for country,sc0 in selected.groupby("country",sort=False):
        sc=sc0.sort_values("source1_entity_id",kind="mergesort");q=base.query_frame(sc)
        for source in [2,3]:
            aux,indexes=base.build_or_load(country,source);aux_ids=aux.entity_id.to_numpy()
            tindptr,tdocs,tkeys=truth_csr(q.entity_id.to_numpy(),aux_ids,truth,source)
            for channel in ["name","address"]:
                xa,idf=indexes[channel]
                c0=cpu_seconds();t=time.perf_counter();xq=base.qmatrix(q,channel,idf);transform_wall+=time.perf_counter()-t;measured_cpu+=cpu_seconds()-c0
                oi,ov,wall,cpu,visits,full,stage,arrays=selective(main,xq,xa,m,threads)
                kernel_wall+=wall;measured_cpu+=cpu;timing+=stage;postings+=int(visits.sum());full_postings+=int(full.sum())
                frames.append(base.rows_from_dense(q.entity_id.to_numpy(),aux_ids,oi,ov,source,country,channel))
                rr,kk=np.where(oi>=0)
                if len(rr):
                    check=np.asarray(xq[rr].multiply(xa[oi[rr,kk]]).sum(axis=1)).ravel().astype(np.float32)
                    score_diff=max(score_diff,float(np.max(np.abs(check-ov[rr,kk]))));checks+=len(check)
                if diagnostics:
                    ex,co=diagnose(diag,arrays,len(q),tindptr,tdocs,m,threads)
                    exposed_keys.update(k for k,v in zip(tkeys,ex) if v);coarse_keys.update(k for k,v in zip(tkeys,co) if v)
                del xa,xq,arrays;gc.collect()
    c0=cpu_seconds();t=time.perf_counter();pairs=base.union_streams(frames);union_wall=time.perf_counter()-t;measured_cpu+=cpu_seconds()-c0
    wall=transform_wall+kernel_wall+union_wall;quality=base.quality(pairs,selected,truth,exact_keys)
    final_keys=set(map(tuple,pairs[["s1","target_id","source"]].itertuples(index=False,name=None)))
    truth_keys=set(map(tuple,truth[["s1","target_id","source"]].itertuples(index=False,name=None)))
    stage_values=list(timing[:4])+[union_wall*threads];stage_names=["extraction_sort","postings_accumulation","coarse_top_m","exact_rerank","union_output"]
    stage_total=sum(stage_values)
    quality.update({"m":m,"threads":threads,"wall_seconds":wall,"transform_seconds":transform_wall,"kernel_seconds":kernel_wall,
                    "union_seconds":union_wall,"s1_per_second":len(selected)/wall,"stream_queries_per_second":4*len(selected)/wall,
                    "cpu_seconds":measured_cpu,"cpu_utilization_total_percent":100*measured_cpu/wall,
                    "cpu_utilization_of_configured_threads_percent":100*measured_cpu/wall/threads,
                    "peak_rss_gb":max(PROC.memory_info().rss,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)/2**30,
                    "projected_full_test_minutes":wall/len(selected)*TEST_S1/60,
                    "postings_visited":postings,"full_postings":full_postings,"postings_avoided_fraction":1-postings/full_postings,
                    "rerank_checks":checks,"rerank_max_abs_difference":score_diff,
                    "timing_breakdown_fraction":dict(zip(stage_names,[x/stage_total for x in stage_values])),
                    "accumulation_estimated_wall_seconds":timing[1]/threads})
    if diagnostics:
        quality.update({"truth_exposed_edges":len(exposed_keys&truth_keys),"truth_exposed_recall":len(exposed_keys&truth_keys)/len(truth_keys),
                        "truth_coarse_edges":len(coarse_keys&truth_keys),"truth_coarse_recall":len(coarse_keys&truth_keys)/len(truth_keys),
                        "truth_final_edges":len(final_keys&truth_keys),"truth_final_recall":len(final_keys&truth_keys)/len(truth_keys)})
    return quality


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--n",type=int,default=10000);ap.add_argument("--ms",type=int,nargs="+",default=[200,400,800]);ap.add_argument("--thread-counts",type=int,nargs="+",default=[6,12,20]);args=ap.parse_args()
    if args.ms != [200,400,800] or args.thread_counts != [6,12,20]: raise RuntimeError("R006B frozen grid violated")
    base.metric_self_test();main_fn,diag_fn=load_functions();selected=base.cohort(args.n);truth=base.truth_for(selected);exact_keys=base.cached_exact_keys(selected)
    exact_oracle=0.9874557321200704
    m_rows=[]
    for m in args.ms:
        row=run_config(main_fn,diag_fn,selected,truth,exact_keys,m,6,True);row["oracle_delta_vs_exact"]=row["oracle_grouped_macro_f05"]-exact_oracle;m_rows.append(row)
        print(f"M_DONE m={m} wall={row['wall_seconds']:.3f} oracle={row['oracle_grouped_macro_f05']:.9f}",flush=True)
    best=max(m_rows,key=lambda x:(x["oracle_grouped_macro_f05"],-x["wall_seconds"]))
    thread_rows=[dict(best)]
    for threads in [12,20]:
        row=run_config(main_fn,diag_fn,selected,truth,exact_keys,best["m"],threads,False);row["oracle_delta_vs_exact"]=row["oracle_grouped_macro_f05"]-exact_oracle;thread_rows.append(row)
        print(f"THREAD_DONE threads={threads} wall={row['wall_seconds']:.3f}",flush=True)
    base6=thread_rows[0]["wall_seconds"]
    for row in thread_rows: row["speedup_vs_6_threads"]=base6/row["wall_seconds"]
    correctness={"cohort_n":len(selected),"fold0_only":bool((selected.fold==0).all()),"fixed_budget":BUDGET,
                 "rerank_max_abs_difference":max(x["rerank_max_abs_difference"] for x in m_rows+thread_rows),
                 "rerank_scores_pass":max(x["rerank_max_abs_difference"] for x in m_rows+thread_rows)<=2e-6,
                 "thread_quality_invariant":len({round(x["oracle_grouped_macro_f05"],15) for x in thread_rows})==1}
    report={"m_scaling":m_rows,"best_m":best["m"],"thread_scaling":thread_rows,"correctness":correctness}
    OUT.mkdir(parents=True,exist_ok=True);base.atomic_json(OUT/"results.json",report)
    pd.DataFrame([{k:v for k,v in x.items() if not isinstance(v,dict)} for x in m_rows]).to_csv(OUT/"m_scaling.csv",index=False)
    pd.DataFrame([{k:v for k,v in x.items() if not isinstance(v,dict)} for x in thread_rows]).to_csv(OUT/"thread_scaling.csv",index=False)
    print(json.dumps(report,indent=2,default=float))


if __name__=="__main__":main()
