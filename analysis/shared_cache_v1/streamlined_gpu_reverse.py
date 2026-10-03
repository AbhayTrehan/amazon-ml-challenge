#!/usr/bin/env python3
"""Exact F20_R3 reverse-top3 parity probe and resumable GPU shard worker.

This is deliberately narrow: it reuses the frozen B2/F20 indexes and the
historical exact CUDA accumulation/tie semantics.  It does not implement a
general retrieval framework.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sparse_dot_topn import sp_matmul_topn

ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "analysis/shared_cache_v1"
REPORT = SHARED / "reports/streamlined_final"
DEV_RAW = SHARED / "candidates/c9b5de321742805728ecac097604dbd3e86a34159699266d7120c0dcfe2097ad"
VAL_TOP5 = SHARED / "candidates/96c3d374fa04d94a14e7c42da1c15c5e1b0357a7ec1ee6ef607b490eea58099e"
K = 3
THRESHOLD = 1e-8
AUX_SHARD = 50_000
COUNTRY_CODE = {"India": 0, "US": 1, "France": 2}

sys.path.insert(0, str(SHARED))
import run_b2_full as b2  # noqa: E402
from cache_utils import atomic_json, content_key  # noqa: E402

sys.path.insert(0, str(ROOT / "analysis/production/p001_e005b/src"))
import benchmark_gpu_retrieval as historical_gpu  # noqa: E402


CUDA = cp.RawModule(code=r'''
extern "C" __global__ void accum_seq(
 const int* ap, const int* ai, const float* ad,
 const int* bp, const int* bi, const float* bd,
 int nq, int nc, float* sums, int* first) {
  int q=blockIdx.x; if(q>=nq) return; long long off=(long long)q*nc;
  int fp=0;
  for(int a=ap[q]; a<ap[q+1]; ++a,++fp) {
    int f=ai[a]; float w=ad[a];
    for(int p=bp[f]+threadIdx.x;p<bp[f+1];p+=blockDim.x) {
      int c=bi[p]; long long z=off+c;
      if(first[z]==2147483647) first[z]=fp;
      sums[z] = __fadd_rn(sums[z], __fmul_rn(w, bd[p]));
    }
    __syncthreads();
  }
}
__device__ inline bool greater_val(float a,float b){return a>b;}
__device__ void push_heap3(int* ho,int* hi,float* hv) {
  int hole=2,parent=0;int o=ho[hole],i=hi[hole];float v=hv[hole];
  while(hole>0 && greater_val(hv[parent],v)) {
    ho[hole]=ho[parent];hi[hole]=hi[parent];hv[hole]=hv[parent];hole=parent;parent=(hole-1)/2;
  }
  ho[hole]=o;hi[hole]=i;hv[hole]=v;
}
__device__ void adjust_heap2(int* ho,int* hi,float* hv,int o,int i,float v) {
  int hole=0,top=0,second=0,len=2;
  while(second<(len-1)/2) {
    second=2*(second+1);if(greater_val(hv[second],hv[second-1]))second--;
    ho[hole]=ho[second];hi[hole]=hi[second];hv[hole]=hv[second];hole=second;
  }
  if((len&1)==0 && second==(len-2)/2) {
    second=2*(second+1);ho[hole]=ho[second-1];hi[hole]=hi[second-1];hv[hole]=hv[second-1];hole=second-1;
  }
  int parent=(hole-1)/2;
  while(hole>top && greater_val(hv[parent],v)) {
    ho[hole]=ho[parent];hi[hole]=hi[parent];hv[hole]=hv[parent];hole=parent;parent=(hole-1)/2;
  }
  ho[hole]=o;hi[hole]=i;hv[hole]=v;
}
extern "C" __global__ void select_heap3_parallel(
 const int* ap,const int* ai,const int* bp,const int* bi,
 int nq,int nc,const float* sums,const int* first,
 int* out_o,int* out_i,float* out_v) {
  int q=blockIdx.x;if(q>=nq)return;
  __shared__ int ho[3],hi[3],cand[256];__shared__ float hv[3];
  if(threadIdx.x<3){ho[threadIdx.x]=2147483647;hi[threadIdx.x]=-1;hv[threadIdx.x]=1e-8f;}
  __shared__ int nset;if(threadIdx.x==0)nset=0;__syncthreads();
  long long off=(long long)q*nc;int nf=ap[q+1]-ap[q];
  for(int fp=nf-1;fp>=0;--fp) {
    int f=ai[ap[q]+fp],begin=bp[f],end=bp[f+1];
    for(int base=end-1;base>=begin;base-=256) {
      int p=base-threadIdx.x,c=-1;float minv=hv[0];
      if(p>=begin){int z=bi[p];if(first[off+z]==fp && sums[off+z]>minv)c=z;}
      cand[threadIdx.x]=c;__syncthreads();
      if(threadIdx.x==0) {
        for(int t=0;t<256;++t){int z=cand[t];if(z<0)continue;float v=sums[off+z];if(!(v>hv[0]))continue;
          int o=ho[2],i=hi[2];float x=hv[2];ho[2]=ho[0];hi[2]=hi[0];hv[2]=hv[0];
          adjust_heap2(ho,hi,hv,o,i,x);ho[2]=nset;hi[2]=z;hv[2]=v;++nset;push_heap3(ho,hi,hv);
        }
      }
      __syncthreads();
    }
  }
  if(threadIdx.x<3){int p=q*3+threadIdx.x;out_o[p]=ho[threadIdx.x];out_i[p]=hi[threadIdx.x];out_v[p]=hv[threadIdx.x];}
}
''', options=("--std=c++14",))
ACCUM = CUDA.get_function("accum_seq")
SELECT = CUDA.get_function("select_heap3_parallel")


def log(event: str, **kw) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}] {event} " + json.dumps(kw, sort_keys=True, default=float), flush=True)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    frame.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)


class ExactGpuTop3:
    def __init__(self, owners: sparse.csr_matrix):
        self.nc = owners.shape[0]
        xc = owners.tocsc()
        self.bp = cp.asarray(xc.indptr.astype(np.int32, copy=False))
        self.bi = cp.asarray(xc.indices.astype(np.int32, copy=False))
        self.bd = cp.asarray(xc.data.astype(np.float32, copy=False))
        lib = ctypes.CDLL(str(Path(historical_gpu.__file__).with_name("tie_sort.so")))
        self.sort = lib.sdtn_value_sort
        self.sort.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float)]

    def run(self, queries: sparse.csr_matrix, batch: int) -> tuple[np.ndarray, np.ndarray, float, int]:
        all_i, all_v = [], []
        started = time.perf_counter(); peak = 0
        for start in range(0, queries.shape[0], batch):
            z = queries[start:start+batch].tocsr(); n = z.shape[0]
            ap=cp.asarray(z.indptr.astype(np.int32,copy=False));ai=cp.asarray(z.indices.astype(np.int32,copy=False));ad=cp.asarray(z.data.astype(np.float32,copy=False))
            sums=cp.zeros((n,self.nc),cp.float32);first=cp.full((n,self.nc),2147483647,cp.int32)
            ACCUM((n,),(128,),(ap,ai,ad,self.bp,self.bi,self.bd,n,self.nc,sums,first))
            go=cp.empty((n,K),cp.int32);gi=cp.empty((n,K),cp.int32);gv=cp.empty((n,K),cp.float32)
            SELECT((n,),(256,),(ap,ai,self.bp,self.bi,n,self.nc,sums,first,go,gi,gv));cp.cuda.Stream.null.synchronize()
            orders=go.get();inds=gi.get();vals=gv.get()
            self.sort(n,K,orders.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),inds.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),vals.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
            all_i.append(inds);all_v.append(vals);peak=max(peak,cp.get_default_memory_pool().used_bytes())
            del ap,ai,ad,sums,first,go,gi,gv;gc.collect()
        return np.vstack(all_i),np.vstack(all_v),time.perf_counter()-started,peak


def load_population(name: str) -> dict:
    manifest=json.loads((SHARED/f"manifests/b2_{name}_s1.manifest.json").read_text())
    frame=pd.read_parquet(SHARED/f"manifests/b2_{name}_s1.parquet")
    return {"manifest":manifest,"frame":frame,"path":SHARED/f"manifests/b2_{name}_s1.parquet"}


def indexes(name: str, info: dict) -> tuple[dict,dict]:
    forward={};reverse={};ready=True
    for country in b2.POP_COUNTRIES[name]:
        if not b2.query_map(name,country,info).exists():ready=False
        for source in [2,3]:
            ref=SHARED/"indexes/b2_forward_refs"/name/country/f"s{source}.manifest.json"
            if not ref.exists():ready=False
            else:forward[f"{country}_s{source}"]=json.loads(ref.read_text())["content_key"]
        for channel in ["name","address"]:
            identity={"kind":"b2_reverse_s1_index","population_key":info["manifest"]["content_key"],"country":country,"channel":channel,"vectorizer":"E001 frozen hashing TFIDF"}
            key=content_key(identity)
            if not (SHARED/"indexes"/key/"manifest.json").exists():ready=False
            reverse[f"{country}_{channel}"]=key
    if ready:return forward,reverse
    return b2.register_indexes(name,info)


def raw_base(name: str, info: dict, fkeys: dict, rkeys: dict) -> Path:
    identity={"kind":"f20_r3_raw_retrieval","population_key":info["manifest"]["content_key"],"forward_index_keys":fkeys,"reverse_index_keys":rkeys,
              "definition":"forward K20 plus reverse top3; literal string country equality",
              "forward":{"k":20,"posting_budget":20000,"coarse_m":800},
              "reverse":{"k":3,"method":"exact sparse_dot_topn","threshold":1e-8}}
    return SHARED/"candidates"/content_key(identity)


def expected_files(country: str, source: int, channel: str) -> list[Path]:
    root=DEV_RAW if country=="India" else VAL_TOP5
    sub=root/"reverse"/country if country=="India" else root/"shards"/country
    return sorted(sub.glob(f"s{source}_{channel}_*.parquet"))


def select_parity_apos(country: str, source: int, channel: str, aux: pd.DataFrame, idf: np.ndarray, owner: sparse.csr_matrix, n: int) -> tuple[np.ndarray,dict]:
    files=expected_files(country,source,channel)
    if not files: raise RuntimeError(f"no persisted CPU shards for {country} s{source} {channel}")
    available=np.unique(np.concatenate([pd.read_parquet(p,columns=["apos"]).apos.to_numpy(np.int32) for p in files]))
    # Deterministic broad candidate pool, then explicitly retain high posting-work rows.
    rng=np.random.default_rng(20260926 + source*17 + (channel=="address")*101 + (country=="US")*1009)
    pool=np.sort(rng.choice(available,size=min(max(n*4,n),len(available)),replace=False))
    z=b2.qmatrix(aux.iloc[pool],channel,idf)
    posting=np.diff(owner.tocsc().indptr)
    work=np.bincount(np.repeat(np.arange(len(pool)),np.diff(z.indptr)),weights=posting[z.indices],minlength=len(pool))
    high=pool[np.argsort(work,kind="stable")[-n//2:]]
    high_set=set(map(int,high));ordinary=np.array([x for x in pool if int(x) not in high_set],dtype=np.int32)[:n-len(high)]
    chosen=np.sort(np.concatenate([high,ordinary]).astype(np.int32))
    return chosen,{"available":len(available),"ordinary":len(ordinary),"high_fanout":len(high),"work_p50":float(np.median(work)),"work_max":int(work.max())}


def dense_top(csr: sparse.csr_matrix) -> tuple[np.ndarray,np.ndarray]:
    n=csr.shape[0];inds=np.full((n,K),-1,np.int32);vals=np.zeros((n,K),np.float32)
    for r in range(n):
        a,b=csr.indptr[r:r+2];m=b-a;inds[r,:m]=csr.indices[a:b];vals[r,:m]=csr.data[a:b]
    return inds,vals


def parity(batch: int, per_stream: int) -> None:
    REPORT.mkdir(parents=True,exist_ok=True);details=[];total=0;maxdiff=0.0;id_bad=rank_bad=persisted_bad=0;started=time.perf_counter()
    for country in ["India","US"]:
        pop="development_train" if country=="India" else "clean_validation";info=load_population(pop);fkeys,rkeys=indexes(pop,info)
        for channel in ["name","address"]:
            ridx=SHARED/"indexes"/rkeys[f"{country}_{channel}"];owner=sparse.load_npz(ridx/"matrix.npz").tocsr();idf=np.load(ridx/"idf.npy");corpus=pd.read_parquet(ridx/"corpus.parquet");owner_ids=corpus.qrow.to_numpy(np.int32)
            engine=ExactGpuTop3(owner)
            for source in [2,3]:
                idx=b2.forward_index(pop,country,source);aux=b2.normalized_aux(pop,country,source,idx)
                apos,selection=select_parity_apos(country,source,channel,aux,idf,owner,per_stream);q=b2.qmatrix(aux.iloc[apos],channel,idf)
                t=time.perf_counter();cpu=sp_matmul_topn(q,owner.T,top_n=K,threshold=THRESHOLD,sort=True,n_threads=20).tocsr();cpu_s=time.perf_counter()-t
                ci,cv=dense_top(cpu);gi,gv,gpu_s,peak=engine.run(q,batch)
                got=owner_ids[np.maximum(gi,0)];want=owner_ids[np.maximum(ci,0)];got[gi<0]=-1;want[ci<0]=-1
                ib=int(np.any(got!=want,axis=1).sum());rb=int((got!=want).sum());valid=(ci>=0)&(gi>=0);sd=float(np.max(np.abs(cv[valid]-gv[valid]))) if valid.any() else 0.0
                # Persisted CPU shard identity check for the chosen auxiliary queries.
                e=pd.concat([pd.read_parquet(p) for p in expected_files(country,source,channel)],ignore_index=True);e=e[e.apos.isin(apos)&e["rank"].le(3)].sort_values(["apos","rank"])
                exp={int(a):list(g.qrow.astype(int)) for a,g in e.groupby("apos",sort=False)}
                pb=sum(list(want[i][want[i]>=0])!=exp.get(int(a),[]) for i,a in enumerate(apos))
                row={"country":country,"source":source,"channel":channel,"queries":len(apos),"selection":selection,"cpu_seconds":cpu_s,"gpu_seconds":gpu_s,"gpu_qps":len(apos)/gpu_s,"peak_gpu_gb":peak/2**30,"id_mismatch_queries":ib,"rank_cell_mismatches":rb,"persisted_cpu_mismatch_queries":int(pb),"max_score_abs_difference":sd}
                details.append(row);log("PARITY_STREAM",**row);total+=len(apos);id_bad+=ib;rank_bad+=rb;persisted_bad+=pb;maxdiff=max(maxdiff,sd)
                del q,cpu,ci,cv,gi,gv,e;gc.collect()
            del engine,owner;cp.get_default_memory_pool().free_all_blocks();torch.cuda.empty_cache();gc.collect()
    persisted_dev_bad=sum(x["persisted_cpu_mismatch_queries"] for x in details if x["country"]=="India")
    persisted_val_top5_bad=sum(x["persisted_cpu_mismatch_queries"] for x in details if x["country"]=="US")
    result={"state":"PASS" if id_bad==0 and rank_bad==0 and persisted_dev_bad==0 else "FAIL","queries":total,"exact_candidate_ids":id_bad==0,"exact_ranks":rank_bad==0,"exact_persisted_development_pair_identities":persisted_dev_bad==0,"id_mismatch_queries":id_bad,"rank_cell_mismatches":rank_bad,"persisted_development_mismatch_queries":persisted_dev_bad,"validation_top5_slice_mismatch_queries":persisted_val_top5_bad,"validation_top5_note":"non-gating: top5-then-slice tie choices are not direct-top3 CPU semantics","max_score_abs_difference":maxdiff,"seconds":time.perf_counter()-started,"details":details}
    atomic_json(REPORT/"gpu_parity.json",result);print(json.dumps(result,indent=2),flush=True)
    if result["state"]!="PASS":raise SystemExit(2)


def worker(population: str, stream_specs: str, batch: int) -> None:
    info=load_population(population);fkeys,rkeys=indexes(population,info);base=raw_base(population,info,fkeys,rkeys);specs=[]
    for s in stream_specs.split(","):
        country,source,channel=s.split(":");specs.append((country,int(source),channel))
    remaining=[]
    for country,source,channel in specs:
        idx=b2.forward_index(population,country,source);aux=b2.normalized_aux(population,country,source,idx)
        for part,start in enumerate(range(0,len(aux),AUX_SHARD)):
            out=base/"reverse"/country/f"s{source}_{channel}_{part:05d}.parquet";meta=out.with_suffix(".meta.json")
            if not(out.exists()and meta.exists()):remaining.append((country,source,channel,part,start,min(len(aux),start+AUX_SHARD)-start))
    total=sum(x[-1] for x in remaining);processed=0;global_start=time.perf_counter();gpu_name=cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    log("GPU_WORKER_START",population=population,gpu=gpu_name,streams=specs,total_queries=total,batch=batch,base=str(base))
    for country,source,channel in specs:
        todo=[x for x in remaining if x[:3]==(country,source,channel)]
        if not todo:continue
        ridx=SHARED/"indexes"/rkeys[f"{country}_{channel}"];owner=sparse.load_npz(ridx/"matrix.npz").tocsr();idf=np.load(ridx/"idf.npy");s1rows=pd.read_parquet(ridx/"corpus.parquet").qrow.to_numpy(np.int32);engine=ExactGpuTop3(owner)
        idx=b2.forward_index(population,country,source);aux=b2.normalized_aux(population,country,source,idx);apos_all=aux.apos.to_numpy(np.int32)
        for _,_,_,part,start,nq in todo:
            out=base/"reverse"/country/f"s{source}_{channel}_{part:05d}.parquet";meta=out.with_suffix(".meta.json");block=aux.iloc[start:start+nq];q=b2.qmatrix(block,channel,idf)
            inds,vals,sec,peak=engine.run(q,batch);rr,kk=np.where(inds>=0)
            frame=pd.DataFrame({"country_code":np.int8(COUNTRY_CODE[country]),"qrow":s1rows[inds[rr,kk]],"source":np.int8(source),"apos":apos_all[start:start+nq][rr],"channel":np.int8(channel=="address"),"rank":(kk+1).astype(np.int16),"score":vals[rr,kk].astype(np.float32)})
            atomic_parquet(frame,out);processed+=nq;elapsed=time.perf_counter()-global_start;qps=processed/elapsed;eta=(total-processed)/qps if qps else None
            row={"direction":"reverse","backend":"exact_cuda_k3","gpu":gpu_name,"country":country,"source":source,"channel":channel,"part":part,"queries":nq,"rows":len(frame),"wall_seconds":sec,"qps":nq/sec,"batch":batch,"peak_gpu_gb":peak/2**30,"processed":processed,"total":total,"elapsed":elapsed,"eta_seconds":eta}
            atomic_json(meta,row);log("GPU_REVERSE_READY",population=population,**row);del q,inds,vals,frame;gc.collect()
        del engine,owner;cp.get_default_memory_pool().free_all_blocks();torch.cuda.empty_cache();gc.collect()
    atomic_json(REPORT/f"gpu_worker_{population}_{os.getpid()}.json",{"state":"COMPLETE","population":population,"gpu":gpu_name,"streams":specs,"queries":processed,"seconds":time.perf_counter()-global_start})


def main() -> None:
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest="cmd",required=True)
    p=sub.add_parser("parity");p.add_argument("--batch",type=int,default=1024);p.add_argument("--per-stream",type=int,default=2500)
    w=sub.add_parser("worker");w.add_argument("--population",choices=["development_train","test"],required=True);w.add_argument("--streams",required=True);w.add_argument("--batch",type=int,default=1024)
    a=ap.parse_args()
    if a.cmd=="parity":parity(a.batch,a.per_stream)
    else:worker(a.population,a.streams,a.batch)


if __name__=="__main__":main()
