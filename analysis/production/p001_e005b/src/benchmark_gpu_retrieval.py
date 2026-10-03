#!/usr/bin/env python3
"""Exact GPU retrieval benchmark/parity probe; never writes production shards."""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer

ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "analysis/production/p001_e005b"
sys.path.insert(0, str(ROOT / "analysis/experiments_v1/e001_classical_baseline/scripts"))
import prepare_baseline as e001  # noqa: E402

ACCUM = cp.RawKernel(r'''
extern "C" __global__ void accum(
 const int* ci, const int* ri, const float* rd,
 const int* qr, const int* qi, const float* qd,
 int nt, int nc, float* out) {
  int t=blockIdx.x; if(t>=nt) return;
  int q=qr[t], f=qi[t]; float w=qd[t];
  for(int p=ci[f]+threadIdx.x; p<ci[f+1]; p+=blockDim.x)
    atomicAdd(&out[(long long)q*nc+ri[p]], w*rd[p]);
}''', "accum")

EXACT = cp.RawModule(code=r'''
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
__device__ void push_heap20(int* ho,int* hi,float* hv) {
  int hole=19,parent=(hole-1)/2;int o=ho[hole],i=hi[hole];float v=hv[hole];
  while(hole>0 && greater_val(hv[parent],v)) {
    ho[hole]=ho[parent];hi[hole]=hi[parent];hv[hole]=hv[parent];hole=parent;parent=(hole-1)/2;
  }
  ho[hole]=o;hi[hole]=i;hv[hole]=v;
}
__device__ void adjust_heap19(int* ho,int* hi,float* hv,int o,int i,float v) {
  int hole=0,top=0,second=0,len=19;
  while(second<(len-1)/2) {
    second=2*(second+1);
    if(greater_val(hv[second],hv[second-1])) second--;
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
extern "C" __global__ void select_heap20(
 const int* ap,const int* ai,const int* bp,const int* bi,
 int nq,int nc,const float* sums,const int* first,
 int* out_o,int* out_i,float* out_v) {
  int q=blockIdx.x;if(q>=nq||threadIdx.x) return;int ho[20],hi[20];float hv[20];
  for(int z=0;z<20;++z){ho[z]=2147483647;hi[z]=-1;hv[z]=1e-8f;}
  int nset=0;long long off=(long long)q*nc;int nf=ap[q+1]-ap[q];
  for(int fp=nf-1;fp>=0;--fp) {
    int f=ai[ap[q]+fp];
    for(int p=bp[f+1]-1;p>=bp[f];--p) {
      int c=bi[p];if(first[off+c]!=fp) continue;float v=sums[off+c];
      if(v>hv[0]) {
        int o=ho[19],i=hi[19];float x=hv[19];ho[19]=ho[0];hi[19]=hi[0];hv[19]=hv[0];
        adjust_heap19(ho,hi,hv,o,i,x);
        ho[19]=nset;hi[19]=c;hv[19]=v;++nset;push_heap20(ho,hi,hv);
      }
    }
  }
  for(int z=0;z<20;++z){int p=q*20+z;out_o[p]=ho[z];out_i[p]=hi[z];out_v[p]=hv[z];}
}
extern "C" __global__ void select_heap20_parallel(
 const int* ap,const int* ai,const int* bp,const int* bi,
 int nq,int nc,const float* sums,const int* first,
 int* out_o,int* out_i,float* out_v) {
  int q=blockIdx.x;if(q>=nq) return;
  __shared__ int ho[20],hi[20],cand[256];__shared__ float hv[20];
  if(threadIdx.x<20){ho[threadIdx.x]=2147483647;hi[threadIdx.x]=-1;hv[threadIdx.x]=1e-8f;}
  __shared__ int nset; if(threadIdx.x==0)nset=0;__syncthreads();
  long long off=(long long)q*nc;int nf=ap[q+1]-ap[q];
  for(int fp=nf-1;fp>=0;--fp) {
    int f=ai[ap[q]+fp],begin=bp[f],end=bp[f+1];
    for(int base=end-1;base>=begin;base-=256) {
      int p=base-threadIdx.x,c=-1;float minv=hv[0];
      if(p>=begin){int z=bi[p];if(first[off+z]==fp && sums[off+z]>minv)c=z;}cand[threadIdx.x]=c;__syncthreads();
      if(threadIdx.x==0) {
        for(int t=0;t<256;++t){int z=cand[t];if(z<0)continue;float v=sums[off+z];if(!(v>hv[0]))continue;
          int o=ho[19],i=hi[19];float x=hv[19];ho[19]=ho[0];hi[19]=hi[0];hv[19]=hv[0];
          adjust_heap19(ho,hi,hv,o,i,x);ho[19]=nset;hi[19]=z;hv[19]=v;++nset;push_heap20(ho,hi,hv);
        }
      }
      __syncthreads();
    }
  }
  if(threadIdx.x<20){int p=q*20+threadIdx.x;out_o[p]=ho[threadIdx.x];out_i[p]=hi[threadIdx.x];out_v[p]=hv[threadIdx.x];}
}
''', options=("--std=c++14",))
ACCUM_SEQ = EXACT.get_function("accum_seq")
SELECT_HEAP20 = EXACT.get_function("select_heap20")
SELECT_HEAP20_PARALLEL = EXACT.get_function("select_heap20_parallel")


def exact_gpu_sdtn(xq, xa, batch):
    """Reproduce sparse_dot_topn float accumulation, insertion heap, and value sort."""
    xc=xa.tocsc();bp=cp.asarray(xc.indptr.astype(np.int32));bi=cp.asarray(xc.indices.astype(np.int32));bd=cp.asarray(xc.data)
    lib=ctypes.CDLL(str(Path(__file__).with_name("tie_sort.so")))
    fn=lib.sdtn_value_sort;fn.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_float)]
    oi=[];ov=[];timings=[];peak=0
    for start in range(0,xq.shape[0],batch):
        z=xq[start:start+batch].tocsr();n=z.shape[0];ap=cp.asarray(z.indptr.astype(np.int32));ai=cp.asarray(z.indices.astype(np.int32));ad=cp.asarray(z.data)
        cp.cuda.Stream.null.synchronize();t=time.perf_counter();sums=cp.zeros((n,xa.shape[0]),cp.float32);first=cp.full((n,xa.shape[0]),2147483647,cp.int32)
        ACCUM_SEQ((n,),(128,),(ap,ai,ad,bp,bi,bd,n,xa.shape[0],sums,first));cp.cuda.Stream.null.synchronize();acc=time.perf_counter()-t
        go=cp.empty((n,20),cp.int32);gi=cp.empty((n,20),cp.int32);gv=cp.empty((n,20),cp.float32);t=time.perf_counter()
        SELECT_HEAP20_PARALLEL((n,),(256,),(ap,ai,bp,bi,n,xa.shape[0],sums,first,go,gi,gv));cp.cuda.Stream.null.synchronize();sel=time.perf_counter()-t
        orders=go.get();inds=gi.get();vals=gv.get();t=time.perf_counter();fn(n,20,orders.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),inds.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),vals.ctypes.data_as(ctypes.POINTER(ctypes.c_float)));sort_s=time.perf_counter()-t
        oi.append(inds);ov.append(vals);timings.append((n,acc,sel,sort_s));peak=max(peak,cp.get_default_memory_pool().used_bytes())
        del sums,first,go,gi,gv,ap,ai,ad
    return np.vstack(oi),np.vstack(ov),timings,peak


def hybrid_gpu_sdtn(xq, xa, batch, overfetch=64):
    """Exact ordered accumulation + fast top-k + sparse-dot-topn heap replay."""
    xc=xa.tocsc();bp=cp.asarray(xc.indptr.astype(np.int32));bi=cp.asarray(xc.indices.astype(np.int32));bd=cp.asarray(xc.data)
    lib=ctypes.CDLL(str(Path(__file__).with_name("tie_sort.so")));fn=lib.sdtn_replay_selected
    fn.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_float)]
    oi=[];ov=[];timings=[];peak=0;overflow=0
    for start in range(0,xq.shape[0],batch):
        z=xq[start:start+batch].tocsr();n=z.shape[0];ap=cp.asarray(z.indptr.astype(np.int32));ai=cp.asarray(z.indices.astype(np.int32));ad=cp.asarray(z.data)
        cp.cuda.Stream.null.synchronize();torch.cuda.synchronize();wall=time.perf_counter();t=time.perf_counter()
        sums=cp.zeros((n,xa.shape[0]),cp.float32);first=cp.full((n,xa.shape[0]),2147483647,cp.int32)
        ACCUM_SEQ((n,),(128,),(ap,ai,ad,bp,bi,bd,n,xa.shape[0],sums,first));cp.cuda.Stream.null.synchronize();acc=time.perf_counter()-t
        ts=torch.utils.dlpack.from_dlpack(sums);tf=torch.utils.dlpack.from_dlpack(first);t=time.perf_counter()
        vals,inds=torch.topk(ts,overfetch,dim=1,largest=True,sorted=True);keys=torch.gather(tf,1,inds);torch.cuda.synchronize();top=time.perf_counter()-t
        iv=inds.cpu().numpy().astype(np.int32,copy=True);vv=vals.cpu().numpy().astype(np.float32,copy=True);kv=keys.cpu().numpy().astype(np.int32,copy=True)
        overflow += int(np.isclose(vv[:,19],vv[:,-1],rtol=0,atol=0).sum())
        t=time.perf_counter();fn(n,overfetch,kv.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),iv.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),vv.ctypes.data_as(ctypes.POINTER(ctypes.c_float)));replay=time.perf_counter()-t
        oi.append(iv[:,:20].copy());ov.append(vv[:,:20].copy());timings.append((n,acc,top,replay,time.perf_counter()-wall));peak=max(peak,cp.get_default_memory_pool().used_bytes(),torch.cuda.max_memory_allocated())
        del sums,first,ts,tf,vals,inds,keys,ap,ai,ad
    return np.vstack(oi),np.vstack(ov),timings,peak,overflow


def matrix(country: str, source: int, channel: str):
    base = EXP / f"cache/indexes/{country}/s{source}"
    xa = sparse.load_npz(base / f"{channel}.npz").tocsr()
    tf = TfidfTransformer(); tf.idf_ = np.load(base / f"{channel}_idf.npy")
    return xa, tf


def gpu_top(xq, xa, batch: int, k: int):
    xc = xa.tocsc()
    gi=cp.asarray(xc.indptr.astype(np.int32)); gr=cp.asarray(xc.indices.astype(np.int32)); gd=cp.asarray(xc.data)
    all_i=[]; all_v=[]; timings=[]; peak=0
    for start in range(0, xq.shape[0], batch):
        z=xq[start:start+batch].tocsr(); n=z.shape[0]
        qrows=np.repeat(np.arange(n,dtype=np.int32),np.diff(z.indptr)).astype(np.int32)
        gqr=cp.asarray(qrows);gqi=cp.asarray(z.indices.astype(np.int32));gqd=cp.asarray(z.data)
        cp.cuda.Stream.null.synchronize(); torch.cuda.synchronize(); wall=time.perf_counter()
        a0=cp.cuda.Event();a1=cp.cuda.Event();a0.record()
        out=cp.zeros((n,xa.shape[0]),dtype=cp.float32)
        ACCUM((len(qrows),),(128,),(gi,gr,gd,gqr,gqi,gqd,len(qrows),xa.shape[0],out))
        a1.record();a1.synchronize();acc=cp.cuda.get_elapsed_time(a0,a1)/1000
        t=time.perf_counter(); tout=torch.utils.dlpack.from_dlpack(out);torch.cuda.synchronize();interop=time.perf_counter()-t
        s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True);s.record()
        vals,inds=torch.topk(tout,k,dim=1,largest=True,sorted=True);e.record();e.synchronize();top=s.elapsed_time(e)/1000
        all_i.append(inds.cpu().numpy());all_v.append(vals.cpu().numpy())
        total=time.perf_counter()-wall;peak=max(peak,cp.get_default_memory_pool().used_bytes(),torch.cuda.max_memory_allocated())
        timings.append((n,acc,interop,top,total));del out,tout,vals,inds,gqr,gqi,gqd
    return np.vstack(all_i),np.vstack(all_v),timings,peak


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--country",default="France");ap.add_argument("--source",type=int,default=2)
    ap.add_argument("--queries",type=int,default=2000);ap.add_argument("--batch",type=int,default=512);ap.add_argument("--k",type=int,default=16)
    args=ap.parse_args(); q=pd.read_parquet(EXP/f"cache/indexes/{args.country}/queries.parquet").iloc[:args.queries].reset_index(drop=True)
    old=pd.read_parquet(EXP/f"cache/features/{args.country}/s{args.source}/part-00000.parquet")
    old=old[old.s1.isin(q.entity_id)].copy();aux=pd.read_parquet(EXP/f"cache/indexes/{args.country}/s{args.source}/aux.parquet")
    report={"country":args.country,"source":args.source,"queries":len(q),"batch":args.batch,"k":args.k,"channels":{}}
    for channel,col in [("name","business_name"),("address","business_address")]:
        xa,tf=matrix(args.country,args.source,channel);xq=tf.transform(e001.vec_for(channel).transform(q[col].fillna(""))).astype(np.float32).tocsr()
        inds,_,timings,peak=gpu_top(xq,xa,args.batch,args.k)
        exact=np.empty_like(inds,dtype=np.float32)
        for r in range(len(q)): exact[r]=np.asarray(xq[r].multiply(xa[inds[r]]).sum(axis=1)).ravel()
        order=np.argsort(-exact,axis=1,kind="stable");inds=np.take_along_axis(inds,order,axis=1);exact=np.take_along_axis(exact,order,axis=1)
        boundary=np.isclose(exact[:,9],exact[:,10],rtol=0,atol=1e-7)
        expected=old[old[f"{channel}_rank"].le(10)].sort_values(["s1",f"{channel}_rank"])
        exp={s:list(g.target_id) for s,g in expected.groupby("s1",sort=False)}; ids=aux.entity_id.to_numpy()
        set_bad=score_bad=rank_bad=0;checked=0
        for r,s1 in enumerate(q.entity_id):
            want=exp.get(s1,[]); got=list(ids[inds[r,:10]])
            if boundary[r]: continue
            checked+=1;set_bad+=set(got)!=set(want);rank_bad+=got!=want
            if set(got)==set(want):
                smap=dict(zip(ids[inds[r]],exact[r]));eg=expected[expected.s1.eq(s1)]
                score_bad+=bool(np.max(np.abs(np.array([smap[x] for x in eg.target_id])-eg[f"{channel}_retrieval_score"].to_numpy()))>2e-6)
        sec=sum(x[4] for x in timings);report["channels"][channel]={"seconds":sec,"qps":len(q)/sec,"accum_seconds":sum(x[1] for x in timings),"interop_seconds":sum(x[2] for x in timings),"topk_seconds":sum(x[3] for x in timings),"peak_gpu_gb":peak/2**30,"boundary_ties":int(boundary.sum()),"boundary_tie_rate":float(boundary.mean()),"unambiguous_checked":checked,"candidate_set_mismatches":set_bad,"score_mismatches":score_bad,"rank_mismatches_before_tie_fallback":rank_bad}
        del xa,xq,inds,exact;cp.get_default_memory_pool().free_all_blocks();torch.cuda.empty_cache()
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
