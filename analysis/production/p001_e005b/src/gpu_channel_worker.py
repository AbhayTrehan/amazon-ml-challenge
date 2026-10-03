#!/usr/bin/env python3
"""Exact single-channel GPU retriever for dynamic two-GPU partition splitting."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np,pandas as pd
import benchmark_gpu_retrieval as gpu
import production as p

def atomic(frame,path):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.parquet.tmp');frame.to_parquet(tmp,index=False,compression='zstd');tmp.replace(path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--country',required=True);ap.add_argument('--source',type=int,required=True);ap.add_argument('--channel',choices=['name','address'],required=True);ap.add_argument('--batch',type=int,default=1024);ap.add_argument('--done',required=True);a=ap.parse_args()
    q=pd.read_parquet(p.query_path(a.country));col='business_name' if a.channel=='name' else 'business_address';xa,tf=gpu.matrix(a.country,a.source,a.channel)
    xq=tf.transform(p.e001.vec_for(a.channel).transform(q[col].fillna(''))).astype(np.float32).tocsr();t=time.time();inds,vals,times,peak=gpu.exact_gpu_sdtn(xq,xa,a.batch);sec=time.time()-t
    out=p.EXP/f'cache/gpu_channels/{a.country}/s{a.source}/{a.channel}'
    for part,start in enumerate(range(0,len(q),p.SHARD_SIZE)):
        op=out/f'part-{part:05d}.parquet'
        if op.exists():continue
        stop=min(len(q),start+p.SHARD_SIZE);n=stop-start
        d=pd.DataFrame({'qpos':np.repeat(np.arange(n,dtype=np.int32),20),'apos':inds[start:stop].ravel().astype(np.int32),f'{a.channel}_rank':np.tile(np.arange(1,21,dtype=np.int16),n),f'{a.channel}_retrieval_score':vals[start:stop].ravel().astype(np.float32)})
        atomic(d,op)
    p.atomic_json(p.EXP/f'diagnostics/gpu_retrieval_{a.country}_s{a.source}_{a.channel}.json',{'country':a.country,'source':a.source,'channel':a.channel,'rows':len(q),'batch':a.batch,'seconds':sec,'qps':len(q)/sec,'peak_gpu_gb':peak/2**30})
    Path(a.done).write_text(f'done {time.time()}\n');print('CHANNEL_DONE',a.country,a.source,a.channel,sec,len(q)/sec,flush=True)

if __name__=='__main__':main()
