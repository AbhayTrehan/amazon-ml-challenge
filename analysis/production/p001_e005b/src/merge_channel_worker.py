#!/usr/bin/env python3
"""Merge exact channel shards and materialize production candidate shards."""
from __future__ import annotations
import argparse,math,time
from pathlib import Path
import numpy as np,pandas as pd
import benchmark_gpu_retrieval as gpu
import production as p

def atomic(frame,path):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.parquet.tmp');frame.to_parquet(tmp,index=False,compression='zstd');tmp.replace(path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--country',required=True);ap.add_argument('--source',type=int,required=True);ap.add_argument('--channel-done',nargs=2,required=True);ap.add_argument('--done',required=True);a=ap.parse_args()
    q=pd.read_parquet(p.query_path(a.country));aux=pd.read_parquet(p.index_paths(a.country,a.source,'name')[1]);mats={}
    for ch,col in [('name','business_name'),('address','business_address')]:
        xa,tf=gpu.matrix(a.country,a.source,ch);xq=tf.transform(p.e001.vec_for(ch).transform(q[col].fillna(''))).astype(np.float32).tocsr();mats[ch]=(xa,xq)
    n=math.ceil(len(q)/p.SHARD_SIZE);part=0
    while part<n:
        op=p.EXP/f'cache/candidates/{a.country}/s{a.source}/part-{part:05d}.parquet'
        if op.exists():part+=1;continue
        inputs=[p.EXP/f'cache/gpu_channels/{a.country}/s{a.source}/{ch}/part-{part:05d}.parquet' for ch in ['name','address']]
        if not all(x.exists() for x in inputs):
            if all(Path(x).exists() for x in a.channel_done):raise RuntimeError(f'missing completed channel shard {part}')
            time.sleep(3);continue
        frames=[pd.read_parquet(x) for x in inputs];d=frames[0].merge(frames[1],on=['qpos','apos'],how='outer');start=part*p.SHARD_SIZE;qs=q.iloc[start:start+p.SHARD_SIZE].reset_index(drop=True)
        d['s1']=qs.entity_id.to_numpy()[d.qpos];d['target_id']=aux.entity_id.to_numpy()[d.apos]
        for ch in ['name','address']:
            xa,xq=mats[ch];d[f'{ch}_char_tfidf']=p.e001.row_cosines(xq[start:start+len(qs)],xa,d.qpos.to_numpy(),d.apos.to_numpy())
        atomic(d,op);print('CANDIDATE_READY',a.country,a.source,part,len(d),flush=True);part+=1
    Path(a.done).write_text(f'done {time.time()}\n')

if __name__=='__main__':main()
