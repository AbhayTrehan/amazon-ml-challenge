#!/usr/bin/env python3
"""Resumable CPU feature/LightGBM consumer for GPU candidate shards."""
from __future__ import annotations
import argparse,gc,json,math,os,time
from pathlib import Path
import pandas as pd
import production as p

p.THREADS=5

def atomic_parquet(frame,path):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.parquet.tmp');frame.to_parquet(tmp,index=False,compression='zstd');tmp.replace(path)

def tasks():
    out=[]
    for country in ['France','India','US']:
        n=math.ceil(len(pd.read_parquet(p.query_path(country),columns=['entity_id']))/p.SHARD_SIZE)
        for source in [2,3]:
            for part in range(n):out.append((country,source,part))
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker',required=True);ap.add_argument('--done-markers',nargs='+',required=True);a=ap.parse_args();model=p.booster();fs=p.features();current=None;cache=None
    while True:
        progressed=False
        for country,source,part in tasks():
            fp=p.EXP/f'cache/features/{country}/s{source}/part-{part:05d}.parquet';pp=p.EXP/f'cache/lightgbm_predictions/{country}/s{source}/part-{part:05d}.parquet'
            if fp.exists() and pp.exists():continue
            cp=p.EXP/f'cache/candidates/{country}/s{source}/part-{part:05d}.parquet'
            if not cp.exists():continue
            lock=cp.with_suffix('.feature.lock')
            try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
            except FileExistsError:continue
            try:
                if current!=(country,source):
                    del cache;gc.collect();q=pd.read_parquet(p.query_path(country));aux=pd.read_parquet(p.index_paths(country,source,'name')[1]);nf=aux.nname.value_counts(dropna=False);af=aux.naddr.value_counts(dropna=False);cache=(q,aux,nf,af);current=(country,source)
                q,aux,nf,af=cache;qs=q.iloc[part*p.SHARD_SIZE:(part+1)*p.SHARD_SIZE].reset_index(drop=True);cand=pd.read_parquet(cp);t=time.time();feat,score_s=p.enrich_and_score(cand,qs,aux,source,country,model,fs,nf,af)
                atomic_parquet(feat,fp);pred=feat[['s1','target_id','source','country','name_rank','name_retrieval_score','address_rank','address_retrieval_score','retrieved_by_name','retrieved_by_address','lightgbm_score']];atomic_parquet(pred,pp)
                print('FEATURE_READY',a.worker,country,source,part,len(feat),time.time()-t,flush=True);progressed=True
            finally:
                lock.unlink(missing_ok=True)
        if all(Path(x).exists() for x in a.done_markers):
            missing=sum(not ((p.EXP/f'cache/features/{c}/s{s}/part-{i:05d}.parquet').exists() and (p.EXP/f'cache/lightgbm_predictions/{c}/s{s}/part-{i:05d}.parquet').exists()) for c,s,i in tasks())
            if missing==0:break
        if not progressed:time.sleep(3)
    print('FEATURE_WORKER_DONE',a.worker,flush=True)

if __name__=='__main__':main()
