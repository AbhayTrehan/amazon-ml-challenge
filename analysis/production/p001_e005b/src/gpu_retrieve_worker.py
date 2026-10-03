#!/usr/bin/env python3
"""Exact GPU K20 retrieval producer; final K10 filtering remains in E001 features."""
from __future__ import annotations
import argparse,gc,json,time
from pathlib import Path
import numpy as np,pandas as pd

import benchmark_gpu_retrieval as gpu
import production as p


def atomic_parquet(frame,path):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.parquet.tmp')
    frame.to_parquet(tmp,index=False,compression='zstd');tmp.replace(path)


def run_partition(country,source,batch):
    q=pd.read_parquet(p.query_path(country));aux=pd.read_parquet(p.index_paths(country,source,'name')[1]);results={};mats={};details={}
    for ch,col in [('name','business_name'),('address','business_address')]:
        xa,tf=gpu.matrix(country,source,ch);xq=tf.transform(p.e001.vec_for(ch).transform(q[col].fillna(''))).astype(np.float32).tocsr()
        t=time.time();inds,vals,times,peak=gpu.exact_gpu_sdtn(xq,xa,batch);sec=time.time()-t
        results[ch]=(inds,vals);mats[ch]=(xa,xq);details[ch]={'seconds':sec,'qps':len(q)/sec,'peak_gpu_gb':peak/2**30}
        print('GPU_CHANNEL',country,source,ch,json.dumps(details[ch]),flush=True)
    outdir=p.EXP/f'cache/candidates/{country}/s{source}';outdir.mkdir(parents=True,exist_ok=True)
    for part,start in enumerate(range(0,len(q),p.SHARD_SIZE)):
        op=outdir/f'part-{part:05d}.parquet'
        if op.exists():continue
        stop=min(len(q),start+p.SHARD_SIZE);n=stop-start;frames=[]
        for ch in ['name','address']:
            inds,vals=results[ch];frames.append(pd.DataFrame({'qpos':np.repeat(np.arange(n,dtype=np.int32),20),'apos':inds[start:stop].ravel().astype(np.int32),f'{ch}_rank':np.tile(np.arange(1,21,dtype=np.int16),n),f'{ch}_retrieval_score':vals[start:stop].ravel().astype(np.float32)}))
        d=frames[0].merge(frames[1],on=['qpos','apos'],how='outer');qs=q.iloc[start:stop].reset_index(drop=True)
        d['s1']=qs.entity_id.to_numpy()[d.qpos];d['target_id']=aux.entity_id.to_numpy()[d.apos]
        for ch in ['name','address']:
            xa,xq=mats[ch];d[f'{ch}_char_tfidf']=p.e001.row_cosines(xq[start:stop],xa,d.qpos.to_numpy(),d.apos.to_numpy())
        atomic_parquet(d,op);print('CANDIDATE_READY',country,source,part,len(d),flush=True)
    diag=p.EXP/f'diagnostics/gpu_retrieval_{country}_s{source}.json';p.atomic_json(diag,{'country':country,'source':source,'rows':len(q),'batch':batch,'channels':details})
    del q,aux,results,mats;gc.collect()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tasks',required=True);ap.add_argument('--batch',type=int,default=1024);ap.add_argument('--done',required=True);a=ap.parse_args()
    for spec in a.tasks.split(','):
        country,source=spec.split(':');run_partition(country,int(source),a.batch)
    Path(a.done).write_text(f'done {time.time()}\n')

if __name__=='__main__':main()
