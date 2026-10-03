#!/usr/bin/env python3
"""Unlabeled France lexical-neighbor confidence (full test-France indexes)."""
import gc,time
from pathlib import Path
import duckdb,numpy as np,pandas as pd,psutil
from sklearn.feature_extraction.text import HashingVectorizer,TfidfTransformer
from sparse_dot_topn import sp_matmul_topn
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'analysis/data_audit_v1/artifacts';DB=OUT/'audit.duckdb';SEED=20260925
con=duckdb.connect(str(DB),read_only=True); rows=[]; proc=psutil.Process()
for src in [2,3]:
 for channel in ['name_char','address_char']:
  fld="coalesce(business_name,'')"if channel=='name_char'else"coalesce(business_address,'')"
  q=con.execute(f"select entity_id,{fld}txt from test_s1 where country='France' order by hash(entity_id,{SEED})limit 1000").fetchdf()
  a=con.execute(f"select entity_id,{fld}txt from test_s{src} where country='France'").fetchdf();t=time.time()
  vec=HashingVectorizer(n_features=2**18 if channel=='name_char'else 2**17,analyzer='char_wb',ngram_range=(3,5)if channel=='name_char'else(3,3),alternate_sign=False,norm=None,dtype=np.float32)
  A=vec.transform(a.txt.fillna(''));Q=vec.transform(q.txt.fillna(''));tf=TfidfTransformer();A=tf.fit_transform(A);Q=tf.transform(Q)
  C=sp_matmul_topn(Q,A.T,top_n=2,threshold=1e-8,sort=True,n_threads=6)
  top1=[];top2=[]
  for i in range(len(q)):
   x=C.data[C.indptr[i]:C.indptr[i+1]];top1.append(x[0]if len(x)else 0);top2.append(x[1]if len(x)>1 else 0)
  x=np.array(top1);y=np.array(top2)
  rows.append({'population':'France test unlabeled','source':src,'channel':channel,'queries':len(q),'top1_mean':x.mean(),'top1_p10':np.quantile(x,.1),'top1_p50':np.quantile(x,.5),'top1_p90':np.quantile(x,.9),'margin_mean':(x-y).mean(),'margin_p50':np.quantile(x-y,.5),'seconds':time.time()-t,'peak_current_rss_mb':proc.memory_info().rss/2**20})
  del a,A,Q,C,tf,vec;gc.collect()
# Comparable labeled-train query top score from stored blockers.
for src in [2,3]:
 for channel in ['name_char','address_char']:
  for country in ['US','India']:
   d=pd.read_parquet(OUT/f'blocking_{channel}_s{src}_{country}.parquet');x=d[d['rank']==1].score.to_numpy()
   rows.append({'population':country+' train queries','source':src,'channel':channel,'queries':len(x),'top1_mean':x.mean(),'top1_p10':np.quantile(x,.1),'top1_p50':np.quantile(x,.5),'top1_p90':np.quantile(x,.9),'margin_mean':None,'margin_p50':None,'seconds':None,'peak_current_rss_mb':None})
pd.DataFrame(rows).to_csv(OUT/'france_retrieval_confidence.csv',index=False);print(pd.DataFrame(rows).to_string(index=False))
