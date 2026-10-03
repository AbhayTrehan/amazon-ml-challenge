#!/usr/bin/env python3
"""Sampled-query/full-index lexical top-K blocking benchmark (deterministic)."""
import argparse,gc,json,time
from pathlib import Path
import duckdb,numpy as np,pandas as pd,psutil
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import HashingVectorizer,TfidfTransformer
from sparse_dot_topn import sp_matmul_topn

ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/'analysis/data_audit_v1/artifacts'; DB=OUT/'audit.duckdb'; SEED=20260925
p=argparse.ArgumentParser();p.add_argument('--source',type=int,required=True);p.add_argument('--country',required=True);p.add_argument('--channel',choices=['name_char','address_char','concat_char','name_word'],required=True);p.add_argument('--queries',type=int,default=5000);p.add_argument('--topk',type=int,default=100);a=p.parse_args()
con=duckdb.connect(str(DB),read_only=True);con.execute("PRAGMA threads=6");con.execute("PRAGMA memory_limit='3GB'")
t0=time.time(); proc=psutil.Process(); peak=proc.memory_info().rss
field={'name_char':"coalesce(business_name,'')",'address_char':"coalesce(business_address,'')",'concat_char':"coalesce(business_name,'')||' '||coalesce(business_address,'')",'name_word':"coalesce(business_name,'')"}[a.channel]
qs=con.execute(f"""SELECT * exclude(rn) FROM (SELECT entity_id,{field} AS txt,row_number()over(order by hash(entity_id,{SEED}))rn
 FROM train_s1 WHERE country=?) WHERE rn<=?""",[a.country,a.queries]).fetchdf()
aux=con.execute(f"SELECT entity_id,{field} AS txt FROM train_s{a.source} WHERE country=?",[a.country]).fetchdf(); load=time.time()-t0;peak=max(peak,proc.memory_info().rss)
if a.channel=='name_word': vec=HashingVectorizer(n_features=2**18,analyzer='word',ngram_range=(1,2),alternate_sign=False,norm=None,dtype=np.float32)
elif a.channel=='address_char': vec=HashingVectorizer(n_features=2**17,analyzer='char_wb',ngram_range=(3,3),alternate_sign=False,norm=None,dtype=np.float32)
else: vec=HashingVectorizer(n_features=2**18,analyzer='char_wb',ngram_range=(3,5),alternate_sign=False,norm=None,dtype=np.float32)
t=time.time(); Xa=vec.transform(aux.txt.fillna('')); Xq=vec.transform(qs.txt.fillna('')); transform=time.time()-t;peak=max(peak,proc.memory_info().rss)
t=time.time(); tf=TfidfTransformer(); Xa=tf.fit_transform(Xa);Xq=tf.transform(Xq);tfidf=time.time()-t;peak=max(peak,proc.memory_info().rss)
t=time.time(); C=sp_matmul_topn(Xq,Xa.T,top_n=a.topk,threshold=1e-8,sort=True,n_threads=6); querytime=time.time()-t;peak=max(peak,proc.memory_info().rss)
# Convert compact top-K to records.
ridx=np.repeat(np.arange(C.shape[0]),np.diff(C.indptr)); ranks=np.concatenate([np.arange(C.indptr[i+1]-C.indptr[i])+1 for i in range(C.shape[0])])
res=pd.DataFrame({'s1':qs.entity_id.to_numpy()[ridx],'target_id':aux.entity_id.to_numpy()[C.indices],'rank':ranks,'score':C.data})
res['source']=a.source;res['country']=a.country;res['channel']=a.channel
path=OUT/f"blocking_{a.channel}_s{a.source}_{a.country}.parquet";res.to_parquet(path,index=False)
# Full truth for sampled queries, including singletons.
con.register('qids',qs[['entity_id']]); truth=con.execute("select q.entity_id s1,e.target_id from qids q left join edges e on q.entity_id=e.s1 and e.target_source=?",[a.source]).fetchdf()
truthmap={k:set(g.target_id.dropna()) for k,g in truth.groupby('s1')}; pred_by_k={}
rows=[]
for k in [1,2,3,5,10,20,50,100]:
 if k>a.topk:continue
 rr=res[res['rank']<=k]; pmap={x:set(g.target_id) for x,g in rr.groupby('s1')}
 edge_total=edge_hit=pos=anyhit=allhit=0; scores=[]
 for s1 in qs.entity_id:
  tr=truthmap.get(s1,set());pr=pmap.get(s1,set()); edge_total+=len(tr);edge_hit+=len(tr&pr)
  if tr: pos+=1;anyhit+=bool(tr&pr);allhit+=tr<=pr
  # Source-specific oracle: empty source truth contributes 1, else F0.5 on surviving truth.
  tp=len(tr&pr);scores.append(1 if not tr else (0 if not tp else 1.25*tp/(1.25*tp+.25*(len(tr)-tp))))
 counts=rr.groupby('s1').size().reindex(qs.entity_id,fill_value=0)
 rows.append({'channel':a.channel,'source':a.source,'country':a.country,'sample_queries':len(qs),'k':k,'true_edges':edge_total,'retrieved_edges':edge_hit,'candidate_recall':edge_hit/edge_total if edge_total else None,'candidate_pairs':len(rr),'mean_candidates':counts.mean(),'p50_candidates':counts.quantile(.5),'p95_candidates':counts.quantile(.95),'p99_candidates':counts.quantile(.99),'positive_entities':pos,'positive_any':anyhit,'positive_all':allhit,'source_specific_oracle_f05':np.mean(scores),'load_seconds':load,'transform_seconds':transform,'tfidf_seconds':tfidf,'query_seconds':querytime,'peak_rss_mb':peak/2**20,'matrix_aux_rows':Xa.shape[0],'matrix_features':Xa.shape[1],'matrix_aux_nnz':Xa.nnz,'matrix_query_nnz':Xq.nnz,'artifact_mb':path.stat().st_size/2**20})
pd.DataFrame(rows).to_csv(OUT/f"blocking_{a.channel}_s{a.source}_{a.country}_metrics.csv",index=False)
print(json.dumps(rows[-1],indent=2))
