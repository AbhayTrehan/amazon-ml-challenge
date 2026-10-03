#!/usr/bin/env python3
"""Scale, exact-block distribution, ambiguity, and retrieval hard negatives."""
from pathlib import Path
import duckdb,pandas as pd,numpy as np,json,time
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'analysis/data_audit_v1/artifacts';DB=OUT/'audit.duckdb';con=duckdb.connect(str(DB),read_only=True);con.execute("PRAGMA threads=6");con.execute("PRAGMA memory_limit='2GB'");start=time.time()
# Search-space scale.
sizes={t:con.execute(f'select count(*)from {t}').fetchone()[0]for t in['train_s1','train_s2','train_s3']}
country=con.execute("""select country,(select count(*)from train_s1 where country=c.country)s1,(select count(*)from train_s2 where country=c.country)s2,(select count(*)from train_s3 where country=c.country)s3 from(select distinct country from train_s1)c""").fetchdf()
scale=[{'scope':'unrestricted','s1xs2':sizes['train_s1']*sizes['train_s2'],'s1xs3':sizes['train_s1']*sizes['train_s3'],'total':sizes['train_s1']*(sizes['train_s2']+sizes['train_s3'])}]
for r in country.itertuples():scale.append({'scope':r.country,'s1xs2':r.s1*r.s2,'s1xs3':r.s1*r.s3,'total':r.s1*(r.s2+r.s3)})
pd.DataFrame(scale).to_csv(OUT/'search_space_scale.csv',index=False)

# Exact union candidates per S1 without materializing pair rows.
q="""WITH an AS(select country,nname,count(*)n from(select*from train_s2 union all by name select*from train_s3)where nname<>''group by 1,2),
aa AS(select country,naddr,count(*)n from(select*from train_s2 union all by name select*from train_s3)where naddr<>''group by 1,2),
ab AS(select country,nname,naddr,count(*)n from(select*from train_s2 union all by name select*from train_s3)where nname<>''and naddr<>''group by 1,2,3),
c AS(select s.entity_id,s.country,coalesce(an.n,0)+coalesce(aa.n,0)-coalesce(ab.n,0)n from train_s1 s left join an using(country,nname)left join aa using(country,naddr)left join ab using(country,nname,naddr))
select coalesce(country,'ALL')country,count(*)queries,sum(n)candidate_pairs,avg(n)mean_candidates,quantile_disc(n,.5)p50,quantile_disc(n,.9)p90,quantile_disc(n,.95)p95,quantile_disc(n,.99)p99,max(n)max from c group by grouping sets((country),())order by country"""
con.execute(f"copy({q})to'{OUT/'exact_block_candidate_distribution.csv'}'(header,delimiter',')")

# Exact block oracle by country/cardinality.
q2="""WITH h AS(select e.s1,e.target_id,(s.nname<>''and s.nname=a.nname)or(s.naddr<>''and s.naddr=a.naddr)hit from edges e join train_s1 s on e.s1=s.entity_id join(select 2 src,*from train_s2 union all by name select 3 src,*from train_s3)a on e.target_id=a.entity_id and e.target_source=a.src),
c AS(select g.source1_entity_id s1,s.country,count(h.target_id)n,count(h.target_id)filter(where hit)t from train_gt g join train_s1 s on g.source1_entity_id=s.entity_id left join h on g.source1_entity_id=h.s1 group by 1,2)
select country,n true_cardinality,count(*)entities,sum(t)retrieved_edges,avg(case when n=0 then 1 when t=0 then 0 else 1.25*t/(1.25*t+.25*(n-t))end)oracle_f05 from c group by 1,2 order by 1,2"""
con.execute(f"copy({q2})to'{OUT/'exact_block_oracle_by_group.csv'}'(header,delimiter',')")

# Retrieval rank/margin diagnostic and highest-scoring false candidates.
frames=[]
for channel in['name_char','address_char']:
 for src in[2,3]:
  for c in['India','US']:
   d=pd.read_parquet(OUT/f'blocking_{channel}_s{src}_{c}.parquet');frames.append(d)
d=pd.concat(frames,ignore_index=True);qids=pd.DataFrame({'s1':d.s1.unique()});con.register('qids',qids);truth=con.execute('select e.s1,e.target_id from edges e join qids q on e.s1=q.s1').fetchdf();truth['true']=1
z=d.merge(truth,on=['s1','target_id'],how='left');z['true']=z.true.fillna(0).astype(int)
top=z[z['rank']<=2].pivot_table(index=['s1','source','country','channel'],columns='rank',values=['score','true'],aggfunc='first').reset_index();top.columns=['_'.join(map(str,x)).rstrip('_')for x in top.columns];top['margin']=top.get('score_1',0)-top.get('score_2',0)
amb=top.groupby(['source','country','channel','true_1']).agg(queries=('s1','size'),top1_mean=('score_1','mean'),top1_p10=('score_1',lambda x:x.quantile(.1)),top1_p50=('score_1','median'),margin_mean=('margin','mean'),margin_p50=('margin','median')).reset_index();amb.to_csv(OUT/'retrieval_ambiguity.csv',index=False)
hard=z[z.true==0].sort_values('score',ascending=False).groupby(['channel','source','country']).head(75)
con.register('hard0',hard[['s1','target_id','channel','source','country','rank','score']]);detail=con.execute("""select h.*,s.business_name s1_name,s.business_address s1_address,a.business_name candidate_name,a.business_address candidate_address from hard0 h join train_s1 s on h.s1=s.entity_id join(select*from train_s2 union all by name select*from train_s3)a on h.target_id=a.entity_id""").fetchdf();detail.to_csv(OUT/'retrieval_hard_negative_examples.tsv',sep='\t',index=False)
(OUT/'extra_runtime.json').write_text(json.dumps({'seconds':time.time()-start},indent=2));print(pd.DataFrame(scale).to_string(index=False));print(amb.to_string(index=False))
