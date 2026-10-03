#!/usr/bin/env python3
"""Empirical random controls and modulo tests for ID/row artifact analysis."""
from pathlib import Path
import duckdb,pandas as pd
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'analysis/data_audit_v1/artifacts';con=duckdb.connect(str(OUT/'audit.duckdb'),read_only=True);SEED=20260925
parts=[]
for src in [2,3]:
 for country in ['India','US']:
  q=f"""WITH s AS(select *,row_number()over(order by hash(entity_id,{SEED}))rn from train_s1 where country='{country}' qualify rn<=20000),
  a AS(select *,row_number()over(order by hash(entity_id,{SEED+src}))rn from train_s{src} where country='{country}' qualify rn<=20000)
  SELECT 'random_same_country' AS family,{src} AS target_source,'{country}' AS country,count(*) AS n,
  avg(abs(try_cast(split_part(s.entity_id,'-',2)as bigint)-try_cast(split_part(a.entity_id,'-',2)as bigint)))mean_iddiff,
  quantile_disc(abs(try_cast(split_part(s.entity_id,'-',2)as bigint)-try_cast(split_part(a.entity_id,'-',2)as bigint)),.01)p1,
  quantile_disc(abs(try_cast(split_part(s.entity_id,'-',2)as bigint)-try_cast(split_part(a.entity_id,'-',2)as bigint)),.5)p50,
  quantile_disc(abs(try_cast(split_part(s.entity_id,'-',2)as bigint)-try_cast(split_part(a.entity_id,'-',2)as bigint)),.99)p99,
  avg(abs(s.row_num::double/(select count(*)from train_s1)-a.row_num::double/(select count(*)from train_s{src})))mean_pctdiff,
  quantile_disc(abs(s.row_num::double/(select count(*)from train_s1)-a.row_num::double/(select count(*)from train_s{src})),.5)pct_p50,
  count(*)filter(where split_part(s.entity_id,'-',2)=split_part(a.entity_id,'-',2))same_suffix,
  count(*)filter(where try_cast(split_part(s.entity_id,'-',2)as bigint)%100=try_cast(split_part(a.entity_id,'-',2)as bigint)%100)same_mod100,
  count(*)filter(where try_cast(split_part(s.entity_id,'-',2)as bigint)%1000=try_cast(split_part(a.entity_id,'-',2)as bigint)%1000)same_mod1000 FROM s join a using(rn)"""
  parts.extend(con.execute(q).fetchdf().to_dict('records'))
pos=con.execute("""SELECT 'positive' AS family,e.target_source,s.country,count(*) AS n,
avg(abs(try_cast(split_part(s.entity_id,'-',2)as bigint)-try_cast(split_part(a.entity_id,'-',2)as bigint)))mean_iddiff,
quantile_disc(abs(try_cast(split_part(s.entity_id,'-',2)as bigint)-try_cast(split_part(a.entity_id,'-',2)as bigint)),.01)p1,
quantile_disc(abs(try_cast(split_part(s.entity_id,'-',2)as bigint)-try_cast(split_part(a.entity_id,'-',2)as bigint)),.5)p50,
quantile_disc(abs(try_cast(split_part(s.entity_id,'-',2)as bigint)-try_cast(split_part(a.entity_id,'-',2)as bigint)),.99)p99,
avg(abs(s.row_num::double/(select count(*)from train_s1)-a.row_num::double/case when target_source=2 then(select count(*)from train_s2)else(select count(*)from train_s3)end))mean_pctdiff,
quantile_disc(abs(s.row_num::double/(select count(*)from train_s1)-a.row_num::double/case when target_source=2 then(select count(*)from train_s2)else(select count(*)from train_s3)end),.5)pct_p50,
count(*)filter(where split_part(s.entity_id,'-',2)=split_part(a.entity_id,'-',2))same_suffix,
count(*)filter(where try_cast(split_part(s.entity_id,'-',2)as bigint)%100=try_cast(split_part(a.entity_id,'-',2)as bigint)%100)same_mod100,
count(*)filter(where try_cast(split_part(s.entity_id,'-',2)as bigint)%1000=try_cast(split_part(a.entity_id,'-',2)as bigint)%1000)same_mod1000
FROM edges e join train_s1 s on e.s1=s.entity_id join(select 2 src,*from train_s2 union all by name select 3 src,*from train_s3)a on e.target_id=a.entity_id and e.target_source=a.src group by all order by 2,3""").fetchdf()
pd.concat([pos,pd.DataFrame(parts)],ignore_index=True).to_csv(OUT/'id_order_artifact_controls.csv',index=False)
over=[]
for src in[1,2,3]:over.append({'source':src,'train_test_id_overlap':con.execute(f'select count(*) from train_s{src} a join test_s{src} b using(entity_id)').fetchone()[0]})
pd.DataFrame(over).to_csv(OUT/'train_test_id_overlap.csv',index=False)
print(pd.concat([pos,pd.DataFrame(parts)],ignore_index=True).to_string(index=False));print(over)
