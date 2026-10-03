#!/usr/bin/env python3
"""Exact SQL analyses: integrity, inventory, graph, metric, collisions, exact blocking."""
import csv, json, math, os, time
from pathlib import Path
import duckdb

ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/'analysis/data_audit_v1/artifacts'; DB=OUT/'audit.duckdb'
con=duckdb.connect(str(DB)); con.execute("PRAGMA threads=6"); con.execute("PRAGMA memory_limit='3GB'"); con.execute(f"PRAGMA temp_directory='{OUT/'ducktmp'}'")
tables=['train_s1','train_s2','train_s3','test_s1','test_s2','test_s3']
start=time.time()
def rows(sql,p=None): return con.execute(sql,p or []).fetchdf().to_dict('records')
def one(sql,p=None): return con.execute(sql,p or []).fetchone()
def dump(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str))
def csvout(name,sql): con.execute(f"COPY ({sql}) TO '{OUT/name}' (HEADER, DELIMITER ',')")

# Ground truth edges, including source/country metadata.
con.execute("""CREATE OR REPLACE TABLE edges AS
 SELECT source1_entity_id s1, trim(x) target_id,
        CASE WHEN starts_with(trim(x),'S2-') THEN 2 WHEN starts_with(trim(x),'S3-') THEN 3 ELSE 0 END target_source
 FROM train_gt, unnest(string_split(coalesce(matched_entity_ids,''), ',')) u(x) WHERE trim(x)<>''""")
con.execute("CREATE INDEX IF NOT EXISTS edges_s1_idx ON edges(s1)"); con.execute("CREATE INDEX IF NOT EXISTS edges_target_idx ON edges(target_id)")

# Schema/integrity.
integrity={}
for t in tables:
    expected=f"S{t[-1]}-"
    integrity[t]=dict(zip(['rows','distinct_ids','duplicate_id_excess','missing_id','bad_prefix','duplicate_row_excess','min_numeric','max_numeric','numeric_distinct','row_numeric_spearman'],one(f"""
      SELECT count(*),count(distinct entity_id),count(*)-count(distinct entity_id),count(*) filter(where entity_id is null or entity_id=''),
      count(*) filter(where not starts_with(entity_id,'{expected}') or try_cast(split_part(entity_id,'-',2) as bigint) is null),
      count(*)-count(distinct hash(entity_id,business_name,business_address,country)),
      min(try_cast(split_part(entity_id,'-',2) as bigint)),max(try_cast(split_part(entity_id,'-',2) as bigint)),count(distinct try_cast(split_part(entity_id,'-',2) as bigint)),
      corr(row_num,try_cast(split_part(entity_id,'-',2) as bigint)) FROM {t}""")))
integrity['ground_truth']=dict(zip(['rows','distinct_s1','duplicate_s1_excess','empty_cells','unknown_s1','malformed_target_prefix','duplicate_targets_within_lists','unknown_targets'],one("""
 SELECT count(*),count(distinct source1_entity_id),count(*)-count(distinct source1_entity_id),
 count(*) filter(where matched_entity_ids is null or matched_entity_ids=''),
 count(*) filter(where s.entity_id is null),
 (select count(*) from edges where target_source=0),
 (select count(*) from (select s1,target_id,count(*) n from edges group by 1,2 having n>1)),
 (select count(*) from edges e left join (select entity_id from train_s2 union all select entity_id from train_s3) a on e.target_id=a.entity_id where a.entity_id is null)
 FROM train_gt g left join train_s1 s on g.source1_entity_id=s.entity_id""")))
integrity['gt_missing_s1_rows']=one("select count(*) from train_s1 s left join train_gt g on s.entity_id=g.source1_entity_id where g.source1_entity_id is null")[0]
integrity['gt_unknown_s1_examples']=rows("select source1_entity_id from train_gt g left join train_s1 s on g.source1_entity_id=s.entity_id where s.entity_id is null limit 20")
dump('integrity.json',integrity)

# Core inventory / field profiles. Quantiles are exact discrete quantiles.
profile=[]
for t in tables:
  split,source=t.split('_'); source=source.upper()
  sql=f"""SELECT country,count(*) n,
   count(*) filter(where business_name is null) name_null,
   count(*) filter(where business_name='') name_empty,
   count(*) filter(where business_name is not null and business_name<>'' and trim(business_name)='') name_whitespace,
   count(*) filter(where lower(trim(coalesce(business_name,''))) in ('na','n/a','null','none','nil','-','--','unknown')) name_sentinel,
   count(*) filter(where business_address is null) address_null,
   count(*) filter(where business_address='') address_empty,
   count(*) filter(where business_address is not null and business_address<>'' and trim(business_address)='') address_whitespace,
   count(*) filter(where lower(trim(coalesce(business_address,''))) in ('na','n/a','null','none','nil','-','--','unknown')) address_sentinel,
   count(*) filter(where coalesce(business_name,'')='' and coalesce(business_address,'')='') both_missing,
   min(length(business_name)) name_min, quantile_disc(length(business_name),.01) name_p1,quantile_disc(length(business_name),.05) name_p5,
   quantile_disc(length(business_name),.5) name_p50,avg(length(business_name)) name_mean,quantile_disc(length(business_name),.95) name_p95,quantile_disc(length(business_name),.99) name_p99,max(length(business_name)) name_max,
   min(length(business_address)) addr_min,quantile_disc(length(business_address),.01) addr_p1,quantile_disc(length(business_address),.05) addr_p5,
   quantile_disc(length(business_address),.5) addr_p50,avg(length(business_address)) addr_mean,quantile_disc(length(business_address),.95) addr_p95,quantile_disc(length(business_address),.99) addr_p99,max(length(business_address)) addr_max,
   avg(length(regexp_replace(coalesce(business_name,''),'[^0-9]','','g'))) name_mean_digits,
   avg(length(regexp_replace(coalesce(business_address,''),'[^0-9]','','g'))) addr_mean_digits,
   count(*) filter(where regexp_matches(coalesce(business_name,''),'[^\\x00-\\x7F]')) name_nonascii,
   count(*) filter(where regexp_matches(coalesce(business_address,''),'[^\\x00-\\x7F]')) addr_nonascii,
   avg(array_length(string_split(trim(coalesce(business_name,'')),' '))) name_mean_tokens,
   avg(array_length(string_split(trim(coalesce(business_address,'')),' '))) addr_mean_tokens
   FROM {t} GROUP BY country ORDER BY country"""
  for r in rows(sql): r.update(split=split,source=source); profile.append(r)
with (OUT/'schema_and_missingness.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(profile[0]));w.writeheader();w.writerows(profile)

# Graph/cardinality.
csvout('ground_truth_cardinality.csv',"""WITH c AS (SELECT g.source1_entity_id s1,s.country,
 count(e.target_id) n,count(e.target_id) filter(where target_source=2) n2,count(e.target_id) filter(where target_source=3) n3
 FROM train_gt g JOIN train_s1 s ON g.source1_entity_id=s.entity_id LEFT JOIN edges e ON g.source1_entity_id=e.s1 GROUP BY 1,2)
 SELECT country,n,n2,n3,count(*) entities FROM c GROUP BY all ORDER BY country,n,n2,n3""")
card=rows("""WITH c AS (SELECT g.source1_entity_id s1,s.country,count(e.target_id) n,count(e.target_id) filter(where target_source=2)n2,count(e.target_id)filter(where target_source=3)n3 FROM train_gt g join train_s1 s on g.source1_entity_id=s.entity_id left join edges e on g.source1_entity_id=e.s1 group by 1,2)
 SELECT coalesce(country,'ALL') country,count(*) entities,sum(n) edges,count(*)filter(where n=0) n0,count(*)filter(where n=1)n1,count(*)filter(where n=2)n2,count(*)filter(where n=3)n3,count(*)filter(where n=4)n4,count(*)filter(where n>=5)n5plus,
 avg(n) filter(where n>0) matched_mean,min(n)filter(where n>0)matched_min,quantile_disc(n,.5)filter(where n>0)matched_median,quantile_disc(n,.9)filter(where n>0)p90,quantile_disc(n,.95)filter(where n>0)p95,quantile_disc(n,.99)filter(where n>0)p99,max(n) matched_max,
 count(*)filter(where n2>0 and n3=0)s2_only,count(*)filter(where n3>0 and n2=0)s3_only,count(*)filter(where n2>0 and n3>0)both_sources
 FROM c GROUP BY grouping sets((country),()) ORDER BY country""")
inv=rows("""WITH refs AS (SELECT target_id,count(*) refs FROM edges GROUP BY target_id), allr AS (
 SELECT 2 src,entity_id,country FROM train_s2 UNION ALL SELECT 3,entity_id,country FROM train_s3)
 SELECT src,coalesce(country,'ALL')country,count(*) total,count(*)filter(where coalesce(refs,0)=0) refs0,count(*)filter(where refs=1)refs1,count(*)filter(where refs>1)refs_gt1,max(coalesce(refs,0))max_refs
 FROM allr left join refs on entity_id=target_id GROUP BY grouping sets((src,country),(src)) ORDER BY src,country""")
# Fix alias ambiguity via separate exact query if needed above.
dump('ground_truth_graph.json',{'cardinality':card,'inverse':inv,'total_edges':one('select count(*) from edges')[0],
 'multi_ref_examples':rows("select target_id,count(*) refs,string_agg(s1,',') s1_ids from edges group by target_id having count(*)>1 order by refs desc,target_id limit 1000")})

# Country link consistency and source comparisons.
csvout('country_mismatches.csv',"""SELECT e.s1,e.target_id,e.target_source,s.country s1_country,a.country target_country
 FROM edges e join train_s1 s on e.s1=s.entity_id join (select entity_id,country from train_s2 union all select entity_id,country from train_s3)a on e.target_id=a.entity_id
 WHERE s.country<>a.country""")

# Full normalized collision summaries.
collision=[]
for t in tables:
 for field in ['business_name','business_address','nname','naddr']:
  r=one(f"""WITH g AS (select {field} v,count(*) n from {t} where coalesce({field},'')<>'' group by 1)
   SELECT count(*) unique_values,sum(n) records_nonempty,count(*)filter(where n>1) duplicated_values,sum(n)filter(where n>1) records_in_collisions,max(n) max_frequency FROM g""")
  collision.append(dict(table=t,field=field,unique_values=r[0],records_nonempty=r[1],duplicated_values=r[2],records_in_collisions=r[3],max_frequency=r[4]))
with (OUT/'normalization_stats.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=collision[0]);w.writeheader();w.writerows(collision)

# Positive exactness by source/country.
csvout('positive_exactness.csv',"""WITH p AS (SELECT e.*,s.country,s.business_name sn,s.business_address sa,s.nname snn,s.naddr sna,
 a.business_name an,a.business_address aa,a.nname ann,a.naddr ana
 FROM edges e join train_s1 s on e.s1=s.entity_id join (select 2 src,* exclude(row_num) from train_s2 union all select 3,* exclude(row_num) from train_s3)a on e.target_id=a.entity_id and e.target_source=a.src)
 SELECT target_source,country,count(*) edges,
 count(*)filter(where sn=an) raw_name_exact,count(*)filter(where sa=aa)raw_addr_exact,
 count(*)filter(where snn=ann and snn<>'')norm_name_exact,count(*)filter(where sna=ana and sna<>'')norm_addr_exact,
 count(*)filter(where sn=an and sa=aa)raw_both_exact,count(*)filter(where snn=ann and sna=ana and (snn<>'' or sna<>''))norm_both_exact,
 count(*)filter(where sn is null or an is null) name_missing_pair,count(*)filter(where sa is null or aa is null)addr_missing_pair
 FROM p GROUP BY all ORDER BY target_source,country""")

# Exact normalized blocker full-scale summary and oracle F.5.
con.execute("""CREATE OR REPLACE TEMP VIEW exact_edge_hit AS SELECT e.*,
 (s.nname<>'' and s.nname=a.nname) name_hit,(s.naddr<>'' and s.naddr=a.naddr) addr_hit
 FROM edges e join train_s1 s on e.s1=s.entity_id join (select 2 src,* exclude(row_num) from train_s2 union all select 3,* exclude(row_num) from train_s3)a on e.target_id=a.entity_id and e.target_source=a.src""")
bench=[]
pair_counts={}
for method,cond in [('exact_name','name_hit'),('exact_address','addr_hit'),('exact_union','name_hit or addr_hit')]:
  hit=one(f"select count(*) from exact_edge_hit where {cond}")[0]
  # candidate count via frequency products; union subtracts intersection.
  if method=='exact_name': expr="sum(ns.n*na.n)"; joins="join (select country,nname,count(*)n from train_s1 where nname<>'' group by 1,2)ns using(country,nname) join (select country,nname,count(*)n from (select * from train_s2 union all by name select * from train_s3) where nname<>'' group by 1,2)na using(country,nname)"
  elif method=='exact_address': expr="sum(ns.n*na.n)"; joins="join (select country,naddr,count(*)n from train_s1 where naddr<>'' group by 1,2)ns using(country,naddr) join (select country,naddr,count(*)n from (select * from train_s2 union all by name select * from train_s3) where naddr<>'' group by 1,2)na using(country,naddr)"
  else: expr=None
  if expr:
   key='nname' if method=='exact_name' else 'naddr'; pairs=one(f"select sum(x*y) from (select s.country,s.{key},count(*)x from train_s1 s where s.{key}<>'' group by 1,2)a join (select country,{key},count(*)y from (select * from train_s2 union all by name select * from train_s3) where {key}<>'' group by 1,2)b using(country,{key})")[0] or 0
   pair_counts[method]=pairs
  else:
   both=one("""SELECT sum(x*y) FROM (select country,nname,naddr,count(*)x from train_s1 where nname<>'' and naddr<>'' group by 1,2,3)a JOIN
     (select country,nname,naddr,count(*)y from (select * from train_s2 union all by name select * from train_s3) where nname<>'' and naddr<>'' group by 1,2,3)b USING(country,nname,naddr)""")[0] or 0
   pairs=pair_counts['exact_name']+pair_counts['exact_address']-both
  # Oracle per S1: prediction is hit truths only; singletons score 1.
  oracle=one(f"""WITH c AS (select g.source1_entity_id s1,count(e.target_id)n,count(e.target_id)filter(where {cond})t from train_gt g left join exact_edge_hit e on g.source1_entity_id=e.s1 group by 1)
   select avg(case when n=0 then 1.0 when t=0 then 0.0 else 1.25*t/(1.25*t+.25*(n-t)) end),count(*)filter(where n>0 and t>0),count(*)filter(where n>0 and t=n) from c""")
  bench.append({'method':method,'true_edges':one('select count(*) from edges')[0],'retrieved_edges':hit,'candidate_recall':hit/one('select count(*) from edges')[0], 'candidate_pairs':pairs,'mean_candidates_per_s1':pairs/one('select count(*) from train_s1')[0], 'oracle_macro_f05':oracle[0],'positive_entities_any':oracle[1],'positive_entities_all':oracle[2]})
with (OUT/'blocking_benchmarks.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=bench[0]);w.writeheader();w.writerows(bench)

# Deterministic validation fold at S1/entity unit, stratification diagnostics retained in columns.
con.execute(f"""COPY (WITH c AS (SELECT g.source1_entity_id,s.country,count(e.target_id)n,count(e.target_id)filter(where target_source=2)n2,count(e.target_id)filter(where target_source=3)n3
 FROM train_gt g join train_s1 s on g.source1_entity_id=s.entity_id left join edges e on g.source1_entity_id=e.s1 group by 1,2)
 SELECT source1_entity_id,country,n true_matches,n2 s2_matches,n3 s3_matches,abs(hash(source1_entity_id,20260925))%5 fold FROM c ORDER BY source1_entity_id)
 TO '{OUT/'proposed_validation_split.tsv'}' (HEADER,DELIMITER '\t')""")

# Metric tests and cardinality contribution.
def f(m,k,t): return 1.0 if m==0 and k==0 else (0.0 if m==0 or k==0 else 1.25*t/(1.25*t+.25*(m-t)+(k-t)))
tests=[('perfect_positive',2,2,2,1),('singleton_empty',0,0,0,1),('singleton_false',0,1,0,0),('one_tp',1,1,1,1),('one_tp_one_fp',1,2,1,5/9),('partial_recall',3,2,2,f(3,2,2)),('empty_positive',1,0,0,0),('readme_example',2,3,2,5/7)]
metric={'formula':'1.25*t / (1.25*t + 0.25*(m-t) + (k-t)); empty/empty=1','unit_tests':[{'case':x[0],'m':x[1],'k':x[2],'t':x[3],'score':f(x[1],x[2],x[3]),'expected':x[4]} for x in tests],
 'all_empty_score':one("select count(*) filter(where matched_entity_ids is null or matched_entity_ids='')::double/count(*) from train_gt")[0],
 'cardinality_contribution':rows("""WITH c AS(select g.source1_entity_id,count(e.target_id)n from train_gt g left join edges e on g.source1_entity_id=e.s1 group by 1)select n,count(*)entities,count(*)::double/(select count(*)from c) population_share from c group by n order by n""")}
dump('metric_forensics.json',metric)
dump('core_runtime.json',{'seconds':time.time()-start})
print('DONE',time.time()-start)
