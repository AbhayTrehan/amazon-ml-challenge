#!/usr/bin/env python3
"""Compare documented non-destructive normalization views and collision risk."""
import csv,json,time,unicodedata,re
from pathlib import Path
import duckdb,pandas as pd
from common import nfkc,punct_norm,token_norm,suffix_norm,accent_fold
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'analysis/data_audit_v1/artifacts';DB=OUT/'audit.duckdb';SEED=20260925
con=duckdb.connect(str(DB),read_only=True);con.execute("PRAGMA threads=6");con.execute("PRAGMA memory_limit='2GB'");start=time.time()
aux="(select 2 src,* from train_s2 union all by name select 3 src,* from train_s3)"
views={
 'R0_raw':('business_name','business_address'),
 'R1_case_ws':("trim(regexp_replace(lower(business_name),'\\s+',' ','g'))","trim(regexp_replace(lower(business_address),'\\s+',' ','g'))"),
 'R2_punct':('nname','naddr'),
 'R4_legal_suffix':("trim(regexp_replace(nname,'\\s+(llc|llp|inc|incorporated|corp|corporation|co|company|ltd|limited|pvt|private|plc|lp|sarl|sas|sa|gmbh|bv|pte|firm|enterprises|enterprise)$','','i'))",'naddr'),
 'R5_accent_fold':('strip_accents(nname)','strip_accents(naddr)')}
recs=[]
for label,(ne,ae) in views.items():
 p=con.execute(f"""select count(*) edges,count(*)filter(where {ne.replace('business_','s.business_').replace('nname','s.nname').replace('naddr','s.naddr')}={ne.replace('business_','a.business_').replace('nname','a.nname').replace('naddr','a.naddr')} and coalesce({ne.replace('business_','s.business_').replace('nname','s.nname').replace('naddr','s.naddr')},'')<>'') name_exact,
 count(*)filter(where {ae.replace('business_','s.business_').replace('nname','s.nname').replace('naddr','s.naddr')}={ae.replace('business_','a.business_').replace('nname','a.nname').replace('naddr','a.naddr')} and coalesce({ae.replace('business_','s.business_').replace('nname','s.nname').replace('naddr','s.naddr')},'')<>'') addr_exact
 from edges e join train_s1 s on e.s1=s.entity_id join {aux}a on e.target_id=a.entity_id and e.target_source=a.src""").fetchone()
 # S1 name collisions among distinct reference identities.
 c=con.execute(f"""with g as(select {ne} v,count(*)n from train_s1 where coalesce({ne},'')<>''group by 1)
 select count(*)filter(where n>1),sum(n)filter(where n>1),max(n) from g""").fetchone()
 recs.append({'view':label,'positive_edges':p[0],'positive_name_exact':p[1],'positive_name_exact_rate':p[1]/p[0],'positive_address_exact':p[2],'positive_address_exact_rate':p[2]/p[0],'s1_colliding_name_values':c[0],'s1_records_in_name_collisions':c[1]or 0,'s1_max_name_frequency':c[2]})

# True NFKC/casefold and token-order views on fixed positive sample because DuckDB lacks NFKC.
sample=con.execute(f"""select * exclude(rn)from(select s.business_name sn,s.business_address sa,a.business_name an,a.business_address aa,
 row_number()over(partition by e.target_source,s.country order by hash(e.s1,e.target_id,{SEED}))rn from edges e join train_s1 s on e.s1=s.entity_id join {aux}a on e.target_id=a.entity_id and e.target_source=a.src)where rn<=20000""").fetchdf()
for label,fun in [('R1_true_NFKC_casefold',nfkc),('R3_tokenized',token_norm)]:
 nex=sum(bool(fun(x))and fun(x)==fun(y)for x,y in zip(sample.sn,sample.an));aex=sum(bool(fun(x))and fun(x)==fun(y)for x,y in zip(sample.sa.fillna(''),sample.aa.fillna('')))
 recs.append({'view':label+' (80k sample)','positive_edges':len(sample),'positive_name_exact':nex,'positive_name_exact_rate':nex/len(sample),'positive_address_exact':aex,'positive_address_exact_rate':aex/len(sample),'s1_colliding_name_values':None,'s1_records_in_name_collisions':None,'s1_max_name_frequency':None})
pd.DataFrame(recs).to_csv(OUT/'normalization_view_comparison.csv',index=False)
(OUT/'normalization_rules.json').write_text(json.dumps({'R0':'raw','R1':'Unicode NFKC + casefold + whitespace collapse (80k deterministic positive sample); full SQL R1 uses lower+whitespace because DuckDB has no NFKC','R2':'R1-like lowercase + all non-letter/non-number to spaces + whitespace collapse; Unicode preserved','R3':'R2 token sequence joined by one space','R4':'R2 name with repeatedly trailing listed legal suffixes removed','R4_suffixes':['llc','llp','inc','incorporated','corp','corporation','co','company','ltd','limited','pvt','private','plc','lp','sarl','sas','sa','gmbh','bv','pte','firm','enterprises','enterprise'],'R5':'R2 then Unicode accent stripping','seed':SEED,'runtime_seconds':time.time()-start},indent=2))
print(pd.DataFrame(recs).to_string(index=False))
