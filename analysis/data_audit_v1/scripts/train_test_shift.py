#!/usr/bin/env python3
"""Full exact train/test overlaps plus deterministic sampled vocabulary shift."""
import json,time
from collections import Counter
from pathlib import Path
import duckdb,pandas as pd,numpy as np
from scipy.spatial.distance import jensenshannon
from common import tokens,punct_norm
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'analysis/data_audit_v1/artifacts';DB=OUT/'audit.duckdb';SEED=20260925
con=duckdb.connect(str(DB),read_only=True);con.execute("PRAGMA threads=6");con.execute("PRAGMA memory_limit='2GB'");start=time.time();recs=[]
for src in [1,2,3]:
 for country in ['US','India','France']:
  n=con.execute(f"select count(*) from test_s{src} where country=?",[country]).fetchone()[0]
  if not n:continue
  for field in ['business_name','business_address','nname','naddr']:
   same=con.execute(f"""select count(*) from test_s{src} t where country=? and coalesce(t.{field},'')<>'' and exists(select 1 from train_s{src} r where r.country=t.country and r.{field}=t.{field})""",[country]).fetchone()[0]
   anyc=con.execute(f"""select count(*) from test_s{src} t where country=? and coalesce(t.{field},'')<>'' and exists(select 1 from train_s{src} r where r.{field}=t.{field})""",[country]).fetchone()[0]
   recs.append({'source':src,'country':country,'field':field,'test_records':n,'same_country_train_overlap_records':same,'same_country_overlap_rate':same/n,'any_train_country_overlap_records':anyc,'any_country_overlap_rate':anyc/n})
 # pair overlaps
 for country in ['US','India','France']:
  n=con.execute(f"select count(*) from test_s{src} where country=?",[country]).fetchone()[0]
  if not n:continue
  for label,x,y in [('raw_pair','business_name','business_address'),('norm_pair','nname','naddr')]:
   same=con.execute(f"""select count(*) from test_s{src} t where country=? and exists(select 1 from train_s{src} r where r.country=t.country and r.{x}=t.{x} and r.{y}=t.{y})""",[country]).fetchone()[0]
   anyc=con.execute(f"""select count(*) from test_s{src} t where country=? and exists(select 1 from train_s{src} r where r.{x}=t.{x} and r.{y}=t.{y})""",[country]).fetchone()[0]
   recs.append({'source':src,'country':country,'field':label,'test_records':n,'same_country_train_overlap_records':same,'same_country_overlap_rate':same/n,'any_train_country_overlap_records':anyc,'any_country_overlap_rate':anyc/n})
pd.DataFrame(recs).to_csv(OUT/'train_test_overlap.csv',index=False)

# Sampled lexical shift: 100k/source train reference; 30k/source-country test.
train=[]
for src in [1,2,3]: train.append(con.execute(f"select business_name,business_address from train_s{src} order by hash(entity_id,?) limit 100000",[SEED]).fetchdf())
tr=pd.concat(train); ref_text=(tr.business_name.fillna('')+' '+tr.business_address.fillna('')).map(punct_norm)
ref_tokens=Counter(t for s in ref_text for t in tokens(s));ref_chars=Counter(s[i:i+3] for s in ref_text for i in range(max(0,len(s)-2)))
shift=[]
for src in [1,2,3]:
 for country in ['US','India','France']:
  te=con.execute(f"select business_name,business_address from test_s{src} where country=? order by hash(entity_id,?) limit 30000",[country,SEED]).fetchdf()
  if te.empty:continue
  txt=(te.business_name.fillna('')+' '+te.business_address.fillna('')).map(punct_norm)
  tc=Counter(t for s in txt for t in tokens(s));cc=Counter(s[i:i+3] for s in txt for i in range(max(0,len(s)-2)))
  tok_total=sum(tc.values());char_total=sum(cc.values())
  vocab=set(ref_tokens)|set(tc);p=np.array([ref_tokens[x] for x in vocab],float);q=np.array([tc[x] for x in vocab],float)
  shift.append({'source':src,'country':country,'sample_records':len(te),'token_occurrences':tok_total,'token_oov_occurrences':sum(v for x,v in tc.items()if x not in ref_tokens),'token_oov_occurrence_rate':sum(v for x,v in tc.items()if x not in ref_tokens)/tok_total,'token_types':len(tc),'token_oov_types':sum(x not in ref_tokens for x in tc),'token_oov_type_rate':sum(x not in ref_tokens for x in tc)/len(tc),'char3_oov_occurrence_rate':sum(v for x,v in cc.items()if x not in ref_chars)/char_total,'token_js_distance':jensenshannon(p/p.sum(),q/q.sum(),base=2)})
pd.DataFrame(shift).to_csv(OUT/'train_test_shift.csv',index=False)
(OUT/'shift_sampling.json').write_text(json.dumps({'seed':SEED,'train_reference':'100,000 hash-selected records per train source, pooled','test':'up to 30,000 hash-selected records per source-country','runtime_seconds':time.time()-start},indent=2))
print(pd.DataFrame(shift).to_string(index=False))
