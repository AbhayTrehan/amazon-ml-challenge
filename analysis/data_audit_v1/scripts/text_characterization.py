#!/usr/bin/env python3
"""Exact text-shape summaries and frequent/collision-prone values."""
from pathlib import Path
import duckdb,pandas as pd
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'analysis/data_audit_v1/artifacts';con=duckdb.connect(str(OUT/'audit.duckdb'),read_only=True);con.execute("PRAGMA threads=6");con.execute("PRAGMA memory_limit='2GB'")
tabs=['train_s1','train_s2','train_s3','test_s1','test_s2','test_s3'];records=[]
for t in tabs:
 split,source=t.split('_')
 for field,label in [('business_name','name'),('business_address','address')]:
  q=f"""WITH x AS(SELECT country,array_length(regexp_extract_all(coalesce({field},''),'[^ ]+')) tok,length(regexp_replace(coalesce({field},''),'[^0-9]','','g'))dig FROM {t})
  SELECT country,count(*)n,min(tok)token_min,quantile_disc(tok,.01)token_p1,quantile_disc(tok,.05)token_p5,quantile_disc(tok,.5)token_p50,avg(tok)token_mean,quantile_disc(tok,.95)token_p95,quantile_disc(tok,.99)token_p99,max(tok)token_max,
  min(dig)digit_min,quantile_disc(dig,.01)digit_p1,quantile_disc(dig,.05)digit_p5,quantile_disc(dig,.5)digit_p50,avg(dig)digit_mean,quantile_disc(dig,.95)digit_p95,quantile_disc(dig,.99)digit_p99,max(dig)digit_max FROM x GROUP BY country"""
  d=con.execute(q).fetchdf();d['split']=split;d['source']=source.upper();d['field']=label;records.append(d)
pd.concat(records).to_csv(OUT/'text_shape_stats.csv',index=False)

tops=[]
for t in tabs:
 for field in ['business_name','business_address','nname','naddr']:
  d=con.execute(f"select * exclude(rn) from(select '{t}' AS table_name,'{field}' AS field,country,{field} AS value,count(*) AS frequency,row_number()over(partition by country order by count(*)desc)rn from {t} where coalesce({field},'')<>'' group by country,{field})where rn<=30").fetchdf();tops.append(d)
pd.concat(tops).to_csv(OUT/'frequent_values.csv',index=False)

suffix=[]
for t in tabs:
 d=con.execute(f"""select '{t}' AS table_name,country,regexp_extract(nname,'([^ ]+)$',1) AS suffix,count(*) AS frequency from {t} where nname<>'' group by country,suffix qualify row_number()over(partition by country order by frequency desc)<=30 order by country,frequency desc""").fetchdf();suffix.append(d)
pd.concat(suffix).to_csv(OUT/'name_suffix_frequencies.csv',index=False)
print('wrote text characterization')
