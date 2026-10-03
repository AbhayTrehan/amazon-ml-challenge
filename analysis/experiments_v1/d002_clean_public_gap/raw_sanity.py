#!/usr/bin/env python3
"""Cheap S1-level numeric/Unicode sanity checks omitted from pair diagnostics."""
from pathlib import Path
import re,unicodedata
import duckdb,numpy as np,pandas as pd

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"analysis/experiments_v1/d002_clean_public_gap"
DB=ROOT/"analysis/data_audit_v1/artifacts/audit.duckdb"
ids=pd.read_csv(OUT/"clean_eval_ids.tsv",sep="\t",usecols=["source1_entity_id"])
con=duckdb.connect(str(DB),read_only=True);con.register("clean_ids",ids)
rows=[]
for pop,table,country,join in [
 ("clean US","train_s1","US","join clean_ids i on i.source1_entity_id=s.entity_id"),
 ("clean India","train_s1","India","join clean_ids i on i.source1_entity_id=s.entity_id"),
 ("test US","test_s1","US",""),("test India","test_s1","India",""),("test France","test_s1","France","")]:
 q=f"""select count(*)n,avg(length(coalesce(business_name,'')))name_len,
 avg(length(coalesce(business_address,'')))address_len,
 avg((coalesce(business_address,'')='')::int)missing_address,
 avg(regexp_matches(coalesce(business_address,''),'[0-9]')::int)numeric_address,
 avg(regexp_matches(coalesce(business_name,'')||coalesce(business_address,''),'[[:cntrl:]]')::int)control_chars
 from {table} s {join} where s.country=?"""
 r=con.execute(q,[country]).fetchdf().iloc[0].to_dict();r.update({"population":pop});rows.append(r)
 # Deterministic raw-string sample for NFKC/non-ASCII incidence.
 d=con.execute(f"select * from (select business_name,business_address from {table} s {join} where s.country=?) using sample 100000 rows(reservoir,20260925)",[country]).fetchdf()
 text=(d.business_name.fillna("")+" "+d.business_address.fillna("")).tolist();r["unicode_sample_n"]=len(text);r["non_ascii_fraction"]=np.mean([any(ord(c)>127 for c in x) for x in text]);r["nfkc_changes_fraction"]=np.mean([unicodedata.normalize("NFKC",x)!=x for x in text])
con.close();pd.DataFrame(rows).to_csv(OUT/"output/raw_input_sanity.csv",index=False)
