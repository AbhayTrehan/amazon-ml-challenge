#!/usr/bin/env python3
"""Create a reusable DuckDB catalog over supplied TSVs (no source mutation)."""
import duckdb, json, os, platform, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'analysis/data_audit_v1/artifacts'; OUT.mkdir(parents=True,exist_ok=True)
DB=OUT/'audit.duckdb'
con=duckdb.connect(str(DB)); con.execute("PRAGMA threads=6"); con.execute("PRAGMA memory_limit='3GB'"); con.execute(f"PRAGMA temp_directory='{OUT/'ducktmp'}'")
files={
 'train_s1':'student_resource/dataset/train/train_source1.tsv',
 'train_s2':'student_resource/dataset/train/train_source2.tsv',
 'train_s3':'student_resource/dataset/train/train_source3.tsv',
 'test_s1':'student_resource/dataset/test/test_source1.tsv',
 'test_s2':'student_resource/dataset/test/test_source2.tsv',
 'test_s3':'student_resource/dataset/test/test_source3.tsv',
 'train_gt':'student_resource/dataset/train/train_ground_truth.tsv'}
start=time.time(); timings={}
for table,rel in files.items():
    t=time.time(); path=ROOT/rel
    cols="{'source1_entity_id':'VARCHAR','matched_entity_ids':'VARCHAR'}" if table=='train_gt' else "{'entity_id':'VARCHAR','business_name':'VARCHAR','business_address':'VARCHAR','country':'VARCHAR'}"
    con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT *, row_number() OVER () AS row_num FROM read_csv(?, delim='\\t', header=true, columns={cols}, quote='', null_padding=false, ignore_errors=false)",[str(path)])
    timings[table]=time.time()-t; print(table,con.execute(f'SELECT count(*) FROM {table}').fetchone()[0],f'{timings[table]:.1f}s',flush=True)
for t in [x for x in files if x!='train_gt']:
    con.execute(f"ALTER TABLE {t} ADD COLUMN nname VARCHAR")
    con.execute(f"ALTER TABLE {t} ADD COLUMN naddr VARCHAR")
    # Unicode-preserving punctuation/whitespace normalization. NFKC/casefold variants are calculated in Python diagnostics.
    con.execute(f"UPDATE {t} SET nname=trim(regexp_replace(regexp_replace(lower(business_name), '[^\\p{{L}}\\p{{N}}]+', ' ', 'g'), '\\s+', ' ', 'g')), naddr=trim(regexp_replace(regexp_replace(lower(business_address), '[^\\p{{L}}\\p{{N}}]+', ' ', 'g'), '\\s+', ' ', 'g'))")
for t in ['train_s1','train_s2','train_s3']:
    con.execute(f"CREATE INDEX IF NOT EXISTS {t}_id_idx ON {t}(entity_id)")
meta={'python':sys.version,'platform':platform.platform(),'duckdb':duckdb.__version__,'threads':6,'memory_limit':'3GB','timings':timings,'total_seconds':time.time()-start}
(OUT/'environment_runtime.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2))
