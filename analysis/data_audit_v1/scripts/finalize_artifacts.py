#!/usr/bin/env python3
"""Small derived artifacts and complete supplied-resource inventory."""
import csv,json,platform,sys
from pathlib import Path
import duckdb,pandas as pd
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'analysis/data_audit_v1';OUT=BASE/'artifacts';con=duckdb.connect(str(OUT/'audit.duckdb'),read_only=True)
con.execute(f"COPY (SELECT s1,target_id,target_source FROM edges ORDER BY s1,target_source,target_id) TO '{OUT/'positive_edges.parquet'}' (FORMAT PARQUET, COMPRESSION ZSTD)")
phys=pd.read_csv(OUT/'physical_integrity.csv'); purpose={
'train_source1.tsv':'deduplicated training reference records','train_source2.tsv':'training auxiliary Source 2 records','train_source3.tsv':'training auxiliary Source 3 records','train_ground_truth.tsv':'one label row per training S1','test_source1.tsv':'test reference/query records','test_source2.tsv':'test auxiliary Source 2 records','test_source3.tsv':'test auxiliary Source 3 records'}
inv=[]
for r in phys.itertuples():
 inv.append({'path':r.path,'bytes':r.bytes,'physical_lines':r.physical_lines,'parsed_rows':r.data_rows,'purpose':purpose[Path(r.path).name],'encoding':'UTF-8 (strict parse succeeded; no BOM)','columns':'source1_entity_id|matched_entity_ids'if'ground_truth'in r.path else'entity_id|business_name|business_address|country','inferred_dtypes':'all supplied columns are strings; row_num is audit-generated BIGINT'})
for rel,purp in [('student_resource/README.md','official problem statement and submission rules'),('student_resource/utils/validate_submission.py','official stdlib submission validator'),('student_resource/Documentation_template.md','required methodology template'),('student_resource/.DS_Store','macOS metadata; no challenge semantics'),('__MACOSX/','archive metadata directory; no challenge semantics')]:
 p=ROOT/rel;inv.append({'path':rel,'bytes':p.stat().st_size if p.is_file()else sum(x.stat().st_size for x in p.rglob('*')if x.is_file()),'physical_lines':sum(1 for _ in p.open(errors='replace'))if p.is_file()else None,'parsed_rows':None,'purpose':purp,'encoding':'text/binary metadata as applicable','columns':'','inferred_dtypes':''})
pd.DataFrame(inv).to_csv(OUT/'dataset_inventory.csv',index=False);(OUT/'dataset_inventory.json').write_text(json.dumps(inv,indent=2))
prof=pd.read_csv(OUT/'schema_and_missingness.csv');prof.assign(country_pct=prof.n/prof.groupby(['split','source']).n.transform('sum')).to_csv(OUT/'country_distribution.csv',index=False)
g=json.loads((OUT/'ground_truth_graph.json').read_text());pd.DataFrame(g['inverse']).to_csv(OUT/'inverse_mapping_stats.csv',index=False)
sim=pd.read_csv(OUT/'similarity_distributions.csv');sim[sim.family=='positive'].to_csv(OUT/'similarity_positive_stats.csv',index=False);sim[sim.family!='positive'].to_csv(OUT/'similarity_negative_stats.csv',index=False)
# Merge benchmark summaries while preserving sampling labels.
b=[]
exact_path=OUT/'exact_blocking_benchmarks.csv'
if exact_path.exists(): ex=pd.read_csv(exact_path)
else:
 ex=pd.read_csv(OUT/'blocking_benchmarks.csv');ex.to_csv(exact_path,index=False)
ex['evaluation']='full training';b.append(ex)
for f in OUT.glob('blocking_*_metrics.csv'):
 d=pd.read_csv(f);d['method']=d.channel+'_'+d.source.astype(str)+'_'+d.country;d['evaluation']='deterministic query sample/full auxiliary index';b.append(d)
u=pd.read_csv(OUT/'blocking_union_benchmarks.csv');u['evaluation']='common deterministic 2,000-S1 sample/full auxiliary indexes';b.append(u)
allb=pd.concat(b,ignore_index=True,sort=False);allb.to_csv(OUT/'blocking_benchmarks_all.csv',index=False);allb.to_csv(OUT/'blocking_benchmarks.csv',index=False)
# Validation balance.
v=pd.read_csv(OUT/'proposed_validation_split.tsv',sep='\t');v['cardinality_bucket']=v.true_matches.clip(upper=5).astype(str).replace({'5':'5+'});v.groupby(['fold','country','cardinality_bucket']).size().rename('entities').reset_index().to_csv(OUT/'validation_split_balance.csv',index=False)
print('inventory rows',len(inv),'artifact files',len(list(OUT.iterdir())))
