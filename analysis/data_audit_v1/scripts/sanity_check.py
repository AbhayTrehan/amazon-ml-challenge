#!/usr/bin/env python3
"""Cross-artifact invariants for the completed audit."""
import json
from pathlib import Path
import duckdb,pandas as pd
ROOT=Path(__file__).resolve().parents[3];A=ROOT/'analysis/data_audit_v1/artifacts';c=duckdb.connect(str(A/'audit.duckdb'),read_only=True)
assert c.execute('select count(*)from train_s1').fetchone()[0]==2_206_821
assert c.execute('select count(*)from edges').fetchone()[0]==7_638_365
assert c.execute('select count(*)from edges group by target_id having count(*)>1 limit 1').fetchone() is None
assert c.execute("select count(*)from train_s1 s left join train_gt g on s.entity_id=g.source1_entity_id where g.source1_entity_id is null").fetchone()[0]==0
assert c.execute("select count(*)from edges e join train_s1 s on e.s1=s.entity_id join(select entity_id,country from train_s2 union all select entity_id,country from train_s3)a on e.target_id=a.entity_id where s.country<>a.country").fetchone()[0]==0
m=json.loads((A/'metric_forensics.json').read_text());assert all(abs(x['score']-x['expected'])<1e-12 for x in m['unit_tests'])
u=pd.read_csv(A/'blocking_union_benchmarks.csv')
for method,g in u.groupby('method'):assert g.sort_values('k').candidate_recall.is_monotonic_increasing
assert len(pd.read_csv(A/'proposed_validation_split.tsv',sep='\t'))==2_206_821
assert len(pd.read_parquet(A/'positive_edges.parquet'))==7_638_365
for f in ['REPORT.md','HANDOFF_TO_CHATGPT.md']:
 assert (A.parent/f).stat().st_size>10_000
print('PASS: graph, metric, retrieval monotonicity, fold coverage, edge export, and report checks')
