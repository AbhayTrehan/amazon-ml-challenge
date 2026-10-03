#!/usr/bin/env python3
from pathlib import Path
import json
import pandas as pd,matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];A=ROOT/'analysis/data_audit_v1/artifacts';P=ROOT/'analysis/data_audit_v1/plots';P.mkdir(exist_ok=True)
u=pd.read_csv(A/'blocking_union_benchmarks.csv')
fig,ax=plt.subplots(figsize=(7,4.5))
for m in ['name','address','name_address','all_union']:
 d=u[u.method==m];ax.plot(d.mean_candidates,d.oracle_macro_f05,marker='o',label=m)
ax.set(xlabel='Mean candidates / S1 (both auxiliary sources)',ylabel='Oracle macro F0.5',title='Candidate frontier (2,000-query deterministic sample)');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(P/'blocking_frontier.png',dpi=160);plt.close(fig)
m=json.loads((A/'metric_forensics.json').read_text());c=pd.DataFrame(m['cardinality_contribution'])
fig,ax=plt.subplots(figsize=(7,4));ax.bar(c.n.astype(str),c.entities);ax.set(xlabel='True matches per S1',ylabel='S1 entities',title='Training match-cardinality distribution');fig.tight_layout();fig.savefig(P/'cardinality_distribution.png',dpi=160);plt.close(fig)
