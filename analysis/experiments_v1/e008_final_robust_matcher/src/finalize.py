#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import duckdb
import pandas as pd

from common import *


def run():
    gate=json.loads((EXP/"validation/production_gate.json").read_text())
    if not gate["pass"]:raise RuntimeError("P003 finalization refused because validation gate failed")
    con=duckdb.connect();con.execute("PRAGMA threads=20");eg=str(P003/"cache/scored/*/*.parquet");ps=str(P002/"cache/shortlist/*/*.parquet");pn=str(P002/"cache/neural/*/*.parquet")
    con.execute(f"create temp view e as select * from read_parquet('{eg}',union_by_name=true)")
    con.execute(f"create temp view p as select s.*,(.6*n.neural_probability+.4*s.lightgbm_score)::float pair_probability from read_parquet('{ps}',union_by_name=true)s join read_parquet('{pn}',union_by_name=true)n using(source1_entity_id,candidate_entity_id,candidate_source)")
    rows=[]
    country_s1={c:len(pd.read_parquet(P001/f"cache/indexes/{c}/queries.parquet",columns=["entity_id"])) for c in ["France","India","US"]}
    for model,view,acc in [("P002","p",str(P002/"output/accepted_pairs.parquet")),("E008","e",str(P003/"final/accepted_pairs.parquet"))]:
        base=con.execute(f"""select country,count(*) pairs,count(distinct source1_entity_id)s1,count(*)::double/count(distinct source1_entity_id) shortlist_per_s1,
          quantile_cont(pair_probability,[.01,.1,.5,.9,.99]) score_percentiles from {view} group by country order by country""").fetchdf()
        match=con.execute(f"select country,count(*)::double/count(distinct source1_entity_id) selected_per_nonempty_s1,count(*) edges,count(distinct source1_entity_id) nonempty_s1 from read_parquet('{acc}')group by country").fetchdf()
        amb=con.execute(f"""with z as(select candidate_entity_id,country,count(distinct source1_entity_id)n from {view} group by all)select country,count(*)filter(where n>1)contested,count(*) auxiliaries,count(*)filter(where n>1)::double/count(*) contested_fraction from z group by country""").fetchdf()
        for _,r in base.merge(match,on="country",how="left").merge(amb,on="country",how="left").iterrows():
            z={"model":model,**r.to_dict()};z["selected_matches_per_s1"]=z["edges"]/country_s1[z["country"]];rows.append(z)
    diag=pd.DataFrame(rows);diag.to_json(P003/"diagnostics/test_shift.json",orient="records",indent=2)
    accepted=pd.read_parquet(P003/"final/accepted_pairs.parquet");source=accepted.groupby(["country","candidate_source"]).size().reset_index(name="predicted_edges")
    dec=json.loads((P003/"diagnostics/decode.json").read_text());val=json.loads((P003/"diagnostics/validation.json").read_text());train=json.loads((EXP/"data/training_report.json").read_text());freeze=json.loads((EXP/"freeze/frozen_config.json").read_text());vres=json.loads((EXP/"validation/results.json").read_text())
    runtime={}
    for p in list((EXP/"runtime").glob("*.json"))+list((P003/"runtime").glob("*.json")):
        try:runtime[str(p.relative_to(ROOT))]=json.loads(p.read_text())
        except Exception:pass
    manifest={"status":"READY_TO_SUBMIT_P003","created_utc":pd.Timestamp.utcnow().isoformat(),"training":train,"freeze":freeze,"validation":vres,"production_gate":gate,
      "production":{"test_s1":1732544,"candidate_pairs":dec["shortlist_pairs"],"predicted_edges":dec["predicted_edges"],"predicted_singletons":dec["predicted_singletons"],"country_source_counts":source.to_dict("records")},
      "outputs":{"matching_results.tsv":{"path":str((P003/"final/matching_results.tsv").resolve()),"sha256":sha256(P003/"final/matching_results.tsv")},"candidate_pairs.tsv":{"path":str((P003/"final/candidate_pairs.tsv").resolve()),"sha256":sha256(P003/"final/candidate_pairs.tsv")}},
      "official_validator":val,"test_shift_diagnostic_only":diag.to_dict("records"),"runtime":runtime,"environment":{"host":platform.node(),"python":platform.python_version()}}
    atomic_json(P003/"production_manifest.json",manifest);(P003/"runtime/COMPLETE").write_text(pd.Timestamp.utcnow().isoformat()+"\n");print(json.dumps(manifest,indent=2,default=float))
if __name__=="__main__":run()
