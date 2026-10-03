#!/usr/bin/env python3
"""Publish the completed raw development retrieval and assemble final F20_R3."""
from __future__ import annotations
import json,sys,time
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[2];SHARED=ROOT/"analysis/shared_cache_v1";OUT=SHARED/"reports/streamlined_final"
sys.path.insert(0,str(SHARED));import run_f20_r3_lgbm as m  # noqa:E402
import streamlined_gpu_reverse as g  # noqa:E402
from cache_utils import atomic_json,complete_manifest  # noqa:E402

def main():
    pop="development_train";info=g.load_population(pop);fkeys,rkeys=g.indexes(pop,info);base=g.raw_base(pop,info,fkeys,rkeys)
    identity={"kind":"f20_r3_raw_retrieval","population_key":info["manifest"]["content_key"],"forward_index_keys":fkeys,"reverse_index_keys":rkeys,"definition":"forward K20 plus reverse top3; literal string country equality","forward":{"k":20,"posting_budget":20000,"coarse_m":800},"reverse":{"k":3,"method":"exact sparse_dot_topn","threshold":1e-8}}
    expected=[]
    for country in m.COUNTRIES[pop]:
        q=pd.read_parquet(m.b2.query_map(pop,country,info))
        for source in [2,3]:
            aux=m.b2.normalized_aux(pop,country,source,m.b2.forward_index(pop,country,source))
            for channel in ["name","address"]:
                expected += [base/"forward"/country/f"s{source}_{channel}_{p:05d}.parquet" for p,_ in enumerate(range(0,len(q),m.QUERY_SHARD))]
                expected += [base/"reverse"/country/f"s{source}_{channel}_{p:05d}.parquet" for p,_ in enumerate(range(0,len(aux),m.AUX_SHARD))]
    missing=[str(p) for p in expected if not(p.exists()and p.with_suffix(".meta.json").exists())]
    if missing:raise RuntimeError(f"development retrieval incomplete: {len(missing)} shards; first={missing[:3]}")
    metas=[json.loads(p.read_text()) for p in sorted(base.glob("forward/*/*.meta.json"))+sorted(base.glob("reverse/*/*.meta.json"))]
    runtime=base/"raw_runtime.json";atomic_json(runtime,{"stages":metas,"wall_seconds":sum(float(x.get("wall_seconds",0)) for x in metas),"note":"mixed exact CPU/GPU resumable shards; identical direct-top3 parity"})
    arts=sorted(base.glob("forward/*/*.parquet"))+sorted(base.glob("forward/*/*.meta.json"))+sorted(base.glob("reverse/*/*.parquet"))+sorted(base.glob("reverse/*/*.meta.json"))+[runtime]
    rm=complete_manifest(base/"raw.manifest.json",identity,arts,{"population":pop,"forward_shards":sum("forward" in str(p) and p.suffix==".parquet" for p in arts),"reverse_shards":sum("reverse" in str(p) and p.suffix==".parquet" for p in arts)})
    started=time.perf_counter();cbase,cm=m.assemble(pop,info,base,rm);result={"state":"COMPLETE","raw_key":rm["content_key"],"candidate_key":cm["content_key"],"candidate_path":str(cbase),"candidate_rows":cm["metadata"]["row_count"],"seconds":time.perf_counter()-started};atomic_json(OUT/"development_candidates.json",result);print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__":main()
