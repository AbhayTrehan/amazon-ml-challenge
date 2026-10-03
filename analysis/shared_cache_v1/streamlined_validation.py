#!/usr/bin/env python3
"""Generate validation BASE49, score with frozen E008 LightGBM, freeze shortlist."""
from __future__ import annotations
import json,sys,time
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
SHARED=ROOT/"analysis/shared_cache_v1";OUT=SHARED/"reports/streamlined_final"
sys.path.insert(0,str(SHARED));import run_f20_r3_lgbm as m  # noqa: E402

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    pop="clean_validation";info={"frame":pd.read_parquet(SHARED/f"manifests/b2_{pop}_s1.parquet"),"path":SHARED/f"manifests/b2_{pop}_s1.parquet","manifest":json.loads((SHARED/f"manifests/b2_{pop}_s1.manifest.json").read_text())}
    cbase=SHARED/"candidates/f5f38769a14fb0052f186d8508a1e819a45fb3dbfc84f1e91b42e1fc922058de";cm=json.loads((cbase/"manifest.json").read_text());schema=json.loads((SHARED/"manifests/f20_r3_feature_schema.json").read_text())
    m.REPORT=OUT;m.SCORE_DIR=OUT/"scores";m.MODEL_DIR=m.E008/"model/lightgbm";m.ALL_FEATURES=m.BASE49
    started=time.perf_counter();fbase,fm=m.feature_cache(pop,info,cbase,cm,schema,base49_only=True)
    scores=m.score_features(pop,fbase);result=m.evaluate_shortlists(info,scores)
    payload={"state":"COMPLETE","feature_key":fm["content_key"],"feature_rows":fm["metadata"]["row_count"],"feature_pairs_per_second":fm["metadata"]["pairs_per_second"],"shortlist":result,"seconds":time.perf_counter()-started}
    m.atomic_json(OUT/"validation_existing_lgbm.json",payload);print(json.dumps(payload,indent=2),flush=True)
    if result["freeze_k"] is None:raise SystemExit(3)

if __name__=="__main__":main()
