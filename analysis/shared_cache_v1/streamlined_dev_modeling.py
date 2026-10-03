#!/usr/bin/env python3
"""BASE49 dev features, existing-model hard negatives, and XLM-R manifest."""
from __future__ import annotations
import json,sys,time
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[2];SHARED=ROOT/"analysis/shared_cache_v1";OUT=SHARED/"reports/streamlined_final"
sys.path.insert(0,str(SHARED));import run_f20_r3_lgbm as m  # noqa:E402
from cache_utils import atomic_json  # noqa:E402

def main():
    val=json.loads((OUT/"validation_existing_lgbm.json").read_text())
    if val["shortlist"]["freeze_k"] is None:raise SystemExit("existing LightGBM failed shortlist gate")
    dev=json.loads((OUT/"development_candidates.json").read_text());cbase=Path(dev["candidate_path"]);cm=json.loads((cbase/"manifest.json").read_text());schema=json.loads((SHARED/"manifests/f20_r3_feature_schema.json").read_text())
    pop="development_train";info={"frame":pd.read_parquet(SHARED/f"manifests/b2_{pop}_s1.parquet"),"path":SHARED/f"manifests/b2_{pop}_s1.parquet","manifest":json.loads((SHARED/f"manifests/b2_{pop}_s1.manifest.json").read_text())}
    m.REPORT=OUT;m.SCORE_DIR=OUT/"scores";m.MODEL_DIR=m.E008/"model/lightgbm";m.ALL_FEATURES=m.BASE49;m.XLMR_DIR=OUT/"xlmr"
    started=time.perf_counter();fbase,fm=m.feature_cache(pop,info,cbase,cm,schema,base49_only=True);scores=m.score_features(pop,fbase);xstats=m.xlmr_manifest(info,scores)
    result={"state":"COMPLETE","shortlist_k":val["shortlist"]["freeze_k"],"feature_key":fm["content_key"],"feature_rows":fm["metadata"]["row_count"],"feature_pairs_per_second":fm["metadata"]["pairs_per_second"],"xlmr":xstats,"seconds":time.perf_counter()-started};atomic_json(OUT/"development_modeling.json",result);print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__":main()
