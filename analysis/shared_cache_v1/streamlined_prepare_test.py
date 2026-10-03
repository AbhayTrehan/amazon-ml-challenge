#!/usr/bin/env python3
"""Prepare only the frozen test query/aux/owner index references for GPU reverse."""
from __future__ import annotations
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];SHARED=ROOT/"analysis/shared_cache_v1";OUT=SHARED/"reports/streamlined_final"
sys.path.insert(0,str(SHARED));import streamlined_gpu_reverse as g  # noqa:E402
from cache_utils import atomic_json  # noqa:E402
def main():
    t=time.perf_counter();info=g.load_population("test");f,r=g.indexes("test",info);result={"state":"COMPLETE","forward_keys":f,"reverse_keys":r,"seconds":time.perf_counter()-t};atomic_json(OUT/"test_indexes.json",result);print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__":main()
