#!/usr/bin/env python3
"""Minimal durable controller for the development-first streamlined path."""
from __future__ import annotations
import json,os,subprocess,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/"analysis/shared_cache_v1/reports/streamlined_final";PY="/home/satvik/venvs/amazonml2026/bin/python"

def log(event,**kw):print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}] {event} "+json.dumps(kw,sort_keys=True),flush=True)

def main():
    while not (OUT/"validation_existing_lgbm.json").exists():time.sleep(30)
    val=json.loads((OUT/"validation_existing_lgbm.json").read_text())
    if val["shortlist"]["freeze_k"] is None:log("STOP_EXISTING_LGBM_GATE_FAILED");raise SystemExit(3)
    log("VALIDATION_GATE_PASS",freeze_k=val["shortlist"]["freeze_k"],oracle=val["shortlist"]["freeze_oracle"])
    subprocess.run([PY,str(ROOT/"analysis/shared_cache_v1/streamlined_prepare_test.py")],cwd=ROOT,check=True)
    log("TEST_INDEXES_READY")
    while True:
        done=[]
        for p in OUT.glob("gpu_worker_development_train_*.json"):
            try:
                if json.loads(p.read_text()).get("state")=="COMPLETE":done.append(str(p))
            except Exception:pass
        if len(done)>=2:break
        time.sleep(30)
    log("DEV_GPU_RETRIEVAL_COMPLETE",workers=len(done))
    # Development has absolute priority. Only now reuse both free GPUs for
    # independent test reverse streams while CPU development work proceeds.
    jobs=[]
    assignments=[("test_gpu_a6000","GPU-be9e124c-5a77-cc0f-586d-9bd963e58e98","France:2:name,France:2:address,India:2:name,India:2:address,US:2:name,US:2:address"),("test_gpu_ada","GPU-f5f574b3-d0dc-fd72-5fd4-2d913d115b17","France:3:name,France:3:address,India:3:name,India:3:address,US:3:name,US:3:address")]
    for name,uuid,streams in assignments:
        handle=(OUT/f"{name}.log").open("ab");env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=uuid
        jobs.append(subprocess.Popen([PY,str(ROOT/"analysis/shared_cache_v1/streamlined_gpu_reverse.py"),"worker","--population","test","--streams",streams,"--batch","1024"],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True))
    log("TEST_GPU_RETRIEVAL_STARTED",pids=[x.pid for x in jobs])
    subprocess.run([PY,str(ROOT/"analysis/shared_cache_v1/streamlined_finalize_dev.py")],cwd=ROOT,check=True)
    log("DEV_CANDIDATES_COMPLETE")
    subprocess.run([PY,str(ROOT/"analysis/shared_cache_v1/streamlined_dev_modeling.py")],cwd=ROOT,check=True)
    log("DEV_MODELING_COMPLETE")

if __name__=="__main__":main()
