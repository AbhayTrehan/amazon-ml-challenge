#!/usr/bin/env python3
"""Frozen epoch-2 XLM-R inference for D002; RTX A6000 must be masked externally."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"analysis/experiments_v1/d002_clean_public_gap"
P002=ROOT/"analysis/production/p002_final"
E005=ROOT/"analysis/experiments_v1/e005_xlmr_crossencoder"
sys.path.insert(0,str(P002/"src"));import xlmr as frozen
sys.path.insert(0,str(E005/"src"));from common import serialize

def atomic_json(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+".tmp");q.write_text(json.dumps(v,indent=2,default=float)+"\n");q.replace(p)
def atomic_parquet(d,p):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+".tmp");d.to_parquet(q,index=False,compression="zstd");q.replace(p)

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--batch",type=int,default=512);a=ap.parse_args()
 if torch.cuda.get_device_name(0)!="NVIDIA RTX A6000":raise RuntimeError(f"wrong GPU: {torch.cuda.get_device_name(0)}")
 tokenizer,model=frozen.load_model();start_all=time.time();rows=0
 for ip in sorted((OUT/"cache/shortlist").glob("*/*.parquet")):
  op=OUT/"cache/neural"/ip.parent.name/ip.name;mp=op.with_suffix(".runtime.json")
  if op.exists() and mp.exists():rows+=len(pd.read_parquet(op,columns=["source1_entity_id"]));continue
  frame=pd.read_parquet(ip);dataset,token_s=frozen.tokenize(frame,tokenizer);probs,meta=frozen.score_dataset(dataset,tokenizer,model,a.batch)
  out=frame[["source1_entity_id","candidate_entity_id","candidate_source"]].copy();out["neural_probability"]=probs;out["encoded_token_length"]=dataset.lengths;atomic_parquet(out,op)
  meta.update({"input":str(ip),"output":str(op),"tokenization_seconds":token_s,"gpu":torch.cuda.get_device_name(0)});atomic_json(mp,meta);rows+=len(out)
  with (OUT/"logs/xlmr_a6000.jsonl").open("a") as f:f.write(json.dumps(meta)+"\n")
  print("XLMR_READY",json.dumps(meta),flush=True)
 atomic_json(OUT/"runtime/xlmr.json",{"seconds":time.time()-start_all,"rows":rows,"gpu":torch.cuda.get_device_name(0),"status":"complete"})
if __name__=="__main__":main()
