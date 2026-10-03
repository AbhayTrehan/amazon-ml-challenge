#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import sys
ROOT=Path(__file__).resolve().parents[4]
E005=ROOT/"analysis/experiments_v1/e005_xlmr_crossencoder"
sys.path.insert(0,str(E005/"src"))
from common import TokenizedCollator, pretokenize, seed_everything, shortlist

EXP=ROOT/"analysis/production/p001_e005b"
CHECKPOINT=E005/"model/best_checkpoint"


def score_frame(d,tok,model,batch_size,workers):
    token_start=time.perf_counter();ds=pretokenize(d,tok,104);token_seconds=time.perf_counter()-token_start
    loader=DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=workers,pin_memory=True,
      persistent_workers=workers>0,collate_fn=TokenizedCollator(tok))
    probs=[];torch.cuda.reset_peak_memory_stats();start=time.perf_counter();dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.inference_mode():
        for batch in loader:
            batch.pop("labels",None);batch={k:v.cuda(non_blocking=True) for k,v in batch.items()}
            with torch.autocast("cuda",dtype=dtype):logits=model(**batch).logits
            probs.append(torch.softmax(logits.float(),dim=1)[:,1].cpu().numpy())
    torch.cuda.synchronize();gpu_seconds=time.perf_counter()-start
    return np.concatenate(probs).astype(np.float32),{"rows":len(d),"tokenization_seconds":token_seconds,"gpu_seconds":gpu_seconds,
      "total_seconds":token_seconds+gpu_seconds,"pairs_per_second":len(d)/(token_seconds+gpu_seconds),"gpu_pairs_per_second":len(d)/gpu_seconds,
      "peak_allocated_gpu_gb":torch.cuda.max_memory_allocated()/2**30,"peak_reserved_gpu_gb":torch.cuda.max_memory_reserved()/2**30,
      "batch_size":batch_size,"precision":"bf16" if dtype==torch.bfloat16 else "fp16"}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--input-file");ap.add_argument("--input-root");ap.add_argument("--output-file");ap.add_argument("--output-root")
    ap.add_argument("--apply-shortlist",action="store_true");ap.add_argument("--batch-size",type=int,default=256);ap.add_argument("--workers",type=int,default=4)
    ap.add_argument("--shard-mod",type=int,default=1);ap.add_argument("--shard-rem",type=int,default=0);ap.add_argument("--worker-name",default="single")
    args=ap.parse_args();seed_everything()
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    tok=AutoTokenizer.from_pretrained(CHECKPOINT,use_fast=True,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(CHECKPOINT,use_safetensors=True,local_files_only=True,attn_implementation="sdpa").cuda().eval()
    if args.input_file:
        items=[(Path(args.input_file),Path(args.output_file))]
    else:
        ir=Path(args.input_root);orr=Path(args.output_root);items=[]
        for j,p in enumerate(sorted(ir.glob("*/*.parquet"))):
            if j % args.shard_mod == args.shard_rem:items.append((p,orr/p.parent.name/p.name))
    total={"rows":0,"tokenization_seconds":0.,"gpu_seconds":0.,"total_seconds":0.,"parts":0,"gpu":torch.cuda.get_device_name(0),"batch_size":args.batch_size}
    log=EXP/"logs/neural_parts.jsonl";log.parent.mkdir(parents=True,exist_ok=True)
    for ip,op in items:
        if op.exists():continue
        d=pd.read_parquet(ip);d=shortlist(d) if args.apply_shortlist else d
        # Production files are unlabeled; pretokenize expects a label column but never uses it for inference.
        if "label" not in d:d["label"]=np.int8(0)
        probs,meta=score_frame(d,tok,model,args.batch_size,args.workers);op.parent.mkdir(parents=True,exist_ok=True)
        pd.DataFrame({"source1_entity_id":d.source1_entity_id,"candidate_entity_id":d.candidate_entity_id,"neural_probability":probs}).to_parquet(op,index=False,compression="zstd")
        meta.update({"input":str(ip),"output":str(op)});op.with_suffix(".runtime.json").write_text(json.dumps(meta,indent=2)+"\n")
        with log.open("a") as f:f.write(json.dumps(meta)+"\n")
        for k in ["rows","tokenization_seconds","gpu_seconds","total_seconds"]:total[k]+=meta[k]
        total["parts"]+=1;total["peak_allocated_gpu_gb"]=max(total.get("peak_allocated_gpu_gb",0),meta["peak_allocated_gpu_gb"])
        print("NEURAL",json.dumps(meta),flush=True)
    if args.input_root:
        saved=[]
        for p in Path(args.output_root).glob("*/*.runtime.json"):
            saved.append(json.loads(p.read_text()))
        total={"rows":sum(x["rows"] for x in saved),"tokenization_seconds":sum(x["tokenization_seconds"] for x in saved),
          "gpu_seconds":sum(x["gpu_seconds"] for x in saved),"total_seconds":sum(x["total_seconds"] for x in saved),"parts":len(saved),
          "gpu":torch.cuda.get_device_name(0),"batch_size":args.batch_size,"peak_allocated_gpu_gb":max((x["peak_allocated_gpu_gb"] for x in saved),default=0)}
    total["pairs_per_second"]=total["rows"]/total["total_seconds"] if total["total_seconds"] else None
    total["gpu_pairs_per_second"]=total["rows"]/total["gpu_seconds"] if total["gpu_seconds"] else None
    (EXP/f"diagnostics/neural_inference_{args.worker_name}.json").write_text(json.dumps(total,indent=2)+"\n");print(json.dumps(total,indent=2))


if __name__=="__main__":main()
