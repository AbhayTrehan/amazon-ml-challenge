#!/usr/bin/env python3
"""Score an already-materialized pair frame without changing its membership."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT=Path(__file__).resolve().parents[3]
E005=ROOT/"analysis/experiments_v1/e005_xlmr_crossencoder"
sys.path.insert(0,str(E005/"src"))
from common import TokenizedCollator, pretokenize, seed_everything  # noqa: E402

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",required=True); ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--output",required=True); ap.add_argument("--batch-size",type=int,default=512); ap.add_argument("--workers",type=int,default=2); a=ap.parse_args()
    seed_everything(); d=pd.read_parquet(a.input); tok=AutoTokenizer.from_pretrained(a.checkpoint,use_fast=True)
    model=AutoModelForSequenceClassification.from_pretrained(a.checkpoint,use_safetensors=True,attn_implementation="sdpa").cuda().eval()
    t0=time.time(); ds=pretokenize(d,tok,104); loader=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0,collate_fn=TokenizedCollator(tok))
    probs=[]; dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16; torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for batch in loader:
            batch.pop("labels",None); batch={k:v.cuda(non_blocking=True) for k,v in batch.items()}
            with torch.autocast("cuda",dtype=dtype): z=model(**batch).logits
            probs.append(torch.softmax(z.float(),dim=1)[:,1].cpu().numpy())
    torch.cuda.synchronize(); d["neural_probability"]=np.concatenate(probs).astype(np.float32)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); d.to_parquet(out,index=False,compression="zstd")
    meta={"rows":len(d),"seconds":time.time()-t0,"pairs_per_second":len(d)/(time.time()-t0),"gpu":torch.cuda.get_device_name(0),"precision":str(dtype),"batch_size":a.batch_size,"peak_allocated_gb":torch.cuda.max_memory_allocated()/2**30}
    out.with_suffix(".runtime.json").write_text(json.dumps(meta,indent=2)+"\n"); print(json.dumps(meta,indent=2))
if __name__=="__main__": main()
