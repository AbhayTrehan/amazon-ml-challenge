#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import TokenizedCollator, pretokenize, seed_everything, shortlist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True); ap.add_argument("--max-length", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=64); ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args(); seed_everything()
    d = shortlist(pd.read_parquet(args.input))
    tok = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint, use_safetensors=True, attn_implementation="sdpa"
    ).cuda().eval()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    token_start=time.perf_counter();dataset=pretokenize(d,tok,args.max_length);token_seconds=time.perf_counter()-token_start
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0, collate_fn=TokenizedCollator(tok))
    probs=[]; start=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for batch in loader:
            batch.pop("labels", None); batch={k:v.cuda(non_blocking=True) for k,v in batch.items()}
            with torch.autocast("cuda", dtype=amp_dtype): logits=model(**batch).logits
            probs.append(torch.softmax(logits.float(), dim=1)[:,1].cpu().numpy())
    torch.cuda.synchronize(); elapsed=time.perf_counter()-start
    d["neural_probability"] = np.concatenate(probs).astype(np.float32)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);d.to_parquet(out,index=False,compression="zstd")
    meta={"rows":len(d),"seconds":elapsed,"tokenization_seconds":token_seconds,"total_seconds":elapsed+token_seconds,
          "pairs_per_second":len(d)/(elapsed+token_seconds),"gpu_only_pairs_per_second":len(d)/elapsed,"batch_size":args.batch_size,
          "precision":"bf16" if amp_dtype == torch.bfloat16 else "fp16",
          "peak_allocated_gpu_gb":torch.cuda.max_memory_allocated()/2**30,
          "peak_reserved_gpu_gb":torch.cuda.max_memory_reserved()/2**30}
    out.with_suffix(".runtime.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps(meta,indent=2))


if __name__ == "__main__": main()
