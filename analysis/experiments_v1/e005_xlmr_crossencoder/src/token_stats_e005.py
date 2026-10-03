#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import model_info
from transformers import AutoTokenizer

from common import MODEL_ID, SEED, serialize, shortlist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sample-size", type=int, default=100000)
    args = ap.parse_args()
    data_dir, out = Path(args.data_dir), Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for name in ["train_pairs.parquet", "calibration_pairs.parquet"]:
        d = pd.read_parquet(data_dir / name, columns=["s1_business_name", "s1_business_address",
            "candidate_business_name", "candidate_business_address", "lightgbm_rank_within_s1", "lightgbm_score"])
        if name.startswith("calibration"):
            d = shortlist(d)
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    n = min(args.sample_size, len(d))
    sample = d.sample(n=n, random_state=SEED).reset_index(drop=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    lengths = []
    batch = 1024
    for start in range(0, n, batch):
        x = sample.iloc[start:start+batch]
        a = [serialize(v, z) for v, z in zip(x.s1_business_name, x.s1_business_address)]
        b = [serialize(v, z) for v, z in zip(x.candidate_business_name, x.candidate_business_address)]
        enc = tok(a, b, truncation=False, padding=False, add_special_tokens=True)
        lengths.extend(map(len, enc["input_ids"]))
    arr = np.asarray(lengths)
    quantiles = {k: float(np.quantile(arr, q, method="higher")) for k, q in
                 [("p50", .5), ("p90", .9), ("p95", .95), ("p99", .99), ("p99_5", .995)]}
    # Multiples of eight retain tensor-core-friendly shapes while avoiding a needlessly
    # conservative power-of-two cap. Pick the smallest measured >=99% coverage.
    choices = [64, 80, 88, 96, 104, 112, 128, 160, 192, 256]
    max_length = next((v for v in choices if np.mean(arr <= v) >= .99), 256)
    try:
        revision = model_info(MODEL_ID).sha
    except Exception:
        revision = getattr(tok, "_commit_hash", None) or tok.init_kwargs.get("_commit_hash")
    result = {
        "model_id": MODEL_ID, "model_revision": revision, "sample_seed": SEED,
        "sample_rows": n, "quantiles": quantiles, "max": int(arr.max()),
        "chosen_max_length": max_length,
        "coverage_at_max_length": float(np.mean(arr <= max_length)),
        "truncation_rate": float(np.mean(arr > max_length)),
        "serialization": "sequence A/B each: name: {business_name}\\naddress: {address or [missing]}",
        "unicode_preserved": True,
    }
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
