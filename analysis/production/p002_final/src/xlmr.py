#!/usr/bin/env python3
"""Frozen BF16 XLM-R inference, benchmarking, and deterministic shard worker."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "analysis/production/p002_final"
E005 = ROOT / "analysis/experiments_v1/e005_xlmr_crossencoder"
CHECKPOINT = E005 / "model/best_checkpoint"
sys.path.insert(0, str(E005 / "src"))
from common import TokenizedCollator, pretokenize, seed_everything  # noqa: E402
from evaluate_e005 import mask_for  # noqa: E402


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=float) + "\n")
    tmp.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)


def load_model():
    seed_everything()
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, use_fast=True, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, use_safetensors=True, local_files_only=True,
        attn_implementation="sdpa").cuda().eval()
    return tokenizer, model


def tokenize(frame: pd.DataFrame, tokenizer):
    d = frame if "label" in frame else frame.assign(label=np.int8(0))
    start = time.perf_counter(); dataset = pretokenize(d, tokenizer, 104)
    return dataset, time.perf_counter() - start


def score_dataset(dataset, tokenizer, model, batch_size: int, workers: int = 4):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0, collate_fn=TokenizedCollator(tokenizer))
    probs = []; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            batch.pop("labels", None)
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**batch).logits
            probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
    torch.cuda.synchronize(); elapsed = time.perf_counter() - start
    return np.concatenate(probs).astype(np.float32), {
        "rows": len(dataset), "gpu_seconds": elapsed, "gpu_pairs_per_second": len(dataset) / elapsed,
        "batch_size": batch_size, "precision": "bf16",
        "peak_allocated_gpu_gb": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gpu_gb": torch.cuda.max_memory_reserved() / 2**30,
    }


def accepted_keys(frame, probabilities):
    score = .60 * probabilities.astype(float) + .40 * frame.lightgbm_score.to_numpy(float)
    mask = mask_for(frame, score, {"decoder": "one_owner_margin", "threshold": .51, "margin": .48, "weight": .60})
    key = ["source1_entity_id", "candidate_entity_id", "candidate_source"]
    return set(map(tuple, frame.loc[mask, key].itertuples(index=False, name=None)))


def benchmark(path: Path, batches: list[int], output: Path):
    frame = pd.read_parquet(path); tokenizer, model = load_model(); dataset, token_seconds = tokenize(frame, tokenizer)
    rows = []; reference = reference_keys = None
    for batch in batches:
        try:
            probs, meta = score_dataset(dataset, tokenizer, model, batch)
            keys = accepted_keys(frame, probs)
            if reference is None: reference, reference_keys = probs, keys
            meta.update({"tokenization_seconds_once": token_seconds,
                "probability_max_abs_vs_batch256": float(np.max(np.abs(probs - reference))),
                "accepted_identity_vs_batch256": keys == reference_keys})
            np.save(output.parent / f"benchmark_probs_batch{batch}.npy", probs)
            rows.append(meta); print("BATCH", json.dumps(meta), flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); rows.append({"batch_size": batch, "oom": True}); print("OOM", batch, flush=True); break
    report = {"gpu": torch.cuda.get_device_name(0), "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
              "input": str(path), "rows": len(frame), "max_length": 104, "dynamic_padding": True,
              "precision": "bf16", "results": rows}
    atomic_json(output, report); print(json.dumps(report, indent=2))


def infer_file(ip: Path, op: Path, batch: int, worker_name: str):
    frame = pd.read_parquet(ip); tokenizer, model = load_model(); dataset, token_seconds = tokenize(frame, tokenizer)
    probs, meta = score_dataset(dataset, tokenizer, model, batch)
    key = ["source1_entity_id", "candidate_entity_id", "candidate_source"]
    out = frame[key].copy(); out["neural_probability"] = probs
    atomic_parquet(out, op); meta.update({"tokenization_seconds": token_seconds,
        "total_seconds": token_seconds + meta["gpu_seconds"], "input": str(ip), "output": str(op),
        "worker": worker_name, "gpu": torch.cuda.get_device_name(0)})
    atomic_json(op.with_suffix(".runtime.json"), meta); print(json.dumps(meta, indent=2))


def infer_root(input_root: Path, output_root: Path, batch: int, shard_mod: int, shard_rem: int, worker_name: str):
    tokenizer, model = load_model(); items = [p for i, p in enumerate(sorted(input_root.glob("*/*.parquet"))) if i % shard_mod == shard_rem]
    totals = []; log = EXP / f"logs/xlmr_{worker_name}.jsonl"; log.parent.mkdir(parents=True, exist_ok=True)
    for ip in items:
        op = output_root / ip.parent.name / ip.name
        if op.exists() and op.with_suffix(".runtime.json").exists(): continue
        frame = pd.read_parquet(ip); dataset, token_seconds = tokenize(frame, tokenizer)
        probs, meta = score_dataset(dataset, tokenizer, model, batch)
        key = ["source1_entity_id", "candidate_entity_id", "candidate_source"]
        out = frame[key].copy(); out["neural_probability"] = probs; atomic_parquet(out, op)
        meta.update({"tokenization_seconds": token_seconds, "total_seconds": token_seconds + meta["gpu_seconds"],
            "input": str(ip), "output": str(op), "worker": worker_name, "gpu": torch.cuda.get_device_name(0)})
        atomic_json(op.with_suffix(".runtime.json"), meta)
        with log.open("a") as f: f.write(json.dumps(meta) + "\n")
        totals.append(meta); print("XLMR_READY", json.dumps(meta), flush=True)
    saved = [json.loads(p.read_text()) for p in output_root.glob("*/*.runtime.json") if json.loads(p.read_text()).get("worker") == worker_name]
    summary = {"worker": worker_name, "gpu": torch.cuda.get_device_name(0), "batch_size": batch,
        "parts": len(saved), "rows": sum(x["rows"] for x in saved), "tokenization_seconds": sum(x["tokenization_seconds"] for x in saved),
        "gpu_seconds": sum(x["gpu_seconds"] for x in saved), "total_seconds": sum(x["total_seconds"] for x in saved),
        "peak_reserved_gpu_gb": max((x["peak_reserved_gpu_gb"] for x in saved), default=0)}
    atomic_json(EXP / f"diagnostics/xlmr_{worker_name}.json", summary)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("benchmark"); b.add_argument("--input", type=Path, required=True); b.add_argument("--batches", type=int, nargs="+", required=True); b.add_argument("--output", type=Path, required=True)
    f = sub.add_parser("file"); f.add_argument("--input", type=Path, required=True); f.add_argument("--output", type=Path, required=True); f.add_argument("--batch", type=int, required=True); f.add_argument("--worker-name", default="single")
    r = sub.add_parser("root"); r.add_argument("--input-root", type=Path, required=True); r.add_argument("--output-root", type=Path, required=True); r.add_argument("--batch", type=int, required=True); r.add_argument("--shard-mod", type=int, default=1); r.add_argument("--shard-rem", type=int, default=0); r.add_argument("--worker-name", required=True)
    a = ap.parse_args()
    if a.command == "benchmark": benchmark(a.input, a.batches, a.output)
    elif a.command == "file": infer_file(a.input, a.output, a.batch, a.worker_name)
    else: infer_root(a.input_root, a.output_root, a.batch, a.shard_mod, a.shard_rem, a.worker_name)


if __name__ == "__main__": main()
