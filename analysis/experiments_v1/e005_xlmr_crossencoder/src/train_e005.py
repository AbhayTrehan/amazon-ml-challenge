#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, math, os, platform, subprocess, time
from pathlib import Path

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common import (MODEL_ID, LengthBucketBatchSampler, PairCollator, PairDataset,
                    SEED, TokenizedCollator, pretokenize, seed_everything)


def gpu_mem_gb():
    return torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30


def loader(frame, tokenizer, max_length, batch_size, shuffle, workers):
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(PairDataset(frame), batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
                      collate_fn=PairCollator(tokenizer, max_length), generator=generator,
                      drop_last=False)


def optimizer_for(model, lr, wd):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if name.endswith("bias") or "LayerNorm.weight" in name else decay).append(p)
    return AdamW([{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}], lr=lr, fused=True)


def save_checkpoint(model, tokenizer, out, state):
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    torch.save(state, out / "training_state.pt")


def benchmark(model, train_loader, infer_loader, precision, accumulation, max_steps):
    device = torch.device("cuda")
    opt = optimizer_for(model, 2e-5, .01)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    model.train(); torch.cuda.reset_peak_memory_stats(); opt.zero_grad(set_to_none=True)
    warmup, measured, examples = 3, 0, 0
    start = None
    for step, batch in enumerate(train_loader):
        if step == warmup:
            torch.cuda.synchronize(); start = time.perf_counter()
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=dtype):
            loss = model(**batch).loss / accumulation
        scaler.scale(loss).backward()
        if (step + 1) % accumulation == 0:
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        if step >= warmup:
            measured += 1; examples += len(batch["labels"])
        if step + 1 >= warmup + max_steps:
            break
    torch.cuda.synchronize(); train_s = time.perf_counter() - start
    peak_alloc, peak_reserved = gpu_mem_gb()
    model.eval(); inf_examples = 0
    with torch.inference_mode():
        # warmup
        for i, batch in enumerate(infer_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != "labels"}
            with torch.autocast("cuda", dtype=dtype): model(**batch)
            if i == 2: break
        torch.cuda.synchronize(); start = time.perf_counter()
        for i, batch in enumerate(infer_loader):
            inf_examples += len(batch["input_ids"])
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != "labels"}
            with torch.autocast("cuda", dtype=dtype): model(**batch)
            if i + 1 >= max_steps: break
        torch.cuda.synchronize(); infer_s = time.perf_counter() - start
    return {
        "training_examples_per_second": examples / train_s,
        "inference_examples_per_second": inf_examples / infer_s,
        "benchmark_training_examples": examples,
        "benchmark_inference_examples": inf_examples,
        "peak_allocated_gpu_gb": peak_alloc,
        "peak_reserved_gpu_gb": peak_reserved,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--token-stats", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--benchmark-only", action="store_true")
    ap.add_argument("--benchmark-output")
    ap.add_argument("--physical-batch-size", type=int, default=16)
    ap.add_argument("--effective-batch-size", type=int, default=64)
    ap.add_argument("--inference-batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--stop-after-epoch", type=int, default=2)
    ap.add_argument("--resume")
    ap.add_argument("--log-file")
    ap.add_argument("--token-cache", default=None)
    args = ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU is required")
    seed_everything()
    stats = json.loads(Path(args.token_stats).read_text()); max_length = int(stats["chosen_max_length"])
    precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    accumulation = math.ceil(args.effective_batch_size / args.physical_batch_size)
    effective = args.physical_batch_size * accumulation
    train = pd.read_parquet(args.train_file)
    tokenizer_source = args.resume if args.resume else MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    model_source = args.resume if args.resume else MODEL_ID
    model = AutoModelForSequenceClassification.from_pretrained(
        model_source, num_labels=2, use_safetensors=True, attn_implementation="sdpa"
    ).cuda()
    param_count = sum(p.numel() for p in model.parameters())
    prep_start=time.perf_counter();tokenized=pretokenize(train,tokenizer,max_length,args.token_cache);prep_seconds=time.perf_counter()-prep_start
    sampler=LengthBucketBatchSampler(tokenized.lengths,args.physical_batch_size,SEED)
    train_loader=DataLoader(tokenized,batch_sampler=sampler,num_workers=args.workers,pin_memory=True,
                            persistent_workers=args.workers>0,collate_fn=TokenizedCollator(tokenizer))
    infer_subset=torch.utils.data.Subset(tokenized,range(min(len(tokenized),16384)))
    infer_loader=DataLoader(infer_subset,batch_size=args.inference_batch_size,shuffle=False,num_workers=args.workers,
                            pin_memory=True,persistent_workers=args.workers>0,collate_fn=TokenizedCollator(tokenizer))
    if args.benchmark_only:
        result = benchmark(model, train_loader, infer_loader, precision, accumulation, 30)
        result.update({
            "gpu": torch.cuda.get_device_name(0), "max_length": max_length,
            "physical_batch_size": args.physical_batch_size, "effective_batch_size": effective,
            "gradient_accumulation_steps": accumulation, "inference_batch_size": args.inference_batch_size,
            "precision": precision, "parameter_count": param_count,
            "pretokenization_seconds_this_run": prep_seconds,
            "projected_epoch_seconds": len(train) / result["training_examples_per_second"],
            "projected_two_epoch_seconds": 2 * len(train) / result["training_examples_per_second"],
            "projected_calibration_shortlist_seconds": 61621 / result["inference_examples_per_second"],
            "projected_evaluation_shortlist_seconds": 61913 / result["inference_examples_per_second"],
            "projected_full_test_shortlist_seconds": (1732544 * 6.19) / result["inference_examples_per_second"],
            "projected_full_test_k10_seconds": (1732544 * 37.85) / result["inference_examples_per_second"],
        })
        Path(args.benchmark_output).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2)); return

    steps_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_steps = steps_per_epoch * 2
    optimizer = optimizer_for(model, 2e-5, .01)
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(total_steps * .05), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    start_epoch, global_step = 1, 0
    if args.resume:
        state = torch.load(Path(args.resume) / "training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"]); start_epoch = state["epoch"] + 1; global_step = state["global_step"]
    log_path = Path(args.log_file); log_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not log_path.exists() or log_path.stat().st_size == 0
    with log_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["epoch", "loss", "seconds", "examples_per_second", "optimizer_steps", "learning_rate", "peak_allocated_gpu_gb", "peak_reserved_gpu_gb"])
        if new_file: writer.writeheader()
        for epoch in range(start_epoch, args.stop_after_epoch + 1):
            sampler.set_epoch(epoch - 1)
            model.train(); optimizer.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats()
            epoch_start = time.perf_counter(); loss_sum = 0.0; examples = 0; opt_steps = 0
            for step, batch in enumerate(train_loader):
                batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
                amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
                with torch.autocast("cuda", dtype=amp_dtype):
                    loss = model(**batch).loss
                loss_sum += loss.item() * len(batch["labels"]); examples += len(batch["labels"])
                scaler.scale(loss / accumulation).backward()
                if (step + 1) % accumulation == 0 or step + 1 == len(train_loader):
                    scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                    scheduler.step(); global_step += 1; opt_steps += 1
            torch.cuda.synchronize(); elapsed = time.perf_counter() - epoch_start
            alloc, reserved = gpu_mem_gb()
            row = {"epoch": epoch, "loss": loss_sum/examples, "seconds": elapsed,
                   "examples_per_second": examples/elapsed, "optimizer_steps": opt_steps,
                   "learning_rate": scheduler.get_last_lr()[0], "peak_allocated_gpu_gb": alloc,
                   "peak_reserved_gpu_gb": reserved}
            writer.writerow(row); fh.flush()
            ckpt = Path(args.output_root) / f"checkpoint-epoch{epoch}"
            save_checkpoint(model, tokenizer, ckpt, {"epoch": epoch, "global_step": global_step,
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "max_length": max_length, "physical_batch_size": args.physical_batch_size,
                "effective_batch_size": effective, "precision": precision, "seed": SEED})
            print(json.dumps(row))


if __name__ == "__main__":
    main()
