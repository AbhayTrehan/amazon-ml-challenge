#!/usr/bin/env python3
"""
blocker_v1.py — scalable dense blocking for the Business Entity Resolution challenge.

What it does
------------
For every Source-1 (S1) record, propose a small set of Source-2/3 (S2/S3) candidate records that may be the same business.

  1. train     Fine-tune a multilingual bi-encoder (intfloat/multilingual-e5-base, MIT, 278M params) on true (S1, aux) pairs
               from the training ground truth, so noisy variants of one business get nearby embeddings.
  2. block     Embed every record once (linear cost), build an approximate nearest-neighbour index per country with FAISS
               IVF-Flat, and search:
                 forward : each S1 -> its top-K nearest S2 records and top-K nearest S3 records (same country)
                 reverse : each S2/S3 record -> its top-1 nearest S1 (same country); that record becomes a candidate of that S1
               Reverse candidates are capped per S1 (default 10) to stop "hub" S1 from collecting thousands of records.
               Output: candidates.parquet (with scores/ranks) and candidate_pairs.tsv in the official submission format.
  3. evaluate  Recall and oracle macro F0.5 of the candidate set against a ground-truth file (train/validation only).

Why it is scalable
------------------
  * Each record is embedded exactly once, independently of every other record -> linear, trivially parallel.
  * Search never compares a query with every record: IVF groups vectors into ~4*sqrt(N) clusters and each query scans only
    `nprobe` of them (~1% of the index at nprobe=64 for millions of records) -> sublinear per query.
  * All indexes and searches stay inside one country value (country is treated as an open set of labels; France works
    with no special code). The candidate count per S1 is bounded: 2*K forward + at most `rev_cap` reverse.

Measured (frozen train holdout, full-size indexes; oracle = F0.5 of a perfect matcher restricted to the candidates):
  K=10, nprobe=64, rev_cap=10 -> ~20.3 candidates/S1, oracle macro F0.5 0.9982 (exact brute-force search: 0.9988)
  K=3                       -> ~6.8  candidates/S1, oracle macro F0.5 0.9945

Usage
-----
  # 1) train once (optionally exclude validation S1 via a split file: TSV with columns source1_entity_id, split)
  python blocker_v1.py train --data dataset/train --out models/e5_blocker --split splits.tsv --train-split dev

  # 2) block a dataset folder (files {prefix}_source1.tsv, {prefix}_source2.tsv, {prefix}_source3.tsv)
  python blocker_v1.py block --data dataset/test --prefix test --model models/e5_blocker --out out/test_blocking

  # 3) evaluate a blocking run on labelled data (restrict to some S1 with --split/--eval-split)
  python blocker_v1.py evaluate --data dataset/train --prefix train --candidates out/train_blocking/candidates.parquet

Requirements: python>=3.10, polars, numpy, torch (CUDA), sentence-transformers>=3, datasets, faiss (faiss-gpu-cu12 or faiss-cpu).
Tip: set HF_HUB_OFFLINE=1 once the base model is cached if outbound network calls are slow or blocked.
"""
import argparse
import os
import sys
import time

import numpy as np
import polars as pl

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------------------------------------------- data
def read_tsv(path):
    # quote_char=None: addresses contain quotes/commas; empty fields become "" (not null)
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False).fill_null("")


def load_sources(data_dir, prefix):
    s1 = read_tsv(os.path.join(data_dir, f"{prefix}_source1.tsv"))
    aux = pl.concat([read_tsv(os.path.join(data_dir, f"{prefix}_source2.tsv")).with_columns(src=pl.lit("S2")),
                     read_tsv(os.path.join(data_dir, f"{prefix}_source3.tsv")).with_columns(src=pl.lit("S3"))])
    return s1, aux


def load_edges(gt_path):
    gt = read_tsv(gt_path)
    return (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids", empty_as_null=True)
              .filter(pl.col("matched_entity_ids") != "")
              .rename({"source1_entity_id": "s1", "matched_entity_ids": "a"}))


def record_text(df):
    """The exact text the model sees for a record. Keep identical between training and blocking."""
    return ("query: " + df["business_name"] + " | " + df["business_address"]).to_list()


# ---------------------------------------------------------------------------------------------------------------- train
def cmd_train(args):
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)
    from sentence_transformers.training_args import BatchSamplers

    s1, aux = load_sources(args.data, args.prefix)
    edges = load_edges(os.path.join(args.data, f"{args.prefix}_ground_truth.tsv"))
    if args.split:   # only train on S1 of the chosen split (keep validation S1 unseen)
        keep = read_tsv(args.split).filter(pl.col("split") == args.train_split).select(pl.col("source1_entity_id").alias("s1"))
        edges = edges.join(keep, on="s1", how="semi")
        log(f"restricted to split '{args.train_split}': {keep.height:,} S1, {edges.height:,} true pairs")
    pairs = (edges.sample(min(args.pairs, edges.height), seed=args.seed)
                  .join(s1.select(pl.col("entity_id").alias("s1"), n1="business_name", a1="business_address"), on="s1")
                  .join(aux.select(pl.col("entity_id").alias("a"), n2="business_name", a2="business_address"), on="a"))
    ds = Dataset.from_dict({
        "anchor": record_text(pairs.select(business_name="n1", business_address="a1")),
        "positive": record_text(pairs.select(business_name="n2", business_address="a2"))})
    log(f"training pairs: {len(ds):,}")

    model = SentenceTransformer(args.base_model)
    model.max_seq_length = args.max_len
    trainer = SentenceTransformerTrainer(
        model=model,
        args=SentenceTransformerTrainingArguments(
            output_dir=args.out, num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size,
            learning_rate=args.lr, warmup_ratio=0.05, bf16=True,
            batch_sampler=BatchSamplers.NO_DUPLICATES,        # an S1 never appears twice in a batch (no false negatives)
            logging_steps=100, save_strategy="no", report_to="none", dataloader_num_workers=4, seed=args.seed),
        train_dataset=ds,
        loss=losses.MultipleNegativesRankingLoss(model))      # contrastive loss with in-batch negatives
    trainer.train()
    model.save(args.out)
    log(f"saved model to {args.out}")


# ---------------------------------------------------------------------------------------------------------------- block
def embed(model, df, cache_path, batch_size):
    """L2-normalised fp16 embeddings, cached on disk so a rerun does not re-encode."""
    if cache_path and os.path.exists(cache_path):
        E = np.load(cache_path)
        if len(E) == df.height:
            return E
    t = time.time()
    E = model.encode(record_text(df), batch_size=batch_size, normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
    if cache_path:
        np.save(cache_path, E)
    log(f"  embedded {len(E):,} records ({len(E) / max(1e-9, time.time() - t):,.0f}/s)")
    return E


class AnnIndex:
    """FAISS IVF-Flat (inner product on normalised vectors = cosine). Falls back to exact Flat for tiny inputs."""

    def __init__(self, X, nprobe, use_gpu, gpu_res=None):
        import faiss
        X = np.ascontiguousarray(X, dtype=np.float32)
        n, d = X.shape
        if n < 20_000:                                    # too small for clustering to pay off
            index = faiss.IndexFlatIP(d)
            self.scanned = 1.0
        else:
            nlist = int(2 ** round(np.log2(4 * np.sqrt(n))))
            index = faiss.index_factory(d, f"IVF{nlist},Flat", faiss.METRIC_INNER_PRODUCT)
            self.scanned = min(1.0, nprobe / nlist)
        if use_gpu:
            index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
        if not index.is_trained:
            sample = X[np.random.default_rng(0).choice(n, size=min(n, 50 * nlist), replace=False)]
            index.train(sample)
            if use_gpu:
                faiss.GpuParameterSpace().set_index_parameter(index, "nprobe", nprobe)
            else:
                faiss.extract_index_ivf(index).nprobe = nprobe
        index.add(X)
        self.index = index

    def search(self, Q, k, chunk=500_000):
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        Ds, Is = [], []
        for i in range(0, len(Q), chunk):
            D, I = self.index.search(Q[i:i + chunk], k)
            Ds.append(D); Is.append(I)
        return np.concatenate(Ds), np.concatenate(Is)


def cmd_block(args):
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer

    os.makedirs(args.out, exist_ok=True)
    s1, aux = load_sources(args.data, args.prefix)
    use_gpu = torch.cuda.is_available() and hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0
    gpu_res = faiss.StandardGpuResources() if use_gpu else None
    model = SentenceTransformer(args.model, device="cuda" if torch.cuda.is_available() else "cpu")
    model.max_seq_length = args.max_len
    if torch.cuda.is_available():
        model.half()
    log(f"S1 {s1.height:,} | S2+S3 {aux.height:,} | FAISS on {'GPU' if use_gpu else 'CPU'}")

    parts = []
    for country in sorted(s1["country"].unique().to_list()):        # open set of country labels
        S = s1.filter(pl.col("country") == country)
        A = aux.filter(pl.col("country") == country)
        log(f"country '{country}': {S.height:,} S1, {A.height:,} S2+S3")
        if A.height == 0:
            continue
        tag = "".join(ch if ch.isalnum() else "_" for ch in country)
        E_s1 = embed(model, S, os.path.join(args.out, f"emb_s1__{tag}.npy"), args.batch_size)
        E_aux = embed(model, A, os.path.join(args.out, f"emb_aux__{tag}.npy"), args.batch_size)

        # forward: S1 -> top-K per source
        for src in ["S2", "S3"]:
            m = (A["src"] == src).to_numpy()
            if m.sum() == 0:
                continue
            ids = A.filter(pl.col("src") == src)["entity_id"]
            idx = AnnIndex(E_aux[m], args.nprobe, use_gpu, gpu_res)
            k = min(args.k, int(m.sum()))
            D, I = idx.search(E_s1, k)
            ok = I.ravel() >= 0
            parts.append(pl.DataFrame({
                "s1": np.repeat(S["entity_id"].to_numpy(), k)[ok], "a": ids.gather(I.ravel()[ok]),
                "fwd_rank": np.tile(np.arange(1, k + 1, dtype=np.int16), len(I))[ok], "score": D.ravel()[ok],
                "is_rev": np.zeros(ok.sum(), np.int8)}))
            log(f"  forward {src}: top-{k}, ~{idx.scanned:.2%} of index scanned per query")
            del idx

        # reverse: each aux -> its best S1; capped per S1
        idx = AnnIndex(E_s1, args.nprobe, use_gpu, gpu_res)
        D, I = idx.search(E_aux, 1)
        del idx
        ok = I[:, 0] >= 0
        rev = (pl.DataFrame({"a": A["entity_id"].filter(pl.Series(ok)), "s1": S["entity_id"].gather(I[ok, 0]),
                             "score": D[ok, 0]})
                 .with_columns(r=pl.col("score").rank("ordinal", descending=True).over("s1"))
                 .filter(pl.col("r") <= args.rev_cap).drop("r"))
        parts.append(rev.with_columns(fwd_rank=pl.lit(None, pl.Int16), is_rev=pl.lit(1, pl.Int8))
                        .select("s1", "a", "fwd_rank", "score", "is_rev"))
        log(f"  reverse: {rev.height:,} candidates kept (cap {args.rev_cap} per S1)")

    C = (pl.concat(parts)
           .group_by("s1", "a")
           .agg(fwd_rank=pl.col("fwd_rank").min(), score=pl.col("score").max(), is_rev=pl.col("is_rev").max()))
    C.write_parquet(os.path.join(args.out, "candidates.parquet"))

    # official format: one row per S1 (empty list allowed), S2/S3 ids only, no duplicates, comma-joined
    lists = C.group_by("s1").agg(pl.col("a").sort().str.join(",").alias("candidate_entity_ids"))
    out = (s1.select(pl.col("entity_id").alias("source1_entity_id"))
             .join(lists.rename({"s1": "source1_entity_id"}), on="source1_entity_id", how="left")
             .with_columns(pl.col("candidate_entity_ids").fill_null("")))
    out.write_csv(os.path.join(args.out, "candidate_pairs.tsv"), separator="\t", quote_style="never")
    n = C.group_by("s1").len()["len"]
    log(f"done: {C.height:,} candidate pairs, {C.height / s1.height:.1f} per S1 (p99 {n.quantile(0.99):.0f}, max {n.max()}) -> {args.out}")


# ------------------------------------------------------------------------------------------------------------- evaluate
def cmd_evaluate(args):
    s1, _ = load_sources(args.data, args.prefix)
    edges = load_edges(os.path.join(args.data, f"{args.prefix}_ground_truth.tsv"))
    C = pl.read_parquet(args.candidates).select("s1", "a")
    S = s1.select(pl.col("entity_id").alias("s1"), "country")
    if args.split:
        keep = read_tsv(args.split).filter(pl.col("split") == args.eval_split).select(pl.col("source1_entity_id").alias("s1"))
        S = S.join(keep, on="s1", how="semi")
    card = edges.group_by("s1").agg(card=pl.len())
    tp = edges.join(C, on=["s1", "a"], how="semi").group_by("s1").agg(t=pl.len())
    n = C.group_by("s1").agg(n=pl.len())
    per = (S.join(card, on="s1", how="left").join(tp, on="s1", how="left").join(n, on="s1", how="left")
            .with_columns(pl.col("card").fill_null(0), pl.col("t").fill_null(0), pl.col("n").fill_null(0))
            .with_columns(f=pl.when(pl.col("card") == 0).then(1.0)          # oracle keeps only true candidates
                             .otherwise(1.25 * pl.col("t") / (1.25 * pl.col("t") + 0.25 * (pl.col("card") - pl.col("t"))))))
    summary = (per.group_by("country").agg(s1=pl.len(), cand_per_s1=pl.col("n").mean(),
                                            edge_recall=pl.col("t").sum() / pl.col("card").sum(), oracle_f05=pl.col("f").mean())
                  .sort("country"))
    print(summary)
    print(f"ALL: S1 {per.height:,} | candidates/S1 {per['n'].mean():.2f} | edge recall {per['t'].sum() / per['card'].sum():.4f} "
          f"| oracle macro F0.5 {per['f'].mean():.4f}")


# ------------------------------------------------------------------------------------------------------------------ cli
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="fine-tune the bi-encoder on true pairs")
    t.add_argument("--data", required=True, help="folder with {prefix}_source1/2/3.tsv and {prefix}_ground_truth.tsv")
    t.add_argument("--prefix", default="train")
    t.add_argument("--out", required=True, help="output model directory")
    t.add_argument("--base-model", default="intfloat/multilingual-e5-base")
    t.add_argument("--split", help="optional TSV (source1_entity_id, split) to restrict training S1")
    t.add_argument("--train-split", default="dev", help="split value to train on when --split is given")
    t.add_argument("--pairs", type=int, default=800_000, help="number of true pairs sampled for training")
    t.add_argument("--epochs", type=float, default=1.0)
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--lr", type=float, default=3e-5)
    t.add_argument("--max-len", type=int, default=64)
    t.add_argument("--seed", type=int, default=0)

    b = sub.add_parser("block", help="embed + index + search -> candidates")
    b.add_argument("--data", required=True)
    b.add_argument("--prefix", required=True, help="e.g. train or test")
    b.add_argument("--model", required=True, help="trained model directory")
    b.add_argument("--out", required=True)
    b.add_argument("--k", type=int, default=10, help="forward top-K per source (S2 and S3 separately)")
    b.add_argument("--nprobe", type=int, default=64, help="IVF clusters scanned per query")
    b.add_argument("--rev-cap", type=int, default=10, help="max reverse-proposed records kept per S1")
    b.add_argument("--batch-size", type=int, default=1024, help="embedding batch size")
    b.add_argument("--max-len", type=int, default=64)

    e = sub.add_parser("evaluate", help="recall / oracle F0.5 of a candidate set (needs ground truth)")
    e.add_argument("--data", required=True)
    e.add_argument("--prefix", default="train")
    e.add_argument("--candidates", required=True, help="candidates.parquet written by `block`")
    e.add_argument("--split", help="optional TSV (source1_entity_id, split) to restrict evaluated S1")
    e.add_argument("--eval-split", default="holdout")

    args = ap.parse_args()
    {"train": cmd_train, "block": cmd_block, "evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
