#!/usr/bin/env python3
"""Extended reporting around the frozen blocker's unchanged candidate semantics."""

import argparse

import polars as pl


def read_tsv(path: str) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False).fill_null("")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--split")
    parser.add_argument("--eval-split", default="holdout")
    args = parser.parse_args()

    s1 = read_tsv(f"{args.data}/{args.prefix}_source1.tsv")
    candidates = pl.read_parquet(args.candidates)
    cohort = s1.select(pl.col("entity_id").alias("s1"), "country")
    if args.split:
        keep = (
            read_tsv(args.split)
            .filter(pl.col("split") == args.eval_split)
            .select(pl.col("source1_entity_id").alias("s1"))
        )
        cohort = cohort.join(keep, on="s1", how="semi")
    c = candidates.join(cohort.select("s1"), on="s1", how="semi")
    counts = cohort.join(c.group_by("s1").len(), on="s1", how="left").with_columns(
        pl.col("len").fill_null(0)
    )

    result = {
        "evaluated_s1": cohort.height,
        "candidate_pairs": c.height,
        "candidate_per_s1": float(counts["len"].mean()),
        "candidate_p99": float(counts["len"].quantile(0.99, interpolation="nearest")),
        "candidate_max": int(counts["len"].max()),
    }

    gt_path = f"{args.data}/{args.prefix}_ground_truth.tsv"
    try:
        gt = read_tsv(gt_path)
    except FileNotFoundError:
        print(result)
        return
    edges = (
        gt.with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids", empty_as_null=True)
        .filter(pl.col("matched_entity_ids") != "")
        .rename({"source1_entity_id": "s1", "matched_entity_ids": "a"})
        .join(cohort.select("s1"), on="s1", how="semi")
    )
    card = edges.group_by("s1").agg(card=pl.len())
    tp = edges.join(c.select("s1", "a"), on=["s1", "a"], how="semi").group_by("s1").agg(t=pl.len())
    per = (
        cohort.join(card, on="s1", how="left")
        .join(tp, on="s1", how="left")
        .with_columns(pl.col("card").fill_null(0), pl.col("t").fill_null(0))
        .with_columns(
            oracle_f05=pl.when(pl.col("card") == 0)
            .then(1.0)
            .otherwise(
                1.25 * pl.col("t")
                / (1.25 * pl.col("t") + 0.25 * (pl.col("card") - pl.col("t")))
            )
        )
    )
    result["true_edges"] = int(per["card"].sum())
    result["found_edges"] = int(per["t"].sum())
    result["edge_recall"] = float(per["t"].sum() / per["card"].sum())
    result["oracle_f05"] = float(per["oracle_f05"].mean())
    result["by_country"] = {
        row["country"]: {
            "s1": row["s1"],
            "edge_recall": row["edge_recall"],
            "oracle_f05": row["oracle_f05"],
        }
        for row in per.group_by("country").agg(
            s1=pl.len(),
            edge_recall=pl.col("t").sum() / pl.col("card").sum(),
            oracle_f05=pl.col("oracle_f05").mean(),
        ).iter_rows(named=True)
    }
    print(result)


if __name__ == "__main__":
    main()
