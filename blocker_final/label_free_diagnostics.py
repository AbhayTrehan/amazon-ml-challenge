#!/usr/bin/env python3
"""Label-free diagnostics for frozen Blocker v1 output; never changes candidates."""

import argparse

import polars as pl


def read_tsv(path: str) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False).fill_null("")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--candidates", required=True)
    args = parser.parse_args()

    s1 = read_tsv(f"{args.data}/{args.prefix}_source1.tsv").select(
        pl.col("entity_id").alias("s1"), "country", pl.col("business_address").alias("s1_address")
    )
    aux = pl.concat(
        [
            read_tsv(f"{args.data}/{args.prefix}_source2.tsv").with_columns(src=pl.lit("S2")),
            read_tsv(f"{args.data}/{args.prefix}_source3.tsv").with_columns(src=pl.lit("S3")),
        ]
    ).select(pl.col("entity_id").alias("a"), "src", pl.col("business_address").alias("a_address"))
    c = pl.read_parquet(args.candidates)

    counts = (
        s1.select("s1", "country")
        .join(c.group_by("s1").len(), on="s1", how="left")
        .with_columns(pl.col("len").fill_null(0))
        .group_by("country")
        .agg(
            s1=pl.len(),
            candidate_pairs=pl.col("len").sum(),
            candidate_per_s1=pl.col("len").mean(),
            candidate_p99=pl.col("len").quantile(0.99, interpolation="nearest"),
            candidate_max=pl.col("len").max(),
        )
        .sort("country")
    )
    print("CANDIDATE_COUNTS")
    print(counts)

    forward = c.filter(pl.col("fwd_rank").is_not_null()).join(aux, on="a", how="left")
    top1 = forward.filter(pl.col("fwd_rank") == 1).select("s1", "src", "a", pl.col("score").alias("top1"), "is_rev", "a_address")
    top2 = forward.filter(pl.col("fwd_rank") == 2).select("s1", "src", pl.col("score").alias("top2"))
    diag = (
        top1.join(top2, on=["s1", "src"], how="left")
        .join(s1, on="s1", how="left")
        .with_columns(
            margin=pl.col("top1") - pl.col("top2"),
            s1_house=pl.col("s1_address").str.extract(r"(\d+)", 1),
            a_house=pl.col("a_address").str.extract(r"(\d+)", 1),
        )
        .with_columns(
            house_comparable=pl.col("s1_house").is_not_null() & pl.col("a_house").is_not_null(),
            same_house=pl.col("s1_house").is_not_null()
            & pl.col("a_house").is_not_null()
            & (pl.col("s1_house") == pl.col("a_house")),
        )
        .group_by("country", "src")
        .agg(
            queries=pl.len(),
            top1_cosine_mean=pl.col("top1").mean(),
            top1_cosine_median=pl.col("top1").median(),
            top1_top2_margin_mean=pl.col("margin").mean(),
            top1_top2_margin_median=pl.col("margin").median(),
            mutual_best_rate=pl.col("is_rev").mean(),
            house_comparable_rate=pl.col("house_comparable").mean(),
            same_house_rate_all=pl.col("same_house").mean(),
            same_house_rate_comparable=pl.col("same_house").sum() / pl.col("house_comparable").sum(),
        )
        .sort("country", "src")
    )
    print("DENSE_FORWARD_DIAGNOSTICS")
    print(diag)


if __name__ == "__main__":
    main()
