#!/usr/bin/env python3
"""Create the canonical Blocker v1 SHA1-mod-5 S1 split."""

import argparse
import hashlib

import polars as pl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source1", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    s1 = pl.read_csv(
        args.source1,
        separator="\t",
        quote_char=None,
        infer_schema=False,
    )
    entity_ids = s1["entity_id"].to_list()
    split = [
        "holdout" if int(hashlib.sha1(entity_id.encode("utf-8")).hexdigest()[:8], 16) % 5 == 0 else "dev"
        for entity_id in entity_ids
    ]
    pl.DataFrame(
        {"source1_entity_id": entity_ids, "split": split}
    ).write_csv(args.out, separator="\t", quote_style="never")


if __name__ == "__main__":
    main()
