# P002 final production

Frozen R006B candidate generation and frozen E005B matching. No training,
tuning, external lookup, API, or web dependency is used. `candidate_pairs.tsv`
is the last candidate set supplied to XLM-R, as required by the official
challenge statement.

The full resumable run is:

```bash
bash /home/satvik/amazon-ml-2026/analysis/production/p002_final/scripts/run_p002.sh
```

Completed blocker, LightGBM, shortlist, and XLM-R parquet shards are detected
and reused. Final files are written under `output/`; the official validator is
run with `--check-ids`.
