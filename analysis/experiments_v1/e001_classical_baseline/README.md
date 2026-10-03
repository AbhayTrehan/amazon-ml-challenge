# E001 classical baseline

This workspace prepares the honest component-disjoint E001/E002/E003 experiment. It reuses the audited normalization, fold assignment, metric semantics, and lexical retrieval design. It reads but never modifies `analysis/data_audit_v1` or challenge data.

LightGBM 4.7.0 is installed in the experiment environment. The completed pipeline contains deterministic samples, cached top-20 retrieval, K10 derived from the same cache, candidate ceilings, compact hard-negative features, one fitted LightGBM model, fold-4 calibration, and untouched fold-0 E001/E002/E003 evaluation. See `HANDOFF.md` for all headline numbers.

Run from repository root:

```bash
PYTHON=/home/satvik/miniforge3/envs/deepchem/bin/python bash analysis/experiments_v1/e001_classical_baseline/run_e001.sh
```

Cached shards are reused on rerun. Pass `--force` only to intentionally regenerate all experiment-owned artifacts.
