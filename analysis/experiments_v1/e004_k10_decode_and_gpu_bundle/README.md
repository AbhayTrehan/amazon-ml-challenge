# E004-lite: K10 decoding and GPU bundle

This is a cache-only continuation of E001. It performs no retrieval, model fitting, feature regeneration, network access, or GPU work. It reads E001 K10 flags and LightGBM scores, freezes decoder parameters on fold 4 before loading fold 0, measures neural shortlist ceilings, and exports compact text-pair Parquets.

Run from the repository root:

```bash
PYTHON=/home/satvik/miniforge3/envs/deepchem/bin/python bash analysis/experiments_v1/e004_k10_decode_and_gpu_bundle/run_e004.sh
```

See `HANDOFF.md` for results and `gpu_bundle/bundle_manifest.json` for exact export provenance and checksums.
