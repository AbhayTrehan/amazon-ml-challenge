#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/satvik/amazon-ml-2026
PY=/home/satvik/venvs/amazonml2026/bin/python
SRC="$ROOT/analysis/experiments_v1/e008_final_robust_matcher/src"
export PYTHONPATH="$SRC"
cd "$ROOT"
"$PY" "$SRC/data_pipeline.py" retrieve-fold4
P002_FEATURE_THREADS=20 "$PY" "$SRC/data_pipeline.py" features-fold4
date -u +%FT%TZ > "$ROOT/analysis/experiments_v1/e008_final_robust_matcher/runtime/FOLD4_PREFREEZE_COMPLETE"
