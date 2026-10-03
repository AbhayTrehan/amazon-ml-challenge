#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/satvik/amazon-ml-2026
PY=/home/satvik/venvs/amazonml2026/bin/python
SRC="$ROOT/analysis/experiments_v1/e008_final_robust_matcher/src"
export PYTHONPATH="$SRC"
export CUDA_VISIBLE_DEVICES=0
cd "$ROOT"
"$PY" "$SRC/neural.py" infer-old-all
date -u +%FT%TZ > "$ROOT/analysis/experiments_v1/e008_final_robust_matcher/runtime/HISTORICAL_SCORING_COMPLETE"
