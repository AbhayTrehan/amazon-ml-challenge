#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/satvik/amazon-ml-2026
EXP="$ROOT/analysis/experiments_v1/e008_final_robust_matcher"
SRC="$EXP/src"
PY=/home/satvik/venvs/amazonml2026/bin/python
ADA_UUID=GPU-f5f574b3-d0dc-fd72-5fd4-2d913d115b17
A6000_UUID=GPU-be9e124c-5a77-cc0f-586d-9bd963e58e98
export PYTHONPATH="$SRC"
cd "$ROOT"
for epoch in 1 2 3; do [[ -f "$EXP/model/xlmr/epoch-$epoch/model.safetensors" ]]; done

# Score independent epoch checkpoints concurrently across both GPUs.
CUDA_VISIBLE_DEVICES="$ADA_UUID" "$PY" "$SRC/neural.py" infer-new --input "$EXP/cache/shortlist/calibration.parquet" --checkpoint "$EXP/model/xlmr/epoch-1" --output "$EXP/cache/neural/calibration_epoch1.parquet" &
p1=$!
CUDA_VISIBLE_DEVICES="$A6000_UUID" "$PY" "$SRC/neural.py" infer-new --input "$EXP/cache/shortlist/calibration.parquet" --checkpoint "$EXP/model/xlmr/epoch-2" --output "$EXP/cache/neural/calibration_epoch2.parquet" &
p2=$!
wait "$p1"; wait "$p2"
CUDA_VISIBLE_DEVICES="$ADA_UUID" "$PY" "$SRC/neural.py" infer-new --input "$EXP/cache/shortlist/calibration.parquet" --checkpoint "$EXP/model/xlmr/epoch-3" --output "$EXP/cache/neural/calibration_epoch3.parquet"

"$PY" "$SRC/calibrate.py"
epoch=$(tr -d '[:space:]' < "$EXP/model/xlmr/selected_epoch.txt")

# E008 and frozen P002 validation inference are independent and use both GPUs.
CUDA_VISIBLE_DEVICES="$ADA_UUID" "$PY" "$SRC/neural.py" infer-new --input "$EXP/cache/shortlist/validation.parquet" --checkpoint "$EXP/model/xlmr/epoch-$epoch" --output "$EXP/cache/neural/validation_selected.parquet" &
ve=$!
CUDA_VISIBLE_DEVICES="$A6000_UUID" "$PY" "$SRC/neural.py" infer-old --input "$EXP/cache/shortlist/p002_validation.parquet" --output "$EXP/cache/neural/p002_validation.parquet" &
vp=$!
wait "$ve"; wait "$vp"

"$PY" "$SRC/evaluate.py"
date -u +%FT%TZ > "$EXP/runtime/VALIDATION_COMPLETE"
date -u +%FT%TZ > "$EXP/runtime/PRODUCTION_WITHHELD_BY_USER"
