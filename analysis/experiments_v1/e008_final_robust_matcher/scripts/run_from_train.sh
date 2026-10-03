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
[[ -f "$EXP/cache/shortlist/calibration.parquet" && -f "$EXP/cache/shortlist/validation.parquet" ]]
CUDA_VISIBLE_DEVICES="$ADA_UUID" "$PY" "$SRC/neural.py" train
for epoch in 1 2 3; do
  CUDA_VISIBLE_DEVICES="$ADA_UUID" "$PY" "$SRC/neural.py" infer-new --input "$EXP/cache/shortlist/calibration.parquet" --checkpoint "$EXP/model/xlmr/epoch-$epoch" --output "$EXP/cache/neural/calibration_epoch$epoch.parquet"
done
"$PY" "$SRC/calibrate.py"
epoch=$(tr -d '[:space:]' < "$EXP/model/xlmr/selected_epoch.txt")
CUDA_VISIBLE_DEVICES="$ADA_UUID" "$PY" "$SRC/neural.py" infer-new --input "$EXP/cache/shortlist/validation.parquet" --checkpoint "$EXP/model/xlmr/epoch-$epoch" --output "$EXP/cache/neural/validation_selected.parquet"
CUDA_VISIBLE_DEVICES="$A6000_UUID" "$PY" "$SRC/neural.py" infer-old --input "$EXP/cache/shortlist/p002_validation.parquet" --output "$EXP/cache/neural/p002_validation.parquet"
"$PY" "$SRC/evaluate.py"
date -u +%FT%TZ > "$EXP/runtime/VALIDATION_COMPLETE"
if "$PY" -c "import json; raise SystemExit(0 if json.load(open('$EXP/validation/production_gate.json'))['pass'] else 1)"; then
  "$PY" "$SRC/production.py" features
  "$PY" "$SRC/production.py" shortlist
  CUDA_VISIBLE_DEVICES="$A6000_UUID" "$PY" "$SRC/production.py" neural
  "$PY" "$SRC/production.py" score
  "$PY" "$SRC/production.py" decode
  "$PY" "$SRC/production.py" validate
  "$PY" "$SRC/finalize.py"
else
  date -u +%FT%TZ > "$EXP/runtime/VALIDATION_FAILED_DO_NOT_SUBMIT"
fi
