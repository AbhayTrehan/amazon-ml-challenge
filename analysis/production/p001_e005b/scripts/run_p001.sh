#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/satvik/amazon-ml-2026"
EXP="$ROOT/analysis/production/p001_e005b"
PY="/home/satvik/venvs/amazonml2026/bin/python"
SRC="$EXP/src"
export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$EXP/runtime" "$EXP/logs"
exec 9>"$EXP/runtime/pipeline.lock"
flock -n 9 || { echo "P001 pipeline is already running"; exit 2; }
if [[ -f "$EXP/runtime/COMPLETE" ]]; then echo "P001 already complete"; exit 0; fi
rm -f "$EXP/runtime/FAILED"
trap 'rc=$?; if [[ "$rc" -ne 0 ]]; then printf "exit_code=%s\nfailed_epoch=%s\n" "$rc" "$(date +%s)" > "$EXP/runtime/FAILED"; fi' EXIT

stage() { echo "STAGE $1 $(date -Is)"; }
stage preflight
"$PY" "$SRC/production.py" preflight
"$PY" "$SRC/production.py" parity_prepare
if [[ ! -f "$EXP/cache/parity/fold0_neural.parquet" ]]; then
  GPU_UUID="$("$PY" "$SRC/select_gpu.py")"; export CUDA_VISIBLE_DEVICES="$GPU_UUID"
  "$PY" "$SRC/infer_shards.py" --input-file "$EXP/cache/parity/fold0_k10.parquet" --output-file "$EXP/cache/parity/fold0_neural.parquet" --apply-shortlist --batch-size 256 --workers 4
fi
"$PY" "$SRC/production.py" parity_finalize --neural-path "$EXP/cache/parity/fold0_neural.parquet"

stage index_build
"$PY" "$SRC/production.py" indexes
stage retrieval_benchmark
if [[ -f "$EXP/diagnostics/retrieval_benchmark.json" ]]; then
  echo "SKIP retrieval_benchmark: cached result exists"
else
  "$PY" "$SRC/production.py" benchmark
fi
stage full_k10_feature_lightgbm
"$PY" "$SRC/production.py" retrieve
stage neural_shortlist
"$PY" "$SRC/production.py" shortlist

stage xlmr_inference
GPU_UUID="$("$PY" "$SRC/select_gpu.py")"; export CUDA_VISIBLE_DEVICES="$GPU_UUID"
"$PY" "$SRC/infer_shards.py" --input-root "$EXP/cache/neural_shortlist" --output-root "$EXP/cache/neural_predictions" --batch-size 256 --workers 4
stage decode_outputs
"$PY" "$SRC/production.py" decode
"$PY" "$SRC/production.py" stats
stage validation_finalize
"$PY" "$SRC/production.py" validate_finalize
echo "P001 COMPLETE $(date -Is)"
