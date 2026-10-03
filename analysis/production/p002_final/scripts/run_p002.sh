#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/satvik/amazon-ml-2026
EXP="$ROOT/analysis/production/p002_final"
PY=/home/satvik/venvs/amazonml2026/bin/python
SRC="$EXP/src"
export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=20
mkdir -p "$EXP/logs" "$EXP/runtime"
exec 9>"$EXP/runtime/pipeline.lock"
flock -n 9 || { echo "P002 already running"; exit 2; }
rm -f "$EXP/runtime/FAILED" "$EXP/runtime/COMPLETE"
find "$EXP/cache/lightgbm" -type f -name '*.lock' -delete
started=$(date +%s)
printf '{"started_epoch":%s,"status":"running","gpu_mode":"single RTX A6000 by user override"}\n' "$started" > "$EXP/runtime/run_state.json"
children=()
cleanup(){ for pid in "${children[@]:-}"; do kill "$pid" 2>/dev/null || true; done; }
trap 'rc=$?; if [[ $rc -ne 0 ]]; then printf "exit_code=%s\nfailed_epoch=%s\n" "$rc" "$(date +%s)" > "$EXP/runtime/FAILED"; cleanup; fi' EXIT

echo "CPU_PIPELINE_START $(date -Is)"
for worker in 0 1 2 3; do
  "$PY" "$SRC/production.py" features --worker "$worker" >> "$EXP/logs/feature_worker_${worker}.log" 2>&1 &
  children+=("$!")
done
"$PY" "$SRC/production.py" retrieve >> "$EXP/logs/retrieval.log" 2>&1 &
retrieval_pid=$!; children+=("$retrieval_pid")
wait "$retrieval_pid"
for pid in "${children[@]:0:4}"; do wait "$pid"; done
children=()

echo "SHORTLIST_START $(date -Is)"
"$PY" "$SRC/production.py" shortlist >> "$EXP/logs/shortlist.log" 2>&1

echo "XLMR_A6000_START $(date -Is)"
CUDA_VISIBLE_DEVICES=GPU-be9e124c-5a77-cc0f-586d-9bd963e58e98 \
  "$PY" "$SRC/xlmr.py" root --input-root "$EXP/cache/shortlist" \
  --output-root "$EXP/cache/neural" --batch 512 --shard-mod 1 --shard-rem 0 \
  --worker-name a6000 >> "$EXP/logs/xlmr_a6000.log" 2>&1

echo "DECODE_VALIDATE_START $(date -Is)"
"$PY" "$SRC/production.py" decode >> "$EXP/logs/decode.log" 2>&1
"$PY" "$SRC/production.py" validate >> "$EXP/logs/validate.log" 2>&1
printf "completed_epoch=%s\n" "$(date +%s)" > "$EXP/runtime/COMPLETE"
echo "P002_COMPLETE $(date -Is)"
