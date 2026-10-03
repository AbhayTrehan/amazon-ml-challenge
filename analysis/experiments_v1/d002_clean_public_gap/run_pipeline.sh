#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/satvik/amazon-ml-2026
OUT="$ROOT/analysis/experiments_v1/d002_clean_public_gap"
PY=/home/satvik/venvs/amazonml2026/bin/python
export PYTHONPATH="$ROOT/analysis/production/p002_final/src"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=20
mkdir -p "$OUT/logs" "$OUT/runtime"
exec 9>"$OUT/runtime/pipeline.lock"
flock -n 9 || { echo "D002 already running"; exit 2; }
rm -f "$OUT/runtime/FAILED" "$OUT/runtime/COMPLETE"
started=$(date +%s)
printf '{"started_epoch":%s,"status":"running","gpu":"RTX A6000"}\n' "$started" > "$OUT/runtime/run_state.json"
trap 'rc=$?; if [[ $rc -ne 0 ]]; then printf "exit_code=%s\nfailed_epoch=%s\n" "$rc" "$(date +%s)" > "$OUT/runtime/FAILED"; fi' EXIT

echo "RETRIEVE_START $(date -Is)"
retrieve_pids=()
for worker in 0 1; do
  "$PY" "$OUT/run_d002.py" retrieve --worker "$worker" --workers 2 >> "$OUT/logs/retrieve_worker_${worker}.log" 2>&1 &
  retrieve_pids+=("$!")
done
for pid in "${retrieve_pids[@]}"; do wait "$pid"; done
echo "FEATURES_START $(date -Is)"
pids=()
for worker in 0 1 2 3; do
  "$PY" "$OUT/run_d002.py" features --worker "$worker" --workers 4 >> "$OUT/logs/feature_worker_${worker}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo "SHORTLIST_START $(date -Is)"
"$PY" "$OUT/run_d002.py" shortlist >> "$OUT/logs/shortlist.log" 2>&1
echo "XLMR_START $(date -Is)"
CUDA_VISIBLE_DEVICES=GPU-be9e124c-5a77-cc0f-586d-9bd963e58e98 \
  "$PY" "$OUT/xlmr_clean.py" --batch 512 >> "$OUT/logs/xlmr_a6000.log" 2>&1
echo "EVALUATE_START $(date -Is)"
"$PY" "$OUT/run_d002.py" evaluate >> "$OUT/logs/evaluate.log" 2>&1
completed=$(date +%s)
printf '{"started_epoch":%s,"completed_epoch":%s,"seconds":%s,"status":"complete"}\n' "$started" "$completed" "$((completed-started))" > "$OUT/runtime/run_state.json"
printf 'completed_epoch=%s\n' "$completed" > "$OUT/runtime/COMPLETE"
echo "D002_EVALUATION_COMPLETE $(date -Is)"
