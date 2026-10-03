#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/satvik/amazon-ml-2026";EXP="$ROOT/analysis/production/p001_e005b";PY="/home/satvik/venvs/amazonml2026/bin/python";SRC="$EXP/src";export PYTHONPATH="$SRC";export TOKENIZERS_PARALLELISM=false
exec 9>"$EXP/runtime/optimized_pipeline.lock";flock -n 9 || { echo "optimized P001 already running";exit 2; }
rm -f "$EXP/runtime/FAILED" "$EXP/runtime/COMPLETE"
ND="$EXP/runtime/india3_name.done";AD="$EXP/runtime/india3_address.done";MD="$EXP/runtime/india3_merge.done";children=()
cleanup(){ for p in "${children[@]:-}";do kill "$p" 2>/dev/null||true;done; }
trap 'rc=$?;if [[ $rc -ne 0 ]];then printf "exit_code=%s\nfailed_epoch=%s\n" "$rc" "$(date +%s)" >"$EXP/runtime/FAILED";cleanup;fi' EXIT
echo "INDIA3_SPLIT_START $(date -Is)"
for w in 0 1 2 3;do "$PY" "$SRC/feature_worker.py" --worker "$w" --done-markers "$EXP/runtime/retrieval_ada.done" "$MD" >>"$EXP/logs/feature_worker_${w}.log" 2>&1 &children+=("$!");done
CUDA_VISIBLE_DEVICES=GPU-be9e124c-5a77-cc0f-586d-9bd963e58e98 "$PY" "$SRC/gpu_channel_worker.py" --country India --source 3 --channel name --batch 1024 --done "$ND" >>"$EXP/logs/india3_name_a6000.log" 2>&1 &g1=$!;children+=("$g1")
CUDA_VISIBLE_DEVICES=GPU-f5f574b3-d0dc-fd72-5fd4-2d913d115b17 "$PY" "$SRC/gpu_channel_worker.py" --country India --source 3 --channel address --batch 1024 --done "$AD" >>"$EXP/logs/india3_address_ada.log" 2>&1 &g2=$!;children+=("$g2")
"$PY" "$SRC/merge_channel_worker.py" --country India --source 3 --channel-done "$ND" "$AD" --done "$MD" >>"$EXP/logs/india3_merge.log" 2>&1 &mg=$!;children+=("$mg")
wait "$g1";wait "$g2";wait "$mg"
for p in "${children[@]:0:4}";do wait "$p";done;children=()
echo "STAGE shortlist $(date -Is)";"$PY" "$SRC/production.py" shortlist >>"$EXP/logs/optimized_post.log" 2>&1
echo "STAGE neural_dual $(date -Is)"
CUDA_VISIBLE_DEVICES=GPU-f5f574b3-d0dc-fd72-5fd4-2d913d115b17 "$PY" "$SRC/infer_shards.py" --input-root "$EXP/cache/neural_shortlist" --output-root "$EXP/cache/neural_predictions" --batch-size 256 --workers 4 --shard-mod 2 --shard-rem 0 --worker-name ada >>"$EXP/logs/neural_ada.log" 2>&1 &n1=$!;children+=("$n1")
CUDA_VISIBLE_DEVICES=GPU-be9e124c-5a77-cc0f-586d-9bd963e58e98 "$PY" "$SRC/infer_shards.py" --input-root "$EXP/cache/neural_shortlist" --output-root "$EXP/cache/neural_predictions" --batch-size 256 --workers 4 --shard-mod 2 --shard-rem 1 --worker-name a6000 >>"$EXP/logs/neural_a6000.log" 2>&1 &n2=$!;children+=("$n2")
wait "$n1";wait "$n2";children=()
echo "STAGE decode_validate $(date -Is)";"$PY" "$SRC/production.py" decode >>"$EXP/logs/optimized_post.log" 2>&1;"$PY" "$SRC/production.py" stats >>"$EXP/logs/optimized_post.log" 2>&1;"$PY" "$SRC/production.py" validate_finalize >>"$EXP/logs/optimized_post.log" 2>&1
printf "complete %s\n" "$(date -Is)" >"$EXP/runtime/COMPLETE";echo "OPTIMIZED_COMPLETE $(date -Is)"
