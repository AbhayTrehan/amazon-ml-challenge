#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/satvik/amazon-ml-2026"
EXP="$ROOT/analysis/experiments_v1/e005_xlmr_crossencoder"
PY="/home/satvik/venvs/amazonml2026/bin/python"
SRC="$EXP/src"
BUNDLE="$ROOT/analysis/experiments_v1/e004_k10_decode_and_gpu_bundle/gpu_bundle"
GPU_UUID="GPU-f5f574b3-d0dc-fd72-5fd4-2d913d115b17"
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$EXP/model" "$EXP/outputs" "$EXP/runtime"
exec 9>"$EXP/runtime/pipeline.lock"
flock -n 9 || { echo "E005 pipeline is already running"; exit 2; }
date +%s > "$EXP/runtime/pipeline_start_epoch.txt"
rm -f "$EXP/runtime/FAILED" "$EXP/runtime/COMPLETE"
trap 'rc=$?; if [ "$rc" -ne 0 ]; then printf "exit_code=%s\nfailed_epoch=%s\n" "$rc" "$(date +%s)" > "$EXP/runtime/FAILED"; fi' EXIT

train_common=(--train-file "$BUNDLE/train_pairs.parquet" --token-stats "$EXP/token_length_stats.json" --output-root "$EXP/model" --physical-batch-size 64 --effective-batch-size 64 --inference-batch-size 256 --workers 4 --log-file "$EXP/training_log.csv" --token-cache "$EXP/runtime/train_tokens.pt")
infer_common=(--max-length 104 --batch-size 256 --workers 4)

echo "STAGE epoch1 $(date -Is)"
"$PY" "$SRC/train_e005.py" "${train_common[@]}" --stop-after-epoch 1
echo "STAGE calibration_epoch1 $(date -Is)"
"$PY" "$SRC/infer_e005.py" --input "$EXP/runtime/calibration_shortlist.parquet" --checkpoint "$EXP/model/checkpoint-epoch1" --output "$EXP/outputs/calibration_predictions_epoch1.parquet" "${infer_common[@]}"
"$PY" "$SRC/evaluate_e005.py" calibration --predictions "$EXP/outputs/calibration_predictions_epoch1.parquet" --metadata "$EXP/runtime/selected_s1_metadata.parquet" --output-dir "$EXP/outputs" --epoch 1 --e005a-only

echo "STAGE epoch2 $(date -Is)"
"$PY" "$SRC/train_e005.py" "${train_common[@]}" --stop-after-epoch 2 --resume "$EXP/model/checkpoint-epoch1"
echo "STAGE calibration_epoch2 $(date -Is)"
"$PY" "$SRC/infer_e005.py" --input "$EXP/runtime/calibration_shortlist.parquet" --checkpoint "$EXP/model/checkpoint-epoch2" --output "$EXP/outputs/calibration_predictions_epoch2.parquet" "${infer_common[@]}"
"$PY" "$SRC/evaluate_e005.py" calibration --predictions "$EXP/outputs/calibration_predictions_epoch2.parquet" --metadata "$EXP/runtime/selected_s1_metadata.parquet" --output-dir "$EXP/outputs" --epoch 2 --e005a-only

"$PY" -c 'import json,pathlib; p=pathlib.Path("analysis/experiments_v1/e005_xlmr_crossencoder/outputs"); a=json.loads((p/"calibration_summary_epoch1.json").read_text())["best"]["E005A"]["macro_f05"]; b=json.loads((p/"calibration_summary_epoch2.json").read_text())["best"]["E005A"]["macro_f05"]; e=1 if a>=b else 2; pathlib.Path("analysis/experiments_v1/e005_xlmr_crossencoder/runtime/selected_epoch.txt").write_text(str(e)+"\n")'
EPOCH="$(tr -d '[:space:]' < "$EXP/runtime/selected_epoch.txt")"
cp "$EXP/outputs/calibration_predictions_epoch${EPOCH}.parquet" "$EXP/outputs/calibration_predictions.parquet"
cp "$EXP/outputs/calibration_predictions_epoch${EPOCH}.runtime.json" "$EXP/outputs/calibration_predictions.runtime.json"

echo "STAGE freeze_selected_epoch_$EPOCH $(date -Is)"
"$PY" "$SRC/evaluate_e005.py" calibration --predictions "$EXP/outputs/calibration_predictions.parquet" --metadata "$EXP/runtime/selected_s1_metadata.parquet" --output-dir "$EXP/outputs" --epoch "$EPOCH"
cp "$EXP/outputs/calibration_search_epoch${EPOCH}.csv" "$EXP/outputs/calibration_search.csv"
"$PY" -c 'import json,pathlib; e=pathlib.Path("analysis/experiments_v1/e005_xlmr_crossencoder"); n=int((e/"runtime/selected_epoch.txt").read_text()); b=json.loads((e/f"outputs/calibration_summary_epoch{n}.json").read_text())["best"]; (e/"model/frozen_parameters.json").write_text(json.dumps(b,indent=2)+"\n")'

echo "STAGE fold0_once $(date -Is)"
"$PY" "$SRC/infer_e005.py" --input "$EXP/runtime/evaluation_shortlist.parquet" --checkpoint "$EXP/model/checkpoint-epoch${EPOCH}" --output "$EXP/outputs/evaluation_predictions.raw.parquet" "${infer_common[@]}"
mv "$EXP/outputs/evaluation_predictions.raw.runtime.json" "$EXP/outputs/evaluation_predictions.runtime.json"
"$PY" "$SRC/evaluate_e005.py" final --predictions "$EXP/outputs/evaluation_predictions.raw.parquet" --metadata "$EXP/runtime/selected_s1_metadata.parquet" --output-dir "$EXP/outputs" --frozen "$EXP/model/frozen_parameters.json" --complete-input "$BUNDLE/evaluation_pairs.parquet"
rm "$EXP/outputs/evaluation_predictions.raw.parquet"

echo "STAGE promote_and_finalize $(date -Is)"
mv "$EXP/model/checkpoint-epoch${EPOCH}" "$EXP/model/best_checkpoint"
rm -rf "$EXP/model/checkpoint-epoch1" "$EXP/model/checkpoint-epoch2"
rm -f "$EXP/model/best_checkpoint/training_state.pt"
"$PY" "$SRC/finalize_e005.py"
echo "E005 COMPLETE $(date -Is)"
