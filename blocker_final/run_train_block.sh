#!/usr/bin/env bash
set -o pipefail

cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="$PWD/blocker_final/hf_cache"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

/usr/bin/time -v blocker_final/.venv/bin/python blocker_final/run_block_with_progress.py block \
  --data student_resource/dataset/train \
  --prefix train \
  --model blocker_final/model \
  --out blocker_final/validation/full_train \
  --k 10 \
  --nprobe 64 \
  --rev-cap 10 \
  --batch-size 1024 \
  --max-len 64 \
  2>&1 | tee blocker_final/validation/full_train/blocking.log
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > blocker_final/validation/full_train/exit_code
if [[ "$rc" -eq 0 ]]; then
  touch blocker_final/validation/full_train/COMPLETE
else
  touch blocker_final/validation/full_train/FAILED
fi
exit "$rc"
