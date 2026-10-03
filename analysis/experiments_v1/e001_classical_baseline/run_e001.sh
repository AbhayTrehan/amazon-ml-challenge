#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-/home/satvik/miniforge3/envs/deepchem/bin/python}"
"$PYTHON" "$ROOT/analysis/experiments_v1/e001_classical_baseline/scripts/prepare_baseline.py" "$@"
if [[ " $* " != *" --sample-only "* ]]; then
  exec "$PYTHON" "$ROOT/analysis/experiments_v1/e001_classical_baseline/scripts/train_and_evaluate.py"
fi
