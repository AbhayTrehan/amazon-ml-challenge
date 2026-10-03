#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-/home/satvik/miniforge3/envs/deepchem/bin/python}"
exec "$PYTHON" "$ROOT/analysis/experiments_v1/e004_k10_decode_and_gpu_bundle/run_e004.py"
