#!/usr/bin/env bash
set -euo pipefail
PY="${PYTHON:-/home/satvik/miniforge3/envs/deepchem/bin/python}"
export PYTHONPATH="analysis/data_audit_v1/scripts"
"$PY" analysis/data_audit_v1/scripts/build_duckdb.py
"$PY" analysis/data_audit_v1/scripts/analyze_core.py
"$PY" analysis/data_audit_v1/scripts/physical_integrity.py
"$PY" analysis/data_audit_v1/scripts/similarity_and_artifacts.py
"$PY" analysis/data_audit_v1/scripts/train_test_shift.py
"$PY" analysis/data_audit_v1/scripts/normalization_views.py
"$PY" analysis/data_audit_v1/scripts/text_characterization.py
"$PY" analysis/data_audit_v1/scripts/id_artifact_controls.py
# Full-index retrieval: deliberately sequential for RAM safety.
"$PY" analysis/data_audit_v1/scripts/blocking_benchmark.py --source 2 --country India --channel name_char --queries 5000 --topk 100
"$PY" analysis/data_audit_v1/scripts/blocking_benchmark.py --source 3 --country India --channel name_char --queries 2000 --topk 50
"$PY" analysis/data_audit_v1/scripts/blocking_benchmark.py --source 2 --country US --channel name_char --queries 2000 --topk 50
"$PY" analysis/data_audit_v1/scripts/blocking_benchmark.py --source 3 --country US --channel name_char --queries 1000 --topk 50
for src in 2 3; do
  for country in India US; do
    "$PY" analysis/data_audit_v1/scripts/blocking_benchmark.py --source "$src" --country "$country" --channel address_char --queries 1000 --topk 50
  done
done
"$PY" analysis/data_audit_v1/scripts/combine_blocking.py
"$PY" analysis/data_audit_v1/scripts/france_retrieval.py
"$PY" analysis/data_audit_v1/scripts/extra_analysis.py
"$PY" analysis/data_audit_v1/scripts/taxonomy.py
"$PY" analysis/data_audit_v1/scripts/finalize_artifacts.py
MPLCONFIGDIR="/tmp/mlchallenge-matplotlib" "$PY" analysis/data_audit_v1/scripts/make_plots.py
