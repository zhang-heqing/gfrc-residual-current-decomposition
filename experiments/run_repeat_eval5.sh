#!/usr/bin/env bash
set -euo pipefail

SPLIT_CONFIG="${1:-data_preprocessing/output/benchmarks/shanse001_rebuilt_entity_split/split_config.json}"
OUT_DIR="${2:-experiments/output_repeat_eval5}"
REPEATS="${3:-5}"

mkdir -p "${OUT_DIR}"

python experiments/repeat_evaluate_runs.py \
  --split-config "${SPLIT_CONFIG}" \
  --output-dir "${OUT_DIR}" \
  --repeats "${REPEATS}"

python experiments/summarize_repeat_evals.py \
  --input-csv "${OUT_DIR}/per_repeat_metrics.csv" \
  --output-csv "${OUT_DIR}/repeat_eval_summary.csv"
