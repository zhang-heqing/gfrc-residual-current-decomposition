#!/usr/bin/env bash
set -euo pipefail

SPLIT_CONFIG="${1:-data_preprocessing/output/benchmarks/shanse001_rebuilt_entity_split/split_config.json}"
OUT_DIR="${2:-experiments/output/robustness}"
GFRC_RUN_DIR="${3:-training_runs/shanse001_rebuilt_entity_split}"
BASELINE_ROOT="${4:-training_runs/baselines}"

mkdir -p "${OUT_DIR}"

python experiments/evaluate_cue_robustness.py \
  --run-dir "${GFRC_RUN_DIR}" \
  --split-config "${SPLIT_CONFIG}" \
  --output-json "${OUT_DIR}/gfrc.json" \
  --output-csv "${OUT_DIR}/gfrc.csv"

python experiments/evaluate_cue_robustness.py \
  --run-dir "${BASELINE_ROOT}/deterministic_physics" \
  --split-config "${SPLIT_CONFIG}" \
  --output-json "${OUT_DIR}/deterministic_physics.json" \
  --output-csv "${OUT_DIR}/deterministic_physics.csv"

python experiments/evaluate_cue_robustness.py \
  --run-dir "${BASELINE_ROOT}/timexer" \
  --split-config "${SPLIT_CONFIG}" \
  --output-json "${OUT_DIR}/timexer.json" \
  --output-csv "${OUT_DIR}/timexer.csv"

python experiments/evaluate_cue_robustness.py \
  --run-dir "${BASELINE_ROOT}/cvae" \
  --split-config "${SPLIT_CONFIG}" \
  --output-json "${OUT_DIR}/cvae.json" \
  --output-csv "${OUT_DIR}/cvae.csv"
