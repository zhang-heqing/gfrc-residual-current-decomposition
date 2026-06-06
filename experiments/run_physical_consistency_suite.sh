#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-.}"
SPLIT_CONFIG="${2:-data_preprocessing/output/benchmarks/shanse001_rebuilt_entity_split/split_config.json}"
OUTPUT_ROOT="${3:-experiments/output_revision/physical_consistency}"
BATCH_SIZE="${4:-8}"
LIMIT_BATCHES="${LIMIT_BATCHES:-}"
NUM_SAMPLES_OVERRIDE="${NUM_SAMPLES_OVERRIDE:-4}"

cd "${ROOT_DIR}"
mkdir -p "${OUTPUT_ROOT}"

RUN_SPECS=(
  "GFRC|training_runs/shanse001_rebuilt_round3_prefix005_weakaug|gfrc"
  "Deterministic-Physics|training_runs/baselines_rerun_best_v1/deterministic_physics|deterministic_physics"
  "CVAE|training_runs/baselines_rerun_best_v1/cvae|cvae"
  "Diffusion|training_runs/baselines_rerun_best_v1/diffusion|diffusion"
)

for SPEC in "${RUN_SPECS[@]}"; do
  IFS="|" read -r LABEL RUN_DIR SLUG <<< "${SPEC}"
  if [[ ! -d "${RUN_DIR}" ]]; then
    echo "[skip] ${LABEL}: missing run dir ${RUN_DIR}"
    continue
  fi
  if [[ ! -f "${RUN_DIR}/best_gfrc_model.pt" && ! -f "${RUN_DIR}/best_model.pt" ]]; then
    echo "[skip] ${LABEL}: no supported checkpoint in ${RUN_DIR}"
    continue
  fi

  echo "[run] ${LABEL} -> ${OUTPUT_ROOT}/${SLUG}.json"
  CMD=(
    python experiments/evaluate_physical_consistency.py
    --run-dir "${RUN_DIR}"
    --split-config "${SPLIT_CONFIG}"
    --output-json "${OUTPUT_ROOT}/${SLUG}.json"
    --output-csv "${OUTPUT_ROOT}/${SLUG}.csv"
    --batch-size "${BATCH_SIZE}"
    --num-samples-override "${NUM_SAMPLES_OVERRIDE}"
  )
  if [[ -n "${LIMIT_BATCHES}" ]]; then
    CMD+=(--limit-batches "${LIMIT_BATCHES}")
  fi
  "${CMD[@]}"
done

python experiments/summarize_physical_consistency.py \
  --input-dir "${OUTPUT_ROOT}" \
  --output-csv "${OUTPUT_ROOT}/summary.csv"

echo "Physical consistency suite complete: ${OUTPUT_ROOT}/summary.csv"
