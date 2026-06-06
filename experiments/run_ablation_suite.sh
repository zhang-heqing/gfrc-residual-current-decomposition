#!/usr/bin/env bash
set -euo pipefail

SPLIT_CONFIG="${1:-data_preprocessing/output/benchmarks/shanse001_rebuilt_entity_split/split_config.json}"
ROOT_OUT="${2:-training_runs/ablations}"

mkdir -p "${ROOT_OUT}"

python gfrc_full_impl/train.py \
  --split-config "${SPLIT_CONFIG}" \
  --output-dir "${ROOT_OUT}/gfrc_full" \
  --epochs 3 \
  --batch-size 16

python gfrc_full_impl/train.py \
  --split-config "${SPLIT_CONFIG}" \
  --output-dir "${ROOT_OUT}/gfrc_no_cue" \
  --epochs 3 \
  --batch-size 16 \
  --disable-cue

python gfrc_full_impl/train.py \
  --split-config "${SPLIT_CONFIG}" \
  --output-dir "${ROOT_OUT}/gfrc_no_physics" \
  --epochs 3 \
  --batch-size 16 \
  --lambda-physical 0.0

python baselines/deterministic_physics/train.py \
  --split-config "${SPLIT_CONFIG}" \
  --output-dir "${ROOT_OUT}/deterministic_backbone" \
  --epochs 3 \
  --batch-size 32 \
  --lambda-physical 0.0

python baselines/deterministic_physics/train.py \
  --split-config "${SPLIT_CONFIG}" \
  --output-dir "${ROOT_OUT}/deterministic_physics" \
  --epochs 3 \
  --batch-size 32 \
  --lambda-physical 0.1

