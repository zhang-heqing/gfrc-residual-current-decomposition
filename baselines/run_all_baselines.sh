#!/usr/bin/env bash
set -euo pipefail

SPLIT_CONFIG="${1:-data_preprocessing/output/benchmarks/shanse001_rebuilt_entity_split/split_config.json}"

python baselines/cnn_bilstm/train.py --split-config "$SPLIT_CONFIG"
python baselines/prophet/train.py --split-config "$SPLIT_CONFIG"
python baselines/tft/train.py --split-config "$SPLIT_CONFIG"
python baselines/timexer/train.py --split-config "$SPLIT_CONFIG"
python baselines/deterministic_physics/train.py --split-config "$SPLIT_CONFIG"
python baselines/cvae/train.py --split-config "$SPLIT_CONFIG"
python baselines/diffusion/train.py --split-config "$SPLIT_CONFIG"
python baselines/cinn/train.py --split-config "$SPLIT_CONFIG"
