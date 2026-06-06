# GFRC Full Implementation

This directory contains a more complete implementation of the paper-aligned
GFRC framework, rebuilt from the original prototype without modifying the
source prototype files.

Implemented components:

- branch-conditioned shared model
- multimodal conditioning from aggregate residual current, aggregate active
  power, and target-branch power cues
- consistent conditional flow matching training objective
- Kirchhoff-consistent physical regularization across all branches
- ODE-based sampling for target-branch reconstruction
- Monte Carlo sampling for predictive dispersion
- train/validation/test split, normalization, and evaluation metrics

Expected CSV columns:

- `total_residual_current`
- `total_power`
- `branch_1_power`, `branch_2_power`, ...
- `branch_1_current`, `branch_2_current`, ...

The number of branches is inferred automatically from matched
`branch_*_power` and `branch_*_current` columns.

Run training:

```bash
python3 train.py --data /path/to/residual_current_data.csv
```

Run training from an explicit entity-level split bundle:

```bash
python3 gfrc_full_impl/train.py \
  --split-config /path/to/safeleak-rcd/benchmark/split_config.json \
  --output-dir training_runs/shanse001_rebuilt_entity_split
```

The released code also supports manifest-based training and custom split generation if you rebuild the benchmark locally with the preprocessing scripts under `data_preprocessing/`.

Run inference from a trained model directory:

```bash
python3 gfrc_full_impl/infer.py \
  --model-dir training_runs/panel_240701_240801 \
  --data data_preprocessing/output/wechat_panels/panel_240701_240801/panel_240701_240801.csv \
  --output-csv inference_outputs/panel_240701_240801_predictions.csv
```

The inference script loads `best_gfrc_model.pt`, `run_config.json`, and `normalization_stats.json` from `--model-dir`, then writes one prediction row per sliding window end timestamp.

Optional arguments:

- `--output-dir` output directory for checkpoints and reports
- `--epochs`
- `--batch-size`
- `--seq-len`
- `--hidden-dim`
- `--num-samples` number of stochastic samples during evaluation

Main files:

- `config.py` configuration dataclass
- `dataset.py` data loading, normalization, and split logic
- `model.py` encoder, conditional flow model, and sampling
- `metrics.py` evaluation metrics
- `trainer.py` training, validation, and evaluation pipeline
- `train.py` command-line entry point
- `../data_preprocessing/build_entity_split_benchmark.py` entity-level split builder that keeps synthetic variants in training only
- `../data_preprocessing/build_temporal_split_benchmark.py` chronological split builder for a single usable nonzero panel
- `../data_preprocessing/convert_wechat_raw_to_csv.py` standalone converter from raw three-phase JSON branch files to model-ready single-phase CSV

Build a dataset from the raw WeChat-exported branch folders:

```bash
python3 data_preprocessing/convert_wechat_raw_to_csv.py \
  --input-root "/path/to/data_by_id_240701.0000-240801.0000"
```

Preprocessing choices in the converter:

- snaps irregular raw uploads to a fixed `10min` grid with a `±180s` tolerance
- linearly interpolates only short gaps by default, then drops any timestamps that are still incomplete
- uses `Inb` (fundamental residual current) by default
- converts three-phase power to a single-phase equivalent branch cue:
  - if one phase dominates long-run power share, keep that phase directly
  - otherwise build a weighted single-phase equivalent from the three phase powers

The generated CSV matches the training loader format:

- `timestamp`
- `total_residual_current`
- `total_power`
- `branch_1_power`, `branch_1_current`, ...

Training dependencies are listed in `requirements.txt`. For this JSON-to-CSV conversion script, no third-party package is required.
