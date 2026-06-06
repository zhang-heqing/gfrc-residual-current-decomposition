# GFRC: Physics-Regularized Conditional Flow Matching for Residual Current Decomposition

This repository accompanies the paper:

`GFRC: Physics-Regularized Conditional Flow Matching for Residual Current Decomposition`

It contains:

- the main GFRC implementation
- deterministic and probabilistic baseline implementations
- data preprocessing and benchmark construction scripts
- paper-oriented experiment and summarization scripts

## Overview

GFRC addresses branch-conditioned residual current decomposition for electrical safety monitoring under:

- low signal-to-noise ratio
- strong cross-branch ambiguity
- limited observability
- low-rate monitoring summaries rather than raw waveform traces

The repository is organized for reproducible experiments on the `SafeLeak-RCD` benchmark.

## Repository Structure

- `gfrc_full_impl/`: main GFRC training, inference, and evaluation code
- `baselines/`: baseline methods for the SafeLeak-RCD benchmark
- `data_preprocessing/`: raw-to-CSV conversion, augmentation, and benchmark builders
- `experiments/`: experiment runners, result summarizers, and paper-table exporters
- `DATASET_CARD.md`: benchmark description and release notes
- `OPEN_SOURCE_RELEASE.md`: code/data release workflow

## Dataset Release

The benchmark dataset is released separately from the GitHub code repository.

- dataset name: `SafeLeak-RCD: Residential Residual Current Decomposition Benchmark`
- host: Hugging Face Datasets
- URL: `https://huggingface.co/datasets/haayan/safeleak-rcd`
- license: `CC-BY-NC-4.0`

The code repository is intended to hold code, documentation, and experiment logic. Large benchmark CSV payloads should be hosted through the dataset release rather than duplicated in GitHub.

## Quickstart

Install dependencies:

```bash
pip install -r gfrc_full_impl/requirements.txt
pip install -r baselines/requirements.txt
```

Train GFRC on a released benchmark split:

```bash
python gfrc_full_impl/train.py \
  --split-config /path/to/safeleak-rcd/benchmark/split_config.json \
  --output-dir training_runs/shanse001_rebuilt_entity_split
```

Run the baseline suite:

```bash
bash baselines/run_all_baselines.sh \
  /path/to/safeleak-rcd/benchmark/split_config.json
```

Export main paper summaries:

```bash
python experiments/summarize_main_results.py
python experiments/summarize_ablations.py
python experiments/summarize_efficiency.py
```

## Benchmark Characteristics

The main SafeLeak-RCD benchmark used in the paper has:

- `12` branches
- `1`-minute target sampling interval
- entity-disjoint `train/validation/test = 5/1/1`
- synthetic variants used in training only
- real-only validation and test splits

The released rows are low-rate monitoring summaries rather than raw instantaneous waveform samples.

## Reproducibility Notes

- Historical manifests may contain author-side absolute paths, but the current loaders resolve benchmark files relative to the bundle location whenever possible.
- To prepare public release bundles locally:
  - GitHub code release: `python ops/prepare_github_release.py`
  - Hugging Face dataset release: `python ops/prepare_hf_dataset_release.py`

## Release Guidance

If you plan to publish this repository, read:

- `RELEASE_CHECKLIST.md`
- `OPEN_SOURCE_RELEASE.md`

These files describe what should go to GitHub, what should go to Hugging Face, and what should remain private.

## License

- Code license: `MIT` (see `LICENSE`)
- Dataset license: `CC-BY-NC-4.0`
