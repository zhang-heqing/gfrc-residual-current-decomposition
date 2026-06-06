# Open-Source Release Plan

This project is prepared for a split release:

- `GitHub`: code, experiment scripts, and release documentation
- `Hugging Face Datasets`: benchmark CSV files and dataset metadata

## Recommended Repositories

Code repository on GitHub:

- suggested name: `gfrc-residual-current-decomposition`

Dataset repository on Hugging Face:

- suggested name: `safeleak-rcd`
- suggested public title: `SafeLeak-RCD: Residential Residual Current Decomposition Benchmark`
- repo type: `dataset`
- chosen dataset license: `CC-BY-NC-4.0`

## What Goes Where

### GitHub

Publish:

- `gfrc_full_impl/`
- `baselines/`
- `experiments/`
- `data_preprocessing/` scripts
- `README.md`
- `DATASET_CARD.md`
- `OPEN_SOURCE_RELEASE.md`

Do not publish in the code repo:

- large benchmark CSV payloads
- local training outputs
- private checkpoints that may still change
- unstable final-result snapshots before the paper result set is frozen

### Hugging Face Datasets

Publish:

- released benchmark split CSVs
- `split_config.json`
- processed-entity bundle and `manifest.json`
- dataset card (`README.md`)

Recommended release layout:

```text
hf_dataset_repo/
├── README.md
├── benchmark/
│   ├── train.csv
│   ├── validation.csv
│   ├── test.csv
│   └── split_config.json
└── processed_entities/
    ├── manifest.json
    ├── shanse001_aug_chunk_01_1min/
    ├── ...
    └── shanse001_jul_chunk_01_1min/
```

## Release Timing

Recommended staging:

1. Now: prepare both repos locally and check that the dataset package is upload-ready.
2. Before final public release: freeze the benchmark files and final table values.
3. Final release: publish code to GitHub and publish dataset to Hugging Face on the same day.
4. After release: tag the GitHub repo, then update the paper with the final public URLs.

## Public vs Gated Dataset

Recommended default:

- public dataset repo on Hugging Face

Use gated access only if you still want request-based control over downloads. Hugging Face supports gated datasets with either automatic or manual approval, and lets authors collect requester information through the dataset settings and dataset card metadata. Official docs:

- Dataset upload and card docs: https://huggingface.co/docs/hub/en/datasets-adding
- Dataset card metadata docs: https://huggingface.co/docs/hub/main/datasets-cards
- Manual split configuration docs: https://huggingface.co/docs/hub/datasets-manual-configuration
- Gated dataset docs: https://huggingface.co/docs/hub/en/datasets-gated

## Local Preparation Workflow

Generate the GitHub code-only release folder:

```bash
python ops/prepare_github_release.py
```

The script writes:

- `release/github_repo/`

This folder is the recommended upload source for the public GitHub repository.

Generate the Hugging Face upload folder:

```bash
python ops/prepare_hf_dataset_release.py
```

The script writes:

- `release/hf_dataset_repo/README.md`
- `release/hf_dataset_repo/benchmark/*`
- `release/hf_dataset_repo/processed_entities/*`

Then upload that folder to your Hugging Face dataset repo.

## Manual Publishing Checklist

### GitHub

1. Create a new GitHub repository.
2. Copy the code-only snapshot there.
3. Confirm that no private data bundle is included.
4. Push the first public tag, e.g. `v1.0.0`.

### Hugging Face

1. Prepare the upload folder with `python ops/prepare_hf_dataset_release.py`.
2. Upload a private staging repo first:

```bash
python ops/upload_hf_dataset.py --repo-id haayan/safeleak-rcd
```

3. Switch the repo to public once the Hub page looks correct.
4. Check that the Dataset Viewer correctly shows `train`, `validation`, and `test`.
5. Verify the dataset card text and that the repo license is set to `CC-BY-NC-4.0`.

## Final Paper Update

Once both repos are public, add:

- GitHub repository URL in the code availability statement
- Hugging Face dataset URL in the data availability statement
- optional version tag / release date if the journal allows it
