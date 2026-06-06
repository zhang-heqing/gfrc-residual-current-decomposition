# Release Checklist

This checklist is for preparing the public GitHub code repository and the Hugging Face dataset release for the GFRC paper.

## 1. Do Not Publish

Do not publish the following to GitHub:

- raw local archives such as `archive.zip`
- private dataset copies under `gfrc_full_impl/data/*.csv`
- local compiled paper artifacts such as `*.aux`, `*.bbl`, `*.blg`, `*.log`, `*.xdv`
- local notes and handoff files such as `CODEX_HANDOFF_CONTEXT.md`, `revision_roadmap.md`, `审稿意见.docx`
- server-side result snapshots under `experiments/server_results_snapshot/`
- files containing machine-specific absolute paths
- private checkpoints, training outputs, or local cache directories

## 2. Publish to GitHub

Recommended GitHub contents:

- `gfrc_full_impl/`
- `data_preprocessing/` scripts only
- `ops/prepare_hf_dataset_release.py`
- `ops/upload_hf_dataset.py`
- `README.md`
- `DATASET_CARD.md`
- `OPEN_SOURCE_RELEASE.md`
- `RELEASE_CHECKLIST.md`

## 3. Review Before Publishing

Check the following manually:

- no GitHub password, token, API key, SSH key, or server alias appears anywhere in the repo
- no absolute paths remain in public-facing README files unless they are clearly marked as local examples
- no private benchmark CSV payload is committed to GitHub if it is intended for Hugging Face only
- all shell scripts intended for release either use relative paths or explain required environment variables

## 4. Preferred Split

Use this split release:

- GitHub: code, scripts, documentation
- Hugging Face Datasets: benchmark CSVs, split metadata, processed manifest, dataset card

## 5. Final Publishing Steps

1. Prepare the Hugging Face dataset folder with:

```bash
python ops/prepare_hf_dataset_release.py
```

2. Create a clean GitHub repo snapshot without private data or local artifacts.

3. Push the GitHub repository using a Personal Access Token or SSH key.

4. Upload the dataset bundle under `release/hf_dataset_repo/` to Hugging Face.

5. After both are public, update the manuscript with final public URLs if needed.
