# Dataset Card: SafeLeak-RCD

## Summary

SafeLeak-RCD targets branch-conditioned residual current decomposition for electrical safety monitoring. The task is to estimate the residual current of one selected branch from:

- aggregate residual current
- aggregate active power
- the selected branch's power cue

The SafeLeak-RCD release is designed for the GFRC paper revision and includes both the processed benchmark CSVs and the scripts needed to regenerate the benchmark structure.

## Files

Primary split bundle:

- `data_preprocessing/output/benchmarks/shanse001_rebuilt_entity_split/`

Processed source manifest used for the paper benchmark:

- `data_preprocessing/output/shanse001_rebuilt_v2/manifest.json`

Main preprocessing scripts:

- `data_preprocessing/convert_wechat_raw_to_csv.py`
- `data_preprocessing/augment_panel_datasets.py`
- `data_preprocessing/build_entity_split_benchmark.py`
- `data_preprocessing/build_temporal_split_benchmark.py`

## Schema

Each benchmark CSV contains time-series rows with columns such as:

- `timestamp`
- `total_residual_current`
- `total_power`
- `branch_1_power`, `branch_2_power`, ...
- `branch_1_current`, `branch_2_current`, ...
- `entity_id`
- `segment_id`
- `synthetic_variant`

The number of branches is inferred from matched `branch_*_power` and `branch_*_current` columns.

## Example Row

Each row is one timestamped panel snapshot containing aggregate channels plus all branch channels. These rows are processed interval-level monitoring summaries produced by the upstream device and subsequent conversion pipeline, not raw instantaneous waveform samples; the public values correspond to quantities integrated or accumulated over finite time windows. A representative row has the form:

```text
timestamp,total_residual_current,total_power,branch_1_power,branch_1_current,...,branch_12_power,branch_12_current,synthetic_variant,segment_id,entity_id
2024-08-01 00:10:00,13.32345,20632.769769,2210.233346,0.0036,...,3974.763915,2.774,0,shanse001_aug_chunk_01_1min_base,shanse001_aug_chunk_01_1min
```

For a branch-conditioned sample targeting branch `k`, the input-output relation is:

- inputs: `total_residual_current`, `total_power`, `branch_k_power`
- target: `branch_k_current`

## Benchmark Statistics

The released paper benchmark has:

- `12` branches
- `7` entities in total
- `1`-minute target sampling interval
- `10`-minute base interval before the synthetic densification step

Split policy used in the paper:

- Train: `5` entities, `104,835` rows total, `34,945` real base rows, synthetic ratio `0.667`
- Validation: `1` entity, `7,091` rows, real-only
- Test: `1` entity, `11,991` rows, real-only

Entity assignment for the main benchmark:

- Train: `shanse001_aug_chunk_01_1min`, `shanse001_aug_chunk_02_1min`, `shanse001_aug_chunk_06_1min`, `shanse001_aug_chunk_03_1min`, `shanse001_aug_chunk_05_1min`
- Validation: `shanse001_aug_chunk_04_1min`
- Test: `shanse001_jul_chunk_01_1min`

## Collection and Processing

The public benchmark is derived from synchronized residential monitoring panels.

Processing stages:

1. Convert raw branch folders to aligned CSV panels with `convert_wechat_raw_to_csv.py`.
2. Upsample and augment those panels into denser synthetic bundles with `augment_panel_datasets.py`.
3. Build the released entity-disjoint benchmark bundle with `build_entity_split_benchmark.py`.

The released benchmark therefore reflects a practical monitoring pipeline in which the public CSV rows correspond to low-rate windowed summaries obtained after finite-window integration and aggregation. Users should not interpret the benchmark as raw oscillographic data.

Important release policy:

- synthetic variants are used for training only
- validation and test remain real-only
- the same entity never appears in more than one split

## Intended Use

This benchmark is intended for:

- branch-conditioned residual current decomposition
- safety-oriented low-SNR inverse learning
- cumulative leakage evaluation
- physical-consistency analysis

It is not intended as a general appliance-level NILM benchmark.

## Limitations

- single-phase residential setting only
- low-frequency interval summaries make the task intentionally difficult
- the released rows are processed monitoring summaries rather than raw instantaneous waveforms
- synthetic densification is part of the training setup and should not be mistaken for raw hardware sampling capability
- heavy-tail cumulative errors remain possible even for the strongest reported models

## Reproducibility Notes

- Use the released `split_config.json` as the default entry point for training and evaluation.
- Historical manifests may contain author-side absolute paths; the repository now resolves these paths relative to the bundle location when possible.

## License Note

This dataset is released under `CC-BY-NC-4.0`. Downstream users may reuse, adapt, and redistribute it with attribution for non-commercial purposes.

Commercial use is not permitted without separate permission from the dataset authors.
