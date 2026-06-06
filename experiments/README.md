# Experiments

This directory contains the paper-oriented experiment entry points. It is kept separate from the core training code so the project root does not become cluttered.

Main scripts:

- `run_ablation_suite.sh`
  Runs the core ablations:
  - GFRC full
  - GFRC w/o cue
  - GFRC w/o physics
  - Deterministic backbone
  - Deterministic + physics

- `evaluate_cue_robustness.py`
  Evaluates a trained run under cue corruption conditions:
  - clean
  - Gaussian noise
  - missing segments
  - scale bias
  - wrong cue

- `run_cue_robustness_suite.sh`
  Runs the cue robustness evaluation for several trained methods.

- `summarize_main_results.py`
  Exports the main comparison table CSV.

- `summarize_ablations.py`
  Exports the ablation table CSV.

- `summarize_efficiency.py`
  Exports the efficiency table CSV.

- `summarize_dataset_stats.py`
  Exports dataset-level statistics for each split configuration:
  - row count
  - entity count
  - branch count
  - inferred time interval
  - synthetic row ratio

- `stratified_analysis.py`
  Exports a stratified comparison by:
  - residual current strength
  - cue power strength
  - active branch count

- `generate_qualitative_cases.py`
  Selects easy / ambiguous / hard cases and writes qualitative bundles.
  If `matplotlib` is available, it also renders PNG figures.

- `summarize_robustness.py`
  Combines the per-method robustness CSV files into one summary CSV.

- `export_latex_tables.py`
  Converts the main CSV summaries into simple LaTeX table snippets.

Typical usage:

```bash
bash experiments/run_ablation_suite.sh
bash experiments/run_cue_robustness_suite.sh
python experiments/summarize_main_results.py
python experiments/summarize_ablations.py
python experiments/summarize_efficiency.py
python experiments/stratified_analysis.py --run-dir training_runs/shanse001_rebuilt_entity_split --run-dir training_runs/baselines/timexer --split-config /path/to/safeleak-rcd/benchmark/split_config.json
python experiments/generate_qualitative_cases.py --reference-run training_runs/shanse001_rebuilt_entity_split --compare-run training_runs/baselines/timexer --split-config /path/to/safeleak-rcd/benchmark/split_config.json
python experiments/export_latex_tables.py
```

Default output locations:

- `training_runs/ablations/`
- `experiments/output/main_comparison.csv`
- `experiments/output/ablation_comparison.csv`
- `experiments/output/efficiency_comparison.csv`
- `experiments/output/dataset_stats.csv`
- `experiments/output/dataset_stats.json`
- `experiments/output/stratified_analysis.csv`
- `experiments/output/robustness/`
- `experiments/output/robustness_summary.csv`
- `experiments/output/qualitative/`
- `experiments/output/latex/`
