# Baselines

All baselines in this directory reuse the rebuilt benchmark split:

`/path/to/safeleak-rcd/benchmark/split_config.json`

Default training policy is aligned across methods:

- `epochs=3`
- same train/val/test split
- separate output folder per method under `training_runs/baselines/`

Implemented methods:

- `cnn_bilstm`
- `prophet` (Prophet-style additive ridge baseline, executable without extra package installs)
- `tft` (lightweight in-repo implementation)
- `timexer` (lightweight in-repo implementation)
- `deterministic_physics`
- `cvae`
- `diffusion` (lightweight conditional denoising diffusion baseline)
- `cinn`

Canonical paper references are collected in [`PROVENANCE.md`](./PROVENANCE.md).

Example commands:

```bash
python baselines/cnn_bilstm/train.py
python baselines/prophet/train.py
python baselines/tft/train.py
python baselines/timexer/train.py
python baselines/deterministic_physics/train.py
python baselines/cvae/train.py
python baselines/diffusion/train.py
python baselines/cinn/train.py
```

Quick smoke-test examples:

```bash
python baselines/cnn_bilstm/train.py --epochs 1 --limit-train-batches 2 --limit-val-batches 1 --limit-test-batches 1 --limit-test-branches 1
python baselines/prophet/train.py --limit-branches 2
```
