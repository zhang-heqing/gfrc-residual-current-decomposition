# Baseline provenance

This repository contains lightweight in-repo reimplementations for paper-oriented comparison. The references below are the canonical works the baselines map to.

| Method | Canonical source | Note |
| --- | --- | --- |
| CNN-BiLSTM | [A CNN-BiLSTM Architecture for Macroeconomic Time Series Forecasting](https://www.mdpi.com/2673-4591/39/1/33) | Hybrid CNN + BiLSTM forecaster used as a deterministic deep baseline. |
| Prophet | [Forecasting at Scale](https://doi.org/10.1080/00031305.2017.1380080) | The baseline here is a lightweight Prophet-style additive regression approximation. |
| TFT | [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://research.google/pubs/temporal-fusion-transformers-for-interpretable-multi-horizon-time-series-forecasting/) | Reference architecture for the attention-based deterministic baseline. |
| TimeXer | [TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables](https://papers.nips.cc/paper_files/paper/2024/hash/0113ef4642264adc2e6924a3cbbdf532-Abstract-Conference.html) | Reference architecture for the exogenous-variable-aware Transformer baseline. |
| Deterministic-Physics | Internal control variant | Matched deterministic version of the GFRC pipeline, used for fairness and attribution. |
| CVAE | [Learning Structured Output Representation using Deep Conditional Generative Models](https://papers.neurips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-generative-models) and [Auto-Encoding Variational Bayes](https://doi.org/10.48550/arXiv.1312.6114) | Conditional latent-variable baseline. |
| Diffusion | [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) and [CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation](https://proceedings.neurips.cc/paper_files/paper/2021/hash/c355dc5b2394b245b0e1f2586b44b4f8-Abstract.html) | Lightweight conditional denoising diffusion sequence baseline adapted to branch-conditioned residual current recovery. |
| cINN | [Guided Image Generation with Conditional Invertible Neural Networks](https://arxiv.org/abs/1907.02392) and [Network-to-Network Translation with Conditional Invertible Neural Networks](https://proceedings.neurips.cc/paper/2020/hash/1cfa81af29c6f2d8cacb44921722e753-Abstract.html) | Conditional invertible baseline adapted to the sequence setting. |

Implementation note: the code in this repo is a task-specific reproduction, not an official release from the cited authors.
