# ADR-0009 — Separate training and inference environments; serve with ONNX Runtime

- **Status:** Accepted
- **Date:** 2026-10-27

## Context

- Deep-learning training needs PyTorch. On Linux, PyTorch from PyPI ships with CUDA: about **1.2 GB for `torch` plus about 2.9 GB of GPU libraries** (measured). Streamlit Community Cloud runs Linux and installs from `requirements.txt`, so that size would slow or break every deploy.
- The free host has limited memory (reported at roughly 0.7–2.7 GB per app; to be re-verified in Phase 14) and sleeps after inactivity.
- The dashboard must never retrain models on page load.

## Decision

**Two environments, one artifact format between them.**

| | Training environment | Inference environment |
|---|---|---|
| Where | Your laptop (CPU); optionally GitHub Actions | Streamlit Community Cloud (and locally) |
| Installs | `requirements-train.txt` (PyTorch CPU, statsmodels, LightGBM, XGBoost, onnx…) | `requirements.txt` (Streamlit, Plotly, pandas, **onnxruntime**, …) — **no PyTorch** |
| Does | Generate data, build features, train, tune, backtest, export | Load artifacts, run inference, scoring, simulation, agents, UI |

**Artifact registry** (replaces MLflow): `artifacts/<model>/<version>/` contains

- `model.onnx` — the serving model, exported with dynamic batch axes
- `model.pt` — PyTorch weights, for retraining and inspection
- `config.json`, `scaler.json` — everything needed to rebuild the inputs
- `metrics.json`, `loss_curve.csv`, `test_predictions.parquet` — what the dashboard displays
- `model_card.md` — purpose, data, split dates, metrics, limitations
- `SHA256SUMS` — integrity check used by the app

The dashboard reads only these files. A chart or metric without an artifact behind it is shown as "not trained yet".

**Parity check:** every export is tested by comparing ONNX Runtime and PyTorch outputs on the same inputs, at batch sizes 1 and N. In the feasibility benchmark, the maximum difference was 4.8 × 10⁻⁷.

## Consequences

- The hosted app installs about 66 MB for ONNX Runtime instead of about 4 GB for PyTorch.
- Every number on screen traces back to a saved, versioned training run.
- Some PyTorch-only features (for example, custom attention extraction) must be exported as extra ONNX outputs or saved at training time. The TFT will save its attention weights and variable-selection weights for the test period as artifacts.
- On Windows, the training environment installs the CPU build of PyTorch explicitly (`--index-url https://download.pytorch.org/whl/cpu`), roughly 200–300 MB.
