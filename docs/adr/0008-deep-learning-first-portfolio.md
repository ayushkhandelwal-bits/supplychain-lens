# ADR-0008 — Deep-learning-first model portfolio

- **Status:** Accepted
- **Date:** 2026-10-27
- **Supersedes:** [ADR-0006](0006-replace-tft-and-ppo.md)

## Context

The project's goal changed to demonstrating deep learning for supply-chain decisions: training, evaluating, deploying and explaining neural networks. NEXUS-SCCT advertises TFT, LSTM and PPO models, but none was trained and none has a working inference path ([feature audit](../feature_audit.md)).

The deciding constraints are ₹0 and CPU only. A feasibility benchmark on a 2-vCPU machine at the planned data scale (400 series × 1,095 days) measured about 20 s per epoch for an LSTM and about 68 s per epoch for a compact TFT at stride 7, with inference under 150 ms for all series (`audit/bench_results_2vcpu.json`). So both are trainable without a GPU.

## Decision

### Core models (PyTorch)

| ID | Model | Task | Must beat |
|---|---|---|---|
| A | Seq2seq **LSTM**, quantile outputs (P10/P50/P90), SKU and warehouse embeddings | 28-day daily demand per SKU × warehouse | Seasonal-naive, Holt-Winters ETS, LightGBM |
| B | **Compact Temporal Fusion Transformer**, implemented in this project: variable selection networks, gated residual networks, static enrichment, LSTM encoder-decoder, masked interpretable multi-head attention, quantile head | Same task | LSTM and the baselines |
| C | **Autoencoder** on 14-day windows of daily operational features (supplier and warehouse level) | Anomaly score = reconstruction error; threshold set on validation data | Rolling robust z-score (median/MAD) |
| D | **Neural disruption classifier** (MLP on engineered features; optional GRU over 28-day history) | P(supplier disruption starts in the next 14 days) | Logistic regression, XGBoost |

### Optional models

- **E. GNN** — deferred unless a pre-registered test shows it beats model D given hand-made neighbour features.
- **F. PPO** — on a small environment built from the digital twin, compared with the base-stock policy over at least 30 episodes using common random numbers.

### Rules

1. **Baselines first.** Every neural model is reported next to its baselines on the same chronological test period (Q4 2025). The selected model is the one that wins on the agreed metric, which may be a baseline.
2. **Own TFT.** We implement the TFT ourselves so that every component can be explained. `pytorch-forecasting` (1.8.0, compatible with Python 3.10–3.14) may be used as an optional cross-check, never as the main result.
3. **Honest outputs.** Quantiles are called "forecast quantiles" until their coverage is measured. Feature importance is labelled *associational*. Anomalies are "investigation required" until confirmed.
4. **Reproducible.** Fixed seeds, saved configs, and the metrics and loss curves for every run stored with the artifact.

## Consequences

- The project contains four trained neural networks, two of which compete on the same task. That gives the Deep Learning Lab page a real comparison to show.
- Training runs on your laptop (minutes to under an hour per model). Serving is covered by ADR-0009.
- If a neural model loses to a baseline, we report it and serve the winner. That is a stronger interview story than an unexamined deep model.
