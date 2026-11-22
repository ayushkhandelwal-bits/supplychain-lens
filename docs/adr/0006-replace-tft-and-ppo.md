# ADR-0006 — Replace TFT and PPO in the deployed path

- **Status:** Superseded by [ADR-0008](0008-deep-learning-first-portfolio.md) on 2026-10-27
- **Date:** 2026-10-03

> Kept for history. The deep-learning-first change request made neural models the core of the project. The baselines and the classical inventory policies described here are still built, now as **benchmarks** that the neural models must beat.

## Context

**Temporal Fusion Transformer (TFT).** Requires PyTorch, pytorch-forecasting and Lightning; trains on its *own* synthetic data rather than the seed data; promotes only if MAPE < 10%; its inference function calls `load_demand_data(df=...)`, which isn't a valid argument (audit check B2). The code path that actually runs when the model is absent is a 30-day mean ± z·σ fallback.

**PPO reinforcement learning.** 2,000,000 training steps on an environment with a 490-value observation and 224-value action; the inference helper builds a 26-value observation, so the trained policy can't be used (check B1). The path that actually runs is a heuristic: reorder if utilization < 30%, transfer out if > 90%.

Neither fits a free CPU host, and neither has evidence that it beats simpler methods on this data.

## Decision

**Forecasting:** a benchmarked ladder, chosen per evidence with rolling-origin backtests:

1. Seasonal naive (weekly) — the baseline every model must beat
2. Holt-Winters / ETS (statsmodels)
3. LightGBM global model with lag, rolling and calendar features, plus quantile models for P10/P50/P90

Metrics: WAPE, MAE, RMSE, bias, pinball loss, prediction-interval coverage. The output keeps the NEXUS interface (P10…P90 per SKU × warehouse × day). TFT remains an optional local notebook.

**Inventory:** classical, explainable policies:

- Base-stock / (s, S) with safety stock = z · σ_demand-over-lead-time, using lead-time variance from shipment history
- EOQ for order quantity
- OR-Tools min-cost transfer plan between warehouses
- Monte Carlo simulation to compare the NEXUS heuristic (baseline) against the new policy

The NEXUS RL environment is documented as a future research direction.

## Consequences

- Runs in seconds on CPU; precomputed offline, loaded by the app.
- Better interview story: *"I benchmarked against a naive baseline and chose the simplest model that won."*
- We give up deep-learning buzzwords in the deployed path, and say so plainly.
