# Roadmap

16 phases. After each one we stop, review and commit. Updated 2026-10-27 after the deep-learning change request ([change impact analysis §I](change_impact_deep_learning.md)).

| # | Phase | Main outputs | Status |
|---|---|---|---|
| 0 | Audit and architecture | 20-check audit, Architecture A/B, ADR-0001…0007, reuse matrix, attribution | ✅ Done |
| 0b | Data readiness and change impact | Feature audit, data-readiness report, model portfolio, CPU benchmark | ✅ Done |
| **1** | **Setup and rename** | `supplychain_lens` package, requirements, Windows setup, VS Code, CI, hardware smoke test, ADR-0008…0010 | ✅ Done |
| 2 | Synthetic data v2 | Generator (`nexus_v1` / `nexus_v2`), config, data dictionary, data-quality report, leakage tests | ⏭ Next |
| 3 | Database, pipeline and artifact registry | SQLite from Parquet, feature builders, `artifacts/` format | |
| 4 | Statistics and forecasting baselines | Demand profiling, seasonal-naive, ETS, LightGBM, backtest harness, metrics | |
| 5 | PyTorch LSTM (model A) | Dataset/DataLoader, training loop, early stopping, checkpoints, quantiles, ONNX export | |
| 6 | Compact TFT (model B) | Own implementation, attention and variable weights, comparison report | |
| 7 | Autoencoder anomaly detection (model C) | Normal-window training, validation threshold, precision/recall vs MAD | |
| 8 | Disruption classifier and risk analytics (model D) | MLP (+GRU), LR, XGBoost, PR-AUC, calibration, expected loss, cascade | |
| 9 | Tools and agents | 7 tools, 6 agents, approval policy, narrator, Ollama client | |
| 10 | Digital twin, scenarios and optimization | Scenario engine, Monte Carlo, base-stock/EOQ, OR-Tools transfers | |
| 11 | Optional PPO / GNN | Only if the conditions in ADR-0008 are met | |
| 12 | Dashboard (10 pages) | Theme, components, all pages | |
| 13 | Evaluation and business impact | Baseline vs AI experiment, model comparison report | |
| 14 | Testing, performance and free deployment | Full test suite, cold-start check, Streamlit Cloud | |
| 15 | Portfolio | README, diagrams, screenshots, demo, resume, LinkedIn | |
