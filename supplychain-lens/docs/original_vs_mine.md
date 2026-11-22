# How SupplyChain Lens differs from NEXUS-SCCT

This table is updated at the end of every phase. The **Status** column shows only what is actually done. Nothing is marked done in advance.

| Area | NEXUS-SCCT | SupplyChain Lens | Status |
|---|---|---|---|
| Core use case | AI supply chain control tower | Same, positioned as deep-learning-driven decision intelligence | — |
| Audit | — | Reproducible audit of the original (20 checks) and feature audit | ✅ Phase 0 |
| Feasibility | — | Measured CPU benchmark for LSTM, TFT, autoencoder, ONNX serving | ✅ Phase 0b |
| Project setup | Flat upload; can't install or start | Installable package, 3 requirement sets, CI, hardware smoke test | ✅ Phase 1 |
| Data | Synthetic generator; 1 year; not used by the app | Same data model and distributions, reimplemented; 10 defects fixed; 3-year causal simulation; hidden truth kept separate | Planned (Phase 2) |
| Database | PostgreSQL, 4 tables, never seeded | SQLite built from Parquet, ~12 tables | Planned (Phase 3) |
| Statistics | Mann-Whitney drift test | Demand profiling, lead-time variance, CIs, FDR-controlled drift, Monte Carlo percentiles, paired experiment tests | Planned (Phase 4) |
| Demand forecasting | TFT via pytorch-forecasting, never trained, broken inference | Own PyTorch **LSTM** and **compact TFT**, trained and compared with seasonal-naive, ETS and LightGBM | Planned (Phases 4–6) |
| Anomaly detection | — | PyTorch **autoencoder** vs robust z-score, evaluated on labelled synthetic anomalies | Planned (Phase 7) |
| Supplier risk | XGBoost on separate data; generator crashes | **Neural classifier** vs logistic regression vs XGBoost; calibration; expected loss | Planned (Phase 8) |
| Model serving | MLflow registry, PyTorch | Versioned artifact folders, ONNX Runtime, parity tests | Planned (Phases 5–8) |
| Agents | 8 LLM-only nodes, linear chain | 6 tool-grounded agents; 7 deterministic tools; computed confidence; policy in code | Planned (Phase 9) |
| LLM | Claude / GPT-4, paid | Templates (deployed) + Ollama (local) | Planned (Phase 9) |
| Digital twin & simulation | Static view; fixed-delta rerouting | Scenario engine on the same simulation that generated history; Monte Carlo | Planned (Phase 10) |
| Optimization | PPO RL (incompatible inference), VRP | Base-stock/EOQ + OR-Tools transfers + mode choice; optional PPO | Planned (Phases 10–11) |
| UI | Single React file, static/random data | Streamlit + Plotly, 10 pages including a Deep Learning Lab, data-driven | Shell only (Phase 1) · full build Phase 12 |
| Business impact | Unsupported numbers in README | Simulated baseline-vs-AI experiment with CIs, clearly labelled | Planned (Phase 13) |
| Testing | 32 mostly mocked tests; broken imports | Unit, data-contract, leakage, model-shape, parity, agent and smoke tests in CI | Started (Phase 1) |
| Documentation | README, PRD, architecture diagrams | Own README, ADRs (10), feature audit, component guide, setup guide | Started (Phases 0–1) |
| Deployment | Docker / K8s (paid to host) | Streamlit Community Cloud (free) | Planned (Phase 14) |

## What is *not* original here

- The product concept: an AI control tower with specialized agents, a digital twin, a supplier risk matrix, a copilot and human-in-the-loop approvals.
- The synthetic data model: entities, reference tables and distributions.
- The decision schema and auto-approval policy idea.
- The idea of using a TFT for probabilistic demand forecasting and PPO for inventory (NEXUS designed both but never trained them).
- The KPI framework and the personas that inspired our PRD.

All of these come from NEXUS-SCCT by Syed Zain Ahmed and are credited in [`ATTRIBUTION.md`](../ATTRIBUTION.md).
