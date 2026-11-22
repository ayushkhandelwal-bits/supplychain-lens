# NEXUS-SCCT feature audit

What the reference repository ([ahmedzainsyed/nexus-scct](https://github.com/ahmedzainsyed/nexus-scct) @ `7ada9fa`) advertises, and what actually exists. Created 2026-10-27 from the [change impact analysis](change_impact_deep_learning.md) §B. Evidence IDs (A1, B2, E1…) refer to [`audit/AUDIT_REPORT.md`](../audit/AUDIT_REPORT.md).

**Status labels:** Implemented and tested · Implemented but unverified · Partially implemented · Documented but not implemented · Missing · Requires replacement for free deployment.

## 1. Status of every advertised model

| Advertised model | Training code | Inference code | Trained artifact in repo | Neural network? | Status |
|---|---|---|---|---|---|
| **Temporal Fusion Transformer** (`tft_model.py`) | Yes: pytorch-forecasting with Lightning, quantile loss, early stopping, MLflow | Yes, but calls `load_demand_data(df=…)`, which doesn't exist (B2) | None | Yes | **Implemented but unverified**; inference **partially implemented**. Trains on its own generated data, not the seed data |
| **LSTM sequence models** (README architecture box) | None | None | None | — | **Documented but not implemented** |
| **Prophet baseline** (README) | None | None | None | No | **Documented but not implemented** |
| **XGBoost supplier risk** (`supplier_risk_model.py`) | Yes | Yes (MLflow or heuristic fallback) | None | No | **Partially implemented**: training-data generator crashes (E1); patched AUC 0.77 is below its own 0.88 gate (E2) |
| **PPO inventory RL** (`rl_optimizer.py`) | Yes: custom Gymnasium env and SB3 PPO with an MLP policy (512-512-256) | Builds a 26-value observation for a 490-value policy (B1) | None | Yes (MLP policy) | **Partially implemented**; the code that actually runs is a utilization-threshold heuristic |
| **Mann-Whitney drift detector** (`tft_model.py`) | n/a | Yes | n/a | No | **Implemented but unverified** |
| **NetworkX cascade** (`supplier_risk_model.py`) | n/a | Yes | n/a | No | **Implemented but unverified**; the seed data has no supplier-to-product graph to run it on (D5) |
| **OR-Tools VRP** (`route_optimizer.py`) | n/a | Yes | n/a | No | **Implemented but unverified**; not connected to data |
| **Multimodal mode selection** (`route_optimizer.py`) | n/a | Yes | n/a | No | **Implemented and tested** (ran its demo successfully) |
| **Autoencoder / anomaly detection** | — | — | — | — | **Missing** (never mentioned) |
| **Graph Neural Network** | — | — | — | — | **Missing** ("GraphDB" is mentioned; no code) |
| **Neural disruption classifier** | — | — | — | — | **Missing** |

**Bottom line:** NEXUS-SCCT has **no trained deep-learning model and no working deep-learning inference path**. The TFT and PPO code is a useful reference design. Every deep-learning result in SupplyLens will be new work, which is a clean story for interviews.

## 2. Status of other advertised platform features

| Feature (README / PRD) | Status |
|---|---|
| 8-agent LangGraph orchestration | Implemented but unverified; **requires replacement for free deployment** (paid Claude API) |
| AI Copilot "ARIA" | Partially implemented (works only inside a Claude artifact sandbox); requires replacement |
| Kafka, Redis, WebSocket streaming | Implemented but unverified; requires replacement |
| Celery scheduled retraining | Implemented but unverified; requires replacement |
| FastAPI REST API | Partially implemented (can't start, A2; returns mocks, A3) |
| JWT, RBAC, rate limiting | Implemented but unverified (not needed for the demo) |
| Command Center / KPI dashboard | Partially implemented (UI only; static and random values) |
| Digital Twin | Partially implemented (static topology view; no simulation behind it) |
| Inventory Intelligence, EOQ optimizer | Partially implemented (UI only); **EOQ: documented but not implemented** (only named in a prompt and a function label) |
| Supplier Risk Matrix | Partially implemented (UI with hard-coded scores) |
| Event stream | Partially implemented (random events every 4 seconds) |
| Human-in-the-loop approvals and audit trail | Partially implemented (endpoint echoes input; no storage) |
| Monte Carlo disruption modelling | Documented but not implemented (roadmap item) |
| CrewAI, DVC, Prophet, satellite imagery, carbon routing | Documented but not implemented |
| MLflow registry | Implemented but unverified; replaceable by local artifact versioning |
| Docker, K8s, NGINX, Prometheus, CI | Implemented but unverified; requires replacement or drop |
