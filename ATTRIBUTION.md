# Attribution and credits

## Reference project

**SupplyChain Lens** is an independent, extended implementation **inspired by and based on the design of NEXUS-SCCT**:

- **Project:** NEXUS — Autonomous Supply Chain Control Tower (NEXUS-SCCT)
- **Author:** Syed Zain Ahmed
- **Repository:** https://github.com/ahmedzainsyed/nexus-scct
- **Version studied:** commit `7ada9fa` (May 2026)
- **License status:** the reference repository has no LICENSE file; its README states Apache-2.0. See [ADR-0001](docs/adr/0001-license-and-reuse-strategy.md).

## What comes from NEXUS-SCCT

Concepts, designs and the data model are reimplemented in our own code, not copied.

| From NEXUS-SCCT | Used here as | Our modules (when built) |
|---|---|---|
| Synthetic data model: entities, reference tables, distributions, seed (`seed_data.py`) | Basis of our generator (`nexus_v1` mode reproduces it) | `supplychain_lens/data/` |
| Product concept: control tower with specialized agents, digital twin, supplier risk matrix, copilot, human-in-the-loop | Product scope | `app/`, `supplychain_lens/agents/` |
| `AgentDecision` schema, decision statuses, auto-approval policy (`orchestrator.py`) | Decision schema and policy | `supplychain_lens/agents/schemas.py`, `policy.py` |
| TFT design for probabilistic demand forecasting (`tft_model.py`) | Reference design for our own compact TFT | `supplychain_lens/models/tft.py` |
| Calendar/holiday features and Mann-Whitney drift detection (`tft_model.py`) | Forecast features, drift monitoring | `supplychain_lens/features/`, `monitoring/drift.py` |
| Supplier-risk heuristics, risk-factor rules, action playbook, cascade simulation idea (`supplier_risk_model.py`) | Rules and cascade | `supplychain_lens/risk/`, `simulation/cascade.py` |
| Gym environment and PPO design; inventory cost parameters; heuristic policy (`rl_optimizer.py`) | Optional RL reference design; simulation assumptions; **baseline** policy | `supplychain_lens/simulation/`, `optimization/` |
| Haversine distance and multimodal scoring idea (`route_optimizer.py`) | Mode selection | `supplychain_lens/optimization/routing.py` |
| KPI framework and personas (`README.md`, `PRD.md`) | Inspiration for our KPIs and PRD | `docs/PRD.md` |

Module files that implement one of these ideas carry a header such as:

```python
# Design adapted from NEXUS-SCCT by Syed Zain Ahmed
# (https://github.com/ahmedzainsyed/nexus-scct), seed_data.py.
# Reimplemented and extended in SupplyChain Lens.
```

## What is not taken from NEXUS-SCCT

- No source code, UI code, styling or documentation text is copied.
- No trained models were taken: NEXUS-SCCT contains none. Every model in this repository is trained here.
- Business-impact figures in the NEXUS README are **not** reused.
- All differences are tracked in [`docs/original_vs_mine.md`](docs/original_vs_mine.md).

## Name

"SupplyChain Lens" is the name of this personal portfolio project. It is not affiliated with SupplyLens™ (Lytica), Amazon Web Services (including the AWS Well-Architected Supply Chain Lens), or SupplyChainLens.com. See [ADR-0010](docs/adr/0010-project-name.md).

## Third-party open-source software

Listed in `requirements.txt`, `requirements-train.txt` and `requirements-dev.txt`, each under its own license. Examples: PyTorch (BSD-3-Clause), ONNX Runtime (MIT), Streamlit (Apache-2.0), Plotly (MIT), pandas (BSD-3-Clause), NumPy (BSD-3-Clause), scikit-learn (BSD-3-Clause), statsmodels (BSD-3-Clause).
