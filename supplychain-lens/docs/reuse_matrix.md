# Reuse matrix — every NEXUS-SCCT file, and what we do with it

Reference: `ahmedzainsyed/nexus-scct` @ `7ada9fa`. Until the repository has a LICENSE file, **no file is copied** ([ADR-0001](adr/0001-license-and-reuse-strategy.md)). "Reuse" below means we reimplement the idea in our own code and credit the source.

*Updated 2026-10-27 for the deep-learning-first plan ([ADR-0008](adr/0008-deep-learning-first-portfolio.md)) and the package name `supplychain_lens` ([ADR-0010](adr/0010-project-name.md)).*

**Strategy labels**

- **KEEP** — concept kept as-is (no meaningful change)
- **REFACTOR** — same behaviour, better structure or quality
- **EXTEND** — original behaviour plus new capability
- **REIMPLEMENT** — our own version, built from understanding the original
- **REPLACE** — dependency removed, free alternative built
- **DROP** — not needed for the ₹0 product

| Original file | Our target (`supplychain_lens/…`) | Strategy | Phase | Notes |
|---|---|---|---|---|
| `seed_data.py` | `data/reference.py`, `data/generator.py` | REIMPLEMENT + EXTEND | 2 | Same entities, reference tables, distributions, seed; `nexus_v1` + `nexus_v2` modes; 3-year causal simulation ([ADR-0007](adr/0007-same-data-model-with-documented-fixes.md)) |
| `core_modules.py` — settings | `config.py` | REIMPLEMENT | 1 | ~10 settings instead of 85; no secrets required |
| `core_modules.py` — ORM tables | `db/schema.sql` | REIMPLEMENT + EXTEND | 3 | Same 4 tables' columns, plus products, assortment, promotions, purchase orders, shipments, inventory, events |
| `core_modules.py` — JWT, RBAC, rate limit | — | DROP | — | Public read-only demo; a role switcher simulates approver vs viewer |
| `main.py`, `all_routes.py` | `db/queries.py` + Streamlit pages | REPLACE | 3, 12 | No HTTP API needed; mocks removed; endpoint list used as a feature checklist |
| `orchestrator.py` — `AgentDecision`, statuses, auto-approve rule | `agents/schemas.py`, `agents/policy.py` | KEEP (concept) + REFACTOR | 9 | Same fields and statuses; confidence computed from model outputs |
| `orchestrator.py` — 8 LLM nodes, prompts | `agents/*.py`, `llm/narrator.py` | REIMPLEMENT | 9 | 6 tool-grounded agents ([ADR-0005](adr/0005-tool-grounded-agents.md)) |
| `supplier_risk_model.py` — 25-feature XGBoost | `models/classifier.py`, `models/baselines.py` | REIMPLEMENT | 8 | Neural classifier (model D) vs logistic regression vs XGBoost, on features from our own tables |
| `supplier_risk_model.py` — heuristic, key-risk rules, action playbook | `risk/rules.py` | KEEP (concept) + EXTEND | 8 | Thresholds tied to our data; actions carry computed impact |
| `supplier_risk_model.py` — NetworkX cascade | `simulation/cascade.py` | REIMPLEMENT + EXTEND | 8 | Runs on real SKU-supplier links; labelled graph analytics, not a GNN |
| `tft_model.py` — TFT (pytorch-forecasting) | `models/tft.py` | REIMPLEMENT | 6 | **Reference design** for our own compact TFT (model B); same quantile idea; trained on our data |
| `tft_model.py` — feature engineering, holiday flags | `features/calendar.py` | REIMPLEMENT | 2, 5 | Same calendar ideas |
| `tft_model.py` — DriftDetector | `monitoring/drift.py` | EXTEND | 4 | + Benjamini-Hochberg, de-seasonalized residuals, PSI |
| `tft_model.py` — statistical fallback (mean ± zσ) | `models/baselines.py` | KEEP (concept) | 4 | One of the forecasting baselines |
| `rl_optimizer.py` — Gym environment + PPO | `simulation/inventory_env.py` (optional) | REIMPLEMENT | 11 | **Reference design** for optional model F; built on our digital twin, not on separate random demand |
| `rl_optimizer.py` — heuristic fallback | `optimization/baseline_policy.py` | KEEP (concept) | 10 | Becomes the **baseline** in the impact experiment |
| `rl_optimizer.py` — cost parameters (holding 25%, transfer cost, order cost) | `config.py` | KEEP (concept) | 10 | Used by the simulation; documented as assumptions |
| `route_optimizer.py` — haversine, mode table, weighted scoring | `optimization/routing.py` | REIMPLEMENT | 10 | Worked when tested; rebuilt for our lanes |
| `route_optimizer.py` — VRP | — | DROP | — | Our problem is lane/mode choice and transfers, not vehicle tours |
| `route_optimizer.py` — `simulate_rerouting` fixed +12% / +18% | `optimization/routing.py` | REPLACE | 10 | Recompute options instead of fixed deltas |
| `streaming_layer.py`, `celery_tasks.py` | `pipelines/build_all.py`, `events` table | REPLACE | 3 | [ADR-0002](adr/0002-sqlite-and-no-streaming-stack.md); task list used as a checklist of offline jobs |
| `SupplyChainControlTower.jsx` | `app/` | REPLACE | 12 | 8 views mapped into 10 pages; no code or styling copied |
| `types-and-api.ts`, `package.json` | — | DROP | — | No JS frontend |
| `PRD.md` | `docs/PRD.md` | REIMPLEMENT | 12–15 | Our own deep-learning-focused PRD; personas inspired by theirs, cited |
| `ARCHITECTURE.md`, `FILE_TREE.md` | `docs/architecture.md` | REIMPLEMENT | 0–1 | Our own diagrams (A = theirs as designed, B = ours) |
| `README.md` | `README.md` | REIMPLEMENT | 1, 15 | Business-impact table **not** reused (not computed anywhere) |
| `test_suite.py`, `pytest.ini` | `tests/` | REIMPLEMENT | 1+ | Tests against real behaviour, not mocks |
| `locustfile.py` | — | DROP | — | Replaced by a small performance check in Phase 14 |
| `ci.yml` | `.github/workflows/ci.yml` | REIMPLEMENT | 1 | Lint + tests only (no image builds or deploys) |
| `docker-compose.yml`, `Dockerfiles`, `deployment.yaml`, `nginx.conf`, `prometheus.yml`, `deploy.sh`, `env.example` | — | DROP | — | Not runnable for ₹0; referenced in docs as Architecture A |

## Copy instructions

Because the license is unconfirmed, every file gets the same instruction for now:

```text
Do not copy this file.

Reason:
  No LICENSE file in the reference repository (audit check A1).

Recommended implementation:
  Use the "Our target" path and strategy from the table above.
  Credit the original file in the module header, e.g.
  # Design adapted from NEXUS-SCCT (github.com/ahmedzainsyed/nexus-scct), seed_data.py
```

If the author adds an Apache-2.0 LICENSE, the candidates that would become **COPY + MODIFY** are: `route_optimizer.py` (haversine, mode scoring), `orchestrator.py` (`AgentDecision` schema), and `tft_model.py` (`DriftDetector`). We'd then follow the Apache-2.0 notice rules in ADR-0001.
