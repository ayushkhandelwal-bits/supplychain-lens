# Change impact analysis — deep-learning-first upgrade ("SupplyLens")

*Change request received 27 Oct 2026. Covers sections A–J of the request. Nothing in the code or the Phase 0 documents changes until you approve; §K lists exactly what will change.*

Items from the change request that the existing plan **already covers** are not repeated in detail. They're listed once in §A.3 with where each is handled.


> **Decisions taken (27 Oct 2026):** project name **SupplyChain Lens** (chosen instead of "SupplyLens" after the trademark check in §A.4) · own compact TFT · ONNX Runtime for hosted inference · conflicts C1–C3 approved.
> This document is kept as the record of the change request; "SupplyLens" and `supplylens/` below are the placeholder names used at the time. The implemented package is `supplychain_lens/`.

---

## A. Current status

### A.1 Completed

**Phase 0 — audit and architecture (done 3 Oct 2026)**, delivered as `nexus-control-tower-phase0.zip`:

- `audit/verify_findings.py`, a 20-check reproducible audit of NEXUS-SCCT (20 of 20 confirmed)
- Architecture A (original) and B (₹0) diagrams
- ADR-0001 to ADR-0007
- Reuse matrix, component guide, originality table, attribution, and a draft license request

No application code has been written. Nothing needs to be thrown away. Three Phase 0 decisions change because of this request (§A.4).

### A.2 Not yet confirmed by you

- Unzipping, running the audit and the first commit
- Repo name and Python version
- Posting the license request

### A.3 Already covered (no new work, just carried forward)

| Request item | Already handled by |
|---|---|
| Continue from NEXUS-SCCT; don't replace the project | Phase 0 scope and all ADRs |
| Inspect license and preserve attribution; don't claim others' work | ADR-0001, `ATTRIBUTION.md`, reuse matrix |
| Reuse NEXUS data and generator; synthetic only where needed; data dictionary | ADR-0007 (`nexus_v1` / `nexus_v2` modes) |
| ₹0 budget; no paid APIs, GPUs, hosting or maps | ADR-0002, ADR-0004, Architecture B |
| Agents call deterministic tools; LLM never invents numbers; works with no LLM; Ollama locally | ADR-0004, ADR-0005 |
| Seasonal-naive and ETS baselines; chronological backtests; WAPE, MAE, RMSE, bias | ADR-0006, roadmap Phase 6 |
| Logistic regression vs boosting; calibration; class balance | Roadmap Phase 7 |
| NetworkX graph is analytics, not ML | Component guide §9 |
| Monte Carlo and what-if scenarios; baseline vs proposed policy | Roadmap Phases 10 and 14 |
| "Simulated" labelling; no fabricated metrics | Pre-phase analysis, original-vs-mine table |
| Training offline, artifacts loaded by the app; no retraining on page load | Architecture B rule 3 |
| Fintech-inspired UI without copying Groww | ADR-0003 |
| PRD, personas, user stories, MVP, roadmap | Roadmap (PRD was planned; now gets a deep-learning angle) |
| Stop after every phase | Working agreement |

### A.4 Conflicts with Phase 0 decisions (need your approval)

| # | Phase 0 decision | Change request | Proposed resolution |
|---|---|---|---|
| C1 | ADR-0006: drop TFT and PPO; forecast with ETS and LightGBM | Deep learning is the core: LSTM, TFT, autoencoder, neural classifier | **Supersede ADR-0006** with ADR-0008, "deep-learning-first model portfolio". Keep the baselines (they're how we prove the models add value). Keep LightGBM as a strong *non-deep* benchmark: it's honest, and it may win |
| C2 | ADR-0005: 5 agents (Risk Sentinel, Demand, Inventory, Logistics, Decision) | 6: Demand Intelligence, Anomaly Investigation, Supplier Risk, Inventory, Simulation, Decision Orchestrator | **Adopt the 6.** Logistics becomes tools used by Supplier Risk and Simulation. KPI monitoring becomes a check inside the Orchestrator |
| C3 | Name `nexus-control-tower`; 9 dashboard pages | Name **SupplyLens**; 10 pages | Pages: adopt the 10 (§H). Name: **see the warning below** |

### ⚠️ Name check — "SupplyLens" is already a trademark in supply chain analytics

I checked before adopting the name, as the request asks. Sources: [Lytica — SupplyLens™ Pro Platform](https://lytica.com/supplylens-pro-platform/), [nirajxdev/SupplyLens](https://github.com/nirajxdev/SupplyLens), [monkastro/supplylens](https://github.com/monkastro/supplylens).

- **SupplyLens™ Pro** is a commercial supply-chain analytics platform by **Lytica**. It covers spend analytics, supplier risk intelligence and should-cost for electronics procurement, and carries the ™ mark. That's the same domain as this project.
- Two public GitHub repositories are already named SupplyLens (`nirajxdev/SupplyLens`, inventory and demand forecasting; `monkastro/supplylens`, supply chain analytics). Repo names only need to be unique within *your* account, so you can still create `<you>/SupplyLens`. It just won't be distinctive.

Using another company's trademarked product name for a supply-chain product can cause confusion, and a recruiter searching the name will find Lytica first. I'm not a lawyer. A personal, non-commercial portfolio project is low-risk, but not zero-risk. Options:

1. **Use "SupplyLens" anyway**, with a README line: *"Personal portfolio project; not affiliated with Lytica's SupplyLens™."*
2. **Use a distinct name** that keeps the idea, for example *SupplySight*, *ChainLens*, *LensChain* or *Tessera SCCT*. Each would still need a check.
3. **Pick your own name.** I'll run the same check before we commit to it.

**My recommendation is option 2.** Everything below says "SupplyLens" as a placeholder; renaming is a single search-and-replace in Phase 1.

---

## B. NEXUS-SCCT feature and deep-learning audit

### B.1 Status of every advertised model

Status labels follow the change request. Evidence IDs refer to `audit/AUDIT_REPORT.md`.

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

### B.2 Status of other advertised platform features

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

Now maintained in [`docs/feature_audit.md`](feature_audit.md).

---

## C. Data readiness report

Measured on the original `seed_data.py` output (default settings, seed 42).

| Property | Value | Why it matters for deep learning |
|---|---|---|
| Demand history | 365 days × 100 SKUs × 4 warehouses = 400 series, 146,000 rows | Only **one** annual cycle; yearly patterns can't be learned *and* validated |
| Series diversity | Coefficient of variation 0.26–0.31 across all series | Every series has the same shape at a different scale; a global model has little to learn beyond a seasonal-naive forecast |
| Price | Re-drawn every day (D1) | Can't be used as a feature; price sensitivity can't exist |
| Promotions | None | No known-future driver, which removes most of TFT's advantage |
| Zero demand | 0% | Simplifies metrics (MAPE is defined), but unrealistic |
| Training windows (56-day lookback, 28-day horizon, last 112 days held out) | 170 per series, 68,000 total | Enough volume for a small LSTM; not enough *variety* |
| Inventory, purchase orders, lead-time history | None | No input for the autoencoder, and no realistic inventory state |
| Shipments | 300, a snapshot over the last 45 days; median 5.5 per supplier; delivered shipments never late (D3) | Too thin for supplier-level sequences |
| Disruptions | 50 events over 60 days, random suppliers regardless of geography (D6); 3 active | Far too few, and incoherent, for a classifier |
| Network | 50 suppliers, 12 warehouses, no SKU-supplier links (D5) | No graph to learn on |

| Model | Verdict on the original data | Main gaps |
|---|---|---|
| **LSTM forecaster** | ⚠️ *Trainable but not meaningful* | One year; identical series shapes; noisy price; no promotions |
| **TFT** | ❌ Insufficient | Needs known-future covariates (promotions, planned price, holidays) and multiple seasons |
| **Autoencoder** | ❌ Insufficient | No daily operational panel (inventory, lead time, delay, utilization) and no ground-truth anomalies to evaluate against |
| **Disruption classifier** | ❌ Insufficient | Needs a supplier × day history with leading signals and a label based on future events. NEXUS's separate 5,000-row risk set is cross-sectional and crashes |
| **GNN** | ❌ Insufficient | No edges; about 62 nodes in one graph |
| **RL** | ✅ Needs a simulator, not historical data | The original environment is disconnected from the seed data; ours will reuse the digital-twin simulator |

**Conclusion:** the original data can't support a credible deep-learning project. We **extend** it, keeping every NEXUS entity, reference table and distribution, rather than replacing it (§E).

---

## D. Recommended model portfolio

The aim is the smallest set that makes this a credible deep-learning project, where every model feeds a decision.

### Core (must build)

| ID | Model | Task | Feeds which decision | Baselines it must beat |
|---|---|---|---|---|
| **A** | **Seq2seq LSTM, probabilistic** (PyTorch) | 28-day daily demand per SKU × warehouse; P10/P50/P90 via quantile (pinball) loss; learned SKU and warehouse embeddings | Replenishment quantity, safety stock, stockout probability | Seasonal-naive (weekly), Holt-Winters ETS, LightGBM |
| **B** | **Compact Temporal Fusion Transformer** (PyTorch, own implementation) | Same task: static covariates, known-future promotions and calendar, variable-selection networks, gated residual networks, interpretable multi-head attention | Same, plus "which drivers matter" | LSTM and the baselines above |
| **C** | **Autoencoder** (PyTorch) | Reconstruct 14-day windows of daily operational features per supplier and per warehouse; anomaly score = reconstruction error; threshold set on validation data | "Investigation required" alerts | Rolling robust z-score (median/MAD); Isolation Forest (optional) |
| **D** | **Neural disruption classifier** (PyTorch; MLP on engineered features, with an optional GRU over 28-day history) | P(supplier disruption starts within the next 14 days) | Supplier mitigation, buffer stock | Logistic regression; XGBoost |

That's **four neural networks**. Two of them, LSTM and TFT, compete on the same task, so the Deep Learning Lab page has a real comparison to show.

**Why build our own compact TFT instead of importing pytorch-forecasting?** The request asks not to "merely import a TFT package". A ~250-line implementation of its main parts lets you explain every layer: variable selection, gated residual networks, static enrichment, LSTM encoder-decoder, and masked interpretable attention. pytorch-forecasting 1.8.0 *is* compatible (Python 3.10–3.14, torch 2.x, Lightning < 2.7). We can use it as an optional cross-check, not as the main implementation. **Your call** (decision D3 in §J).

### Optional (only after the core works)

| ID | Model | Condition for building it | Default |
|---|---|---|---|
| **E** | GNN (PyTorch Geometric, GraphSAGE/GCN) for supplier node risk | Only if a pre-registered test shows it beats the classifier from D with neighbour features added by hand. With ~160 nodes in one graph, that's unlikely | **Deferred**, with a written explanation |
| **F** | PPO (Stable-Baselines3, MLP policy) for replenishment | Small environment (1 warehouse × a few SKUs) built on our digital twin; ≥ 30 evaluation episodes on common random numbers vs the base-stock policy; report cost, service level, stockouts and their spread | Build if time allows; report honestly if it loses |

---

## E. Synthetic data requirements

### E.1 Principles

1. **Extend, don't replace.** All NEXUS entities, the 12-country reference table, the 10 categories, the 6 disruption types, the 8 carriers and the 12 warehouse locations are kept. `nexus_v1` mode reproduces the original output.
2. **One causal simulation, not independent tables.** A daily loop drives everything: latent supplier health → disruptions → lead times and capacity → receipts → inventory → fulfilled demand and stockouts → KPIs. A disruption at a supplier therefore shows up consistently in its shipments, its SKUs' inventory and their stockouts.
3. **Hidden truth is stored apart from observable data** (a separate `truth` schema). Feature code can't import it, and a test enforces that. This prevents label leakage, the "synthetic shortcut" the request warns about.
4. **Fixed seed, fixed calendar and a configuration file** (`config/data_v2.yaml`). The same config always produces byte-identical data.
5. **Not stores.** NEXUS has no store layer. Its warehouses are named Hub, DC and Gateway, which we use as facility types, and warehouses are the demand nodes. Adding stores would multiply the data without adding a decision.

### E.2 Time period and split (proposed)

- **Calendar:** 1 Jan 2023 → 31 Dec 2025 (1,096 days, since 2024 is a leap year; 3 Black Fridays). *(Corrected from "1,095" during Phase 1; the benchmark below used 1,095 days, which makes no practical difference.)*
- **Snapshot date ("today" in the app):** 31 Dec 2025.
- **Split:** train 2023-01-01 → 2025-06-30 · validation 2025-07-01 → 2025-09-30 · **test 2025-10-01 → 2025-12-31**. The test quarter contains peak season, the hardest and most business-relevant period. Rolling-origin backtests run inside train + validation only.

### E.3 Entities: what exists, what's added

| Entity | NEXUS original | SupplyLens v2 | Change |
|---|---|---|---|
| `suppliers` | 50 | 50, same fields, plus `health_score` (renamed from `risk_score`, D9), `capacity_units_per_day` | Fixed + extension |
| `warehouses` | 12 | 12, plus `facility_type` (Hub/DC/Gateway, derived from the original names) | Extension |
| `products` (SKU master) | missing | 100 SKUs: category, `unit_cost`, `list_price` (constant, D1), primary and backup supplier (category-matched, D5), shelf weight/volume | **New** |
| `sku_warehouse` (assortment) | implicit (4 warehouses) | Each SKU stocked in 4 warehouses, **spread across all 12** (D4); reorder point, order-up-to level, capacity share | **New** (fixes D4) |
| `promotions` | missing | Planned promotion calendar per SKU (~4% of days, 10–30% discount); known in advance, so a legitimate future input | **New** |
| `demand_daily` | 146k rows, 1 year | **438k rows**: true demand, sales (censored by stockouts), price, promo flag, holiday flags. Same NEXUS formula (trend, weekday, monthly sine, Black Friday ×3.4, Cyber Monday ×2.8, lognormal noise), plus product-specific profiles and promotion/price effects | Fixed + extension |
| `purchase_orders` | missing | One row per replenishment: order date, qty, supplier, promised and actual receipt (≈ 25–40k rows) | **New** |
| `shipments` | 300 snapshot | One per PO over 3 years, same columns; delays driven by supplier health and disruptions (fixes D2, D3) | Fixed + extension |
| `inventory_daily` | missing | On hand, on order, lost sales and utilization per SKU × warehouse × day (438k rows), from the replenishment simulation | **New** |
| `disruptions` | 50 events / 60 days | ≈ 150–250 events / 3 years from a hazard process: geography- and category-matched (D6), regionally correlated (one typhoon hits several suppliers), effects on capacity and lead time | Fixed + extension |
| `kpi_daily` | 90 days with a built-in trend (D7) | **Derived** from the simulation (fill rate, OTIF, utilization…). The original trend file is kept only in `nexus_v1` | Fixed |
| `truth.supplier_daily_latent` | — | Hidden health and hazard per supplier × day (never a feature) | **New** (evaluation only) |
| `truth.injected_anomalies` | — | ≈ 200 labelled anomalies (demand spike, lead-time jump, inventory shrink, utilization surge, delay cluster), each with entity, window and type | **New** (evaluation only) |

### E.4 Labels

| Label | Definition | Leakage guard |
|---|---|---|
| `disruption_next_14d` (supplier × day) | 1 if a disruption affecting that supplier **starts** in (t, t+14] | Features use data ≤ t only; tested by shifting the label and checking that performance collapses to chance |
| Anomaly ground truth | Membership in `truth.injected_anomalies` windows | The autoencoder never sees labels; labels are used only for evaluation and to pick thresholds on validation data |
| Forecast target | `demand_daily.true_demand` (and `sales` for a censored-demand comparison) | Scalers fit on train only; windows never cross the train/validation boundary |

### E.5 Data products in each phase

`docs/data_dictionary.md` (every column marked Original / Fixed / Extension / Derived / Truth), `config/data_v2.yaml`, `reports/data_quality.md` (row counts, ranges, missing values, foreign-key checks, distribution checks against the NEXUS formulas), and a dashboard banner: *"All data is synthetic, generated from the NEXUS-SCCT data model."*

---

## F. Architecture changes

```text
TRAINING ENVIRONMENT (your laptop, CPU)              INFERENCE ENVIRONMENT (Streamlit Cloud, CPU)
───────────────────────────────────────              ─────────────────────────────────────────────
config/*.yaml                                        app/  (10 pages)
   ↓                                                    ↓
supplylens/data/generator.py  →  data/supplylens.db     supplylens/serving/ (loads artifacts once, cached)
   ↓                                    ↓                    ↓
supplylens/features/ (no-leak feature builders)         tools: forecast_demand(), detect_anomalies(),
   ↓                                                           score_disruption_risk(),
supplylens/models/  lstm.py · tft.py · autoencoder.py          calculate_stockout_probability(),
                    classifier.py · baselines.py               simulate_supplier_disruption(),
   ↓                                                           compare_inventory_policies(),
supplylens/training/ train.py · early stopping ·               calculate_business_impact()
                     checkpoints · seeds                     ↓
   ↓                                                    agents/: Demand Intelligence · Anomaly
supplylens/evaluation/ backtests · metrics ·                    Investigation · Supplier Risk ·
                       calibration · reports                    Inventory · Simulation · Decision
   ↓                                                            Orchestrator → policy.py → approval
artifacts/<model>/<version>/                             ↓
   model.onnx (or .pt) · config.json · scaler.json ·     llm/: templates (default) · Ollama (local)
   metrics.json · loss_curve.csv · predictions.parquet
   · model_card.md · SHA-256 checksum   ────────────────────────────────────→  (committed, small)
```

Key changes compared with Architecture B:

1. **A model registry folder** (`artifacts/`) replaces MLflow. Each model version stores its config, scaler, metrics, loss curve, test predictions and a model card. The dashboard reads *only* these files, so every chart and metric it shows was produced by an actual run.
2. **Serving format.** Train in PyTorch, export to **ONNX**, and serve with **ONNX Runtime** (a small CPU library), so the hosted app doesn't need PyTorch. Fallback: load `.pt` with the PyTorch CPU build. Measured sizes and speeds are in §G.
3. **Agents: 6, each with a distinct job** (conflict C2). The Orchestrator applies the approval policy in code. Confidence comes from model outputs (calibrated probability, forecast interval width), never from an LLM.
4. **Digital twin = the data generator's simulation engine**, reused at inference time with scenario parameters and model forecasts as inputs. One engine is used for both history and what-if analysis, which keeps them consistent.
5. **Uncertainty is split in two.** Forecast uncertainty comes from P10–P90 sampling. Scenario uncertainty comes from Monte Carlo over the assumption ranges (for example, lead-time +3 to +7 days). Both appear separately in scenario results.

---

## G. Free-compute feasibility

*Measured in my cloud workspace (2 vCPU, 7 GB RAM, no GPU, PyTorch 2.14.1), roughly comparable to a mid-range laptop, at the proposed scale (400 series × 1,095 days, 56-day lookback, 28-day horizon). The models were untuned sizing prototypes on stand-in data, not the project's code. Rerun with `python audit/bench_feasibility.py`; raw output is in `audit/bench_results_2vcpu.json`.*

### G.1 Measured results

| Model (prototype size) | Parameters | Weights file | Training cost | Inference for all 400 series × 28 days |
|---|---|---|---|---|
| **LSTM** seq2seq, hidden 64, SKU/warehouse embeddings, 3 quantiles | 41,235 | 165 KB | 7.7 s per 20k windows → **≈ 20 s/epoch** at stride 7 (51k windows); ≈ 2.3 min/epoch on all 360k windows | **33 ms** (PyTorch) |
| **Compact TFT**, d=32, 4 heads (variable selection, GRNs, LSTM enc/dec, masked attention) | 97,444 | 429 KB | 26.5 s per 20k windows → **≈ 68 s/epoch** at stride 7; ≈ 8 min/epoch on all windows | **149 ms** |
| **Autoencoder**, dense 12→32→4→32→12, 54,750 rows (50 suppliers × 3 years) | 1,104 | < 10 KB | **0.4 s/epoch** | negligible |
| **Holt-Winters ETS** baseline (statsmodels) | — | — | ≈ **2 min** to fit all 400 series | — |
| **LSTM exported to ONNX**, run in ONNX Runtime | — | 265 KB | — | **37 ms**; max difference vs PyTorch 4.8 × 10⁻⁷ |

Both forecasters' training loss fell from about 0.55 to 0.06 (pinball loss, scaled units) within a single pass over 20k windows. That shows the pipeline learns; it is **not** an accuracy result. Accuracy is measured in Phases 4–6 against the baselines.

### G.2 What this means

| Item | Estimate (from G.1) | Feasible on CPU? |
|---|---|---|
| LSTM, 30 epochs, stride 7, early stopping | ≈ 10 min | ✅ |
| LSTM tuning, ~8 configurations | ≈ 1–1.5 h total (run once) | ✅ |
| TFT, 30 epochs, stride 7 | ≈ 35 min | ✅ (the request's "if feasible" condition is met at this scale) |
| Autoencoder + classifier | minutes | ✅ |
| PPO (optional), small environment | tens of minutes to a few hours, depending on environment size | ⚠️ Optional; size decided in Phase 11 |
| GPU | Not needed | — |

**Your laptop may be faster or slower** than my 2-vCPU workspace. Phase 1 includes a 30-second smoke test that measures it, and model sizes are fixed after that.

### G.3 Hosting constraints

| Constraint | Finding | Design response |
|---|---|---|
| **PyTorch install size on Linux** | The PyPI build for Linux (which Streamlit Cloud uses) bundles CUDA: **1.2 GB for torch + ≈ 2.9 GB of GPU libraries ≈ 4.1 GB**. On Windows, `pip install torch` gets the CPU build (a few hundred MB) | Serve with **ONNX Runtime (66 MB installed)** in the hosted app, so PyTorch isn't a deployment dependency. Use PyTorch only for training |
| Streamlit Community Cloud memory | Reported at roughly 690 MB–2.7 GB per app; apps sleep after 12 hours without traffic and take ~30–60 s to wake (secondary source, 2026; to be re-verified against Streamlit's docs in Phase 14) | Load small artifacts once and cache them; no training in the app; keep peak memory well under 1 GB |
| Data size | 2 × 438k-row daily tables ≈ **11 MB as Parquet** (upper bound, random values) or ≈ 61 MB as indexed SQLite | Commit compressed Parquet; build the SQLite file on first app start (≈ 5 s, cached) |
| Model artifacts | All four models together < 1 MB | Commit them to the repo |
| ONNX caveat | The legacy exporter warns that LSTM exports with a fixed batch size can fail at other batch sizes | Export with dynamic batch axes and test batch sizes 1 and N (Phase 5 test) |
| Local LLM | Not possible on the free host | Templates online; Ollama on your laptop (ADR-0004, unchanged) |

---

## H. Dashboard changes

The 9 planned pages become the 10 requested. Mapping:

| # | Page (requested) | Replaces / merges | New charts and metrics for deep learning |
|---|---|---|---|
| 1 | **Executive Control Tower** | Overview | Service level, stockout rate, inventory value, open alerts, predicted disruption exposure (Σ probability × value), 28-day demand outlook, **model health** (latest test WAPE, drift flag, artifact version), top 3 recommendations |
| 2 | **Deep Learning Lab** ⭐ | new | Model cards (architecture diagram, parameter count, inputs and outputs), dataset summary and split dates, **train/validation loss curves**, actual vs predicted, metric table (LSTM vs TFT vs baselines, by horizon), training time, **measured inference latency**, artifact version and checksum, badge for the model selected for inference |
| 3 | **Demand Intelligence** | Demand & Forecasts | History + P50 forecast + P10–P90 band; filters for SKU, warehouse, category and date; error by SKU and segment; STL trend and seasonality; demand anomalies overlaid |
| 4 | **Anomaly Detection** | new | Anomaly table (score, threshold, entity, window); reconstruction error over time with threshold line; per-feature deviation (which inputs reconstructed worst); records for the flagged window; status (Detected → Investigating → Confirmed / Dismissed) |
| 5 | **Supplier Risk Intelligence** | Suppliers & Risk | Ranked disruption probability; reliability diagram; precision/recall at the chosen threshold; drivers (permutation importance, *labelled associational*); affected SKUs and warehouses; exposure ₹/$ |
| 6 | **Supply Network** | Logistics & Network | Plotly geo map and network graph (suppliers → SKUs → warehouses); centrality and sole-source flags *labelled as graph analytics*; separate GNN tab only if model E is built |
| 7 | **Digital Twin & Scenario Simulator** | Scenarios | Sliders (demand %, lead time +days, supplier capacity %, warehouse capacity %, supplier offline, transport delay); baseline vs scenario distributions for stockouts, service level, inventory, cost, revenue exposure, recovery time; forecast uncertainty shown separately from scenario uncertainty |
| 8 | **AI Decision Room** | Decision Center + Ask NEXUS | Alert → model evidence → recommended and alternative actions with computed cost and service impact and uncertainty → approve/reject (reason required) → audit trail; "Ask SupplyLens" panel |
| 9 | **Business Impact** | Impact & Methods (part) | Baseline vs proposed policy: stockout rate, service level and cost differences with 95% CIs; evaluation period; assumptions; every number labelled **Simulated impact** |
| 10 | **Model Evaluation** | Impact & Methods (part) | Split dates, loss curves, actual vs predicted, residual plots, error by horizon, calibration, confusion matrix at the threshold, limitations |

Rule: a chart appears only if the artifact behind it exists. A missing artifact shows "not trained yet", never a placeholder number.

---

## I. Updated roadmap

The old 19 phases and the requested Phases A–O merge into **16 phases**. Phase 0 is not redone.

| # | Phase | Request phase | Old phase(s) | Main outputs |
|---|---|---|---|---|
| 0 | Audit and architecture | A | 0 | ✅ Done; feature audit (§B) added on approval |
| 0b | Data readiness and change impact | B | — | ✅ This document |
| **1** | **Setup and rename** | — | 1 | Repo scaffold under the chosen name, Windows venv, PyTorch CPU, VS Code, CI, ADR updates |
| 2 | Synthetic data v2 | C | 2 | Generator, config, data dictionary, data-quality report, leakage tests |
| 3 | Database, pipeline and artifact registry | — | 3 | SQLite, feature builders, `artifacts/` format |
| 4 | Statistics and forecasting baselines | D | 4, 5 | Demand profiling, seasonal-naive, ETS, LightGBM, backtest harness, metrics |
| 5 | PyTorch LSTM | E | 6 | Dataset/DataLoader, training loop, early stopping, checkpoints, quantiles, evaluation vs baselines |
| 6 | Compact TFT | F | — | Own implementation, attention and variable-importance outputs, comparison |
| 7 | Autoencoder anomaly detection | G | — | Training on normal windows, threshold on validation data, precision/recall/FPR vs MAD baseline |
| 8 | Disruption classifier and risk analytics | H | 7 | MLP (+GRU), LR, XGBoost; PR-AUC, calibration; expected loss; cascade |
| 9 | Tools and agents | I | 8, 9 | 7 deterministic tools, 6 agents, policy, narrator, Ollama client |
| 10 | Digital twin, scenarios and optimization | J | 10, 11 | Scenario engine, Monte Carlo, base-stock/EOQ, OR-Tools transfers |
| 11 | Optional PPO / GNN | K | — | Only if it passes the conditions in §D |
| 12 | Dashboard (10 pages) | L | 12, 13 | Theme, components, pages |
| 13 | Evaluation and business impact | M | 14 | Baseline vs AI experiment, model comparison report |
| 14 | Testing, performance and free deployment | N | 15, 16, 17 | Full test suite, cold-start check, Streamlit Cloud |
| 15 | Portfolio | O | 18, 19 | README, diagrams, screenshots, demo, resume, LinkedIn |

The same stop-after-each-phase format continues.

---

## J. Next action and decisions needed

**Single next task:** Phase 1, the repository scaffold and environment under the chosen name, with PyTorch CPU installed and a 30-second smoke test that trains a tiny LSTM on your machine. That smoke test gives *your* hardware's timing, which then sets the model sizes for Phases 5–8.

Before Phase 1, please answer:

| # | Decision | Options | My recommendation |
|---|---|---|---|
| D1 | Project name | SupplyLens with a disclaimer · a distinct name · your own | A distinct name (trademark note in §A.4) |
| D2 | Your hardware | CPU model, RAM, GPU (if any), Python version | — |
| D3 | TFT implementation | Own compact TFT · pytorch-forecasting · both | Own compact TFT; pytorch-forecasting as an optional cross-check |
| D4 | Hosted serving runtime | ONNX Runtime · PyTorch CPU | ONNX Runtime (see §G) |
| D5 | Approve conflicts C1–C3 | yes / change | Yes |

## K. What changes in existing documents once approved

| File | Change |
|---|---|
| `docs/adr/0006-replace-tft-and-ppo.md` | Status → *Superseded by ADR-0008* (kept for history) |
| `docs/adr/0008-deep-learning-first-portfolio.md` | **New**: §D and §G decisions |
| `docs/adr/0005-tool-grounded-agents.md` | Revised: 6 agents, 7 named tools |
| `docs/adr/0007-same-data-model-with-documented-fixes.md` | Extended with §E (3-year calendar, causal simulation, truth schema) |
| `docs/adr/0009-training-vs-inference-environments.md` | **New**: artifact registry, ONNX serving |
| `docs/architecture.md` | Architecture B redrawn with the training/inference split |
| `docs/feature_audit.md` | **New**: §B |
| `docs/original_vs_mine.md` | Deep-learning rows added; name updated |
| `docs/reuse_matrix.md` | `tft_model.py` and `rl_optimizer.py` → REIMPLEMENT (reference design) |
| `README.md`, `ATTRIBUTION.md` | Name; deep-learning positioning |
