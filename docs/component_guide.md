# Component guide — understanding NEXUS-SCCT before rebuilding it

For each major component: **what** it is, **why** a control tower needs it, **how** it works, **why NEXUS used** its technology, **what we do** with it, **how it connects**, and the **interview question** you're likely to get, with a strong answer.

---

## 1. Synthetic data generator (`seed_data.py`)

- **What:** A script that produces fake but realistic suppliers, warehouses, shipments, disruptions, daily KPIs and a year of daily demand.
- **Why we need it:** Real supply-chain data is confidential. A generator with known rules lets you build and test everything, and lets you *know the truth* when checking whether a model works.
- **How:** Fixed random seed (42). Suppliers inherit a country risk baseline; on-time rate falls as risk rises. Demand = trend × weekday factor × monthly seasonality × holiday spikes × random noise.
- **Why NEXUS used it:** No dataset exists with suppliers, shipments, disruptions and demand all linked; generating it is the standard approach.
- **What we do:** Reimplement with the same model, fix the 10 defects, and add a fixed snapshot date ([ADR-0007](adr/0007-same-data-model-with-documented-fixes.md)).
- **Connects to:** Database loader → every analytic, model and agent.
- **Interview Q:** *"Your data is synthetic — why should I trust your results?"*
  **A:** "I don't claim real-world accuracy. Synthetic data with known rules lets me check that the models recover patterns I planted — for example, that the risk model ranks suppliers whose true delay rate is higher as riskier. Every business result is labelled simulated. The value is the pipeline and the decision logic, which would run unchanged on real ERP data."

## 2. Database (PostgreSQL → SQLite)

- **What:** Where the operational data and agent decisions live.
- **Why:** One source of truth that every page and agent queries the same way.
- **How (NEXUS):** Async SQLAlchemy with 4 ORM tables, created on startup — but never filled with the seed data.
- **Why NEXUS used PostgreSQL:** Standard for production multi-user apps with concurrent writes.
- **What we do:** SQLite file, plain SQL, ~10 tables ([ADR-0002](adr/0002-sqlite-and-no-streaming-stack.md)).
- **Interview Q:** *"Why not Postgres?"* **A:** "Read-mostly, ~150k rows, single process, free hosting. SQLite is the right tool for that; I kept SQL portable so switching is a connection-string change."

## 3. Backend API (FastAPI)

- **What:** HTTP endpoints for disruptions, inventory, suppliers, KPIs and agent decisions, plus WebSockets.
- **Why:** In a production system, the UI, mobile apps and ERP integrations all talk to one API.
- **How (NEXUS):** Endpoints return hard-coded lists and random numbers (audit A3); `main.py` can't start (A2).
- **What we do:** Not needed for a single Streamlit app; a query module plays the same role. The endpoint list becomes our feature checklist.
- **Interview Q:** *"How would you expose this to other systems?"* **A:** "The query and agent layers are plain Python functions with typed inputs and outputs, so wrapping them in FastAPI is mechanical. I didn't build it because the demo has one client."

## 4. Event streaming (Kafka, Redis pub/sub, WebSockets)

- **What:** A pipeline that pushes events (a port closes, a shipment is late) to consumers in real time.
- **Why:** Control towers are valued for *detecting* problems fast (NEXUS's "mean time to detect").
- **How (NEXUS):** Producers publish to 5 topics; consumers trigger agents; Redis relays to the browser.
- **What we do:** An `events` table plus a replay control that steps through time ([ADR-0002](adr/0002-sqlite-and-no-streaming-stack.md)).
- **Interview Q:** *"What does Kafka give you that a table doesn't?"* **A:** "Durable, ordered, replayable streams that many independent consumers can read at their own pace, at high volume. At my demo's scale a table with timestamps gives the same semantics; I'd add Kafka when there are many producers or consumers."

## 5. Background jobs (Celery)

- **What:** Scheduled tasks: retrain models, detect drift, snapshot KPIs, rescore suppliers.
- **What we do:** One offline pipeline script, optionally run by GitHub Actions on a schedule.
- **Interview Q:** *"How do models stay fresh?"* **A:** "A pipeline regenerates data, retrains, backtests and only publishes artifacts if metrics beat the previous version — the same promotion-gate idea NEXUS used with MLflow."

## 6. Demand forecasting (TFT → our own LSTM and compact TFT, vs baselines)

- **What:** Predicts daily demand per SKU × warehouse for 28 days, with uncertainty (P10–P90).
- **Why:** Inventory and stockout decisions depend on expected demand *and* its spread.
- **How (NEXUS):** Temporal Fusion Transformer — attention-based deep model with quantile loss. In practice the statistical fallback is what runs.
- **Why NEXUS chose TFT:** Strong published results on multi-series forecasting and built-in quantiles.
- **What we do:** Train two neural forecasters ourselves — a seq2seq **LSTM** with quantile outputs and a **compact TFT** implemented layer by layer — and compare both against seasonal-naive, Holt-Winters ETS and LightGBM on the same chronological test quarter (Q4 2025). The winner on WAPE and interval coverage is served ([ADR-0008](adr/0008-deep-learning-first-portfolio.md)).
- **Interview Q:** *"Why build both an LSTM and a TFT?"* **A:** "The LSTM is the simplest sequence model that can learn from history plus known future inputs like promotions. The TFT adds variable selection, static enrichment and attention, which should help when drivers differ between products. Having both on the same test period tells me whether that extra complexity actually pays off — and both have to beat a seasonal-naive baseline first."

## 7. Drift detection

- **What:** Checks whether recent demand looks statistically different from history.
- **How (NEXUS):** Mann-Whitney U per SKU (last 14 days vs older); retrain if > 10% of SKUs have p < 0.05.
- **Issue:** With 400 tests at 5%, ~20 flags appear by chance alone; and daily demand is autocorrelated and seasonal, which breaks the test's assumptions.
- **What we do:** Test de-seasonalized residuals, control the false discovery rate (Benjamini-Hochberg), add PSI.
- **Interview Q:** *"What's the multiple-testing problem?"* **A:** "If you run many tests at α = 0.05, about 5% are false alarms even when nothing changed. Benjamini-Hochberg controls the expected share of false discoveries instead."

## 7b. Anomaly detection (new — autoencoder)

- **What:** Flags unusual operational behaviour: demand spikes, lead-time jumps, abnormal inventory movement, delay clusters, utilization surges.
- **Why:** Problems are cheaper to fix when someone looks early. A control tower needs a "something looks off here" signal before a disruption is confirmed.
- **How (NEXUS):** Not present.
- **What we do:** An autoencoder learns to reconstruct 14-day windows of normal daily features; the reconstruction error is the anomaly score; the threshold is chosen on validation data. It's evaluated against injected, labelled synthetic anomalies and compared with a robust z-score baseline. Alerts read "investigation required", never "disruption confirmed".
- **Interview Q:** *"How do you pick the anomaly threshold without real labels?"* **A:** "On validation data I choose the threshold that hits an alert budget the operations team can handle — for example, the top 1% of windows — then report precision and recall on the synthetic labels at that threshold. In production I'd tune it with analyst feedback on the alerts."

## 8. Supplier risk model (XGBoost → neural classifier vs baselines)

- **What:** Probability a supplier causes a disruption in the next 90 days.
- **How (NEXUS):** 25 features on a separate 5,000-row synthetic set; 5-fold ROC-AUC; promote if ≥ 0.88.
- **Found:** The generator crashes (E1). Patched, XGBoost reaches AUC ≈ 0.77 and logistic regression ≈ 0.79 (E2) — below the gate. The 25 features don't exist for the operational suppliers.
- **What we do:** A supplier × day panel built by our simulation; label = a disruption *starts* in the next 14 days; a neural classifier (MLP, optionally a GRU over 28 days of history) compared with logistic regression and XGBoost on PR-AUC, recall at a fixed alert budget and calibration; rank suppliers by expected loss = probability × exposure.
- **Interview Q:** *"Why did logistic regression beat XGBoost?"* **A:** "The label is drawn from a probability that depends almost monotonically on a few features — a near-linear signal with heavy label noise. A deep tree ensemble mostly fits noise there. It's a reminder to always run a simple baseline."

## 9. Disruption cascade (NetworkX)

- **What:** Follows a failing supplier through the products it supplies to estimate what's affected.
- **How:** Breadth-first search with impact weighted by volume share; betweenness centrality flags critical nodes.
- **What we do:** Same idea, run on the SKU-supplier-warehouse links our generator adds.
- **Interview Q:** *"What makes a supplier 'critical' beyond its risk score?"* **A:** "Exposure: sole-source items, revenue of those items, and position in the network. A low-risk sole source can matter more than a high-risk supplier with alternatives."

## 10. Inventory optimization (PPO RL → classical policies, optional PPO)

- **What:** Decides when and how much to reorder, and whether to move stock between warehouses.
- **How (NEXUS):** PPO trained for 2M steps; inference is incompatible with the trained policy (B1), so a utilization-threshold heuristic runs.
- **What we do:** Safety stock from forecast quantiles and lead-time variability, base-stock policy, EOQ and OR-Tools transfers, all compared in Monte Carlo. Optionally (Phase 11), a small PPO agent trained on our digital twin, evaluated against the base-stock policy over ≥ 30 episodes with common random numbers, and reported even if it loses.
- **Interview Q:** *"When would you use RL here?"* **A:** "When the policy space is too complex for analytical rules — many interacting echelons, capacity limits, non-stationary demand — and you have a trustworthy simulator. I'd first need a classical baseline to beat, which is exactly what I built."

## 11. Route / mode optimization (OR-Tools + scoring)

- **What:** For a shipment, compares sea, air, rail and road by cost, time, CO₂ and on-time probability.
- **How:** Great-circle distance × mode factors, congestion and disruption penalties, weighted score.
- **What we do:** Reimplement mode scoring for our lanes; use OR-Tools for warehouse transfers (a min-cost-flow problem) instead of vehicle routing.
- **Interview Q:** *"Why not VRP?"* **A:** "VRP plans multi-stop vehicle tours — last-mile delivery. A control tower's logistics decisions are lane and mode choices plus inventory moves between nodes, which is a transportation/min-cost-flow problem."

## 12. Multi-agent orchestration

- **What:** Specialized "agents" that each analyse part of a disruption, then a planner that combines them.
- **How (NEXUS):** 8 LLM calls in a fixed chain; numbers and confidence are LLM-generated (A4, A5).
- **What we do:** 6 agents — Demand Intelligence, Anomaly Investigation, Supplier Risk, Inventory, Simulation and a Decision Orchestrator — that call 7 deterministic tools backed by the trained models; approval policy in code ([ADR-0005](adr/0005-tool-grounded-agents.md)).
- **Interview Q:** *"Isn't your 'agent' just a function?"* **A:** "Yes — deliberately. An agent here is a component with a goal, tools and structured output. Making the reasoning deterministic means recommendations are reproducible and auditable, which matters when they move money. The LLM is used where it adds value: explaining and answering questions."

## 13. AI copilot (ARIA → "Ask SupplyChain Lens")

- **What:** Chat for questions like "which suppliers are most at risk?"
- **How (NEXUS):** Browser calls Claude with a hard-coded snapshot of the platform state.
- **What we do:** Questions map to tools; answers are built from tool results; Ollama optionally rephrases ([ADR-0004](adr/0004-llm-optional-templates-and-ollama.md)).
- **Interview Q:** *"How do you stop hallucinated numbers?"* **A:** "The model only sees tool outputs, and I check that every number in its answer appears in those outputs; if not, I show the template answer instead."

## 14. Human-in-the-loop governance

- **What:** Rules for which AI decisions run automatically and which need approval, with an audit trail.
- **How (NEXUS):** Auto-approve if confidence ≥ 0.85, not CRITICAL, value < $5M; rejections need a reason.
- **What we do:** **Keep** the policy concept; compute confidence; persist approvals and the audit trail in SQLite.
- **Interview Q:** *"How did you decide what's automated?"* **A:** "By reversibility and value at stake. Cheap, reversible actions with calibrated high confidence can be automated; expensive or irreversible ones always need a human."

## 15. Dashboard

- **What:** 8 views in NEXUS (Command Center, Agents, Digital Twin, Inventory, Supplier Risk, Event Stream, Copilot, Analytics).
- **What we do:** 10 Streamlit pages in a calm fintech style, with a dedicated Deep Learning Lab ([ADR-0003](adr/0003-streamlit-ui.md)).
- **Interview Q:** *"What should a manager see first?"* **A:** "What's wrong, how much money is exposed, and what to do — in that order. The Overview leads with the top three decisions, not with charts."

## 16. Infrastructure (Docker, K8s, NGINX, Prometheus, CI)

- **What:** Packaging, scaling, routing, monitoring and automated testing for production.
- **What we do:** Keep CI (free on GitHub); drop the rest for the demo; describe it as the scale-up path.
- **Interview Q:** *"How would you take this to production?"* **A:** "Postgres for concurrent writes, FastAPI in front of the same Python modules, a scheduler for the pipeline, containerized, with model and data monitoring. Architecture A is a good target; I built the core it was missing."
