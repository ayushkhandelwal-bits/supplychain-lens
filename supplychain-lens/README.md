# SupplyChain Lens

**Deep learning–driven supply chain resilience & decision intelligence.**

*A portfolio implementation inspired by and extended from [NEXUS-SCCT](https://github.com/ahmedzainsyed/nexus-scct) by Syed Zain Ahmed. See [ATTRIBUTION.md](ATTRIBUTION.md).*

> **All data is synthetic**, generated from the NEXUS-SCCT data model. Every business result is **simulated** in a digital twin and is not real company performance.

## What it does

```text
Synthetic supply network (50 suppliers · 12 countries · 100 SKUs · 12 warehouses · 400 demand series · 2023–2025)
 → deep learning: TFT and LSTM forecasts (P10/P50/P90), autoencoder anomalies, disruption risk
 → ONNX export, served with ONNX Runtime (no PyTorch at inference)
 → 7 deterministic tools → 5 specialist agents → Decision Orchestrator
 → approval policy as code → human approval
 → business impact measured in a digital-twin replay (simulated)
```

## Results (held-out test quarter, Q4 2025)

| Model | Task | Test result | vs best baseline |
|---|---|---|---|
| **Temporal Fusion Transformer** (own PyTorch implementation) | 28-day demand forecast | **WAPE 12.5%**, P10–P90 coverage 79% | Best overall; beats LSTM (13.6%), LightGBM (13.7%), Holt-Winters (25.4%), seasonal naive (28.2%); all gaps significant (paired bootstrap) |
| Seq2seq LSTM | same | WAPE 13.6% | Beats LightGBM narrowly |
| Autoencoder | operational anomalies | 10% precision on injected anomalies at 100 alerts; 6 of 16 events found | Beats a robust z-score at the same alert count (5%, 5 of 16); a z-score at its own threshold is more precise (22%, 23 alerts) |
| MLP classifier | supplier disruption within 14 days | PR-AUC 0.058, ROC-AUC 0.713 | **Does not win**: XGBoost PR-AUC 0.077, logistic regression ROC-AUC 0.747 |

**Digital-twin replay of Q4 2025** (30 replications, same actual demand for both policies): fill rate 86.1% → 92.4%, lost revenue −34%, total cost −14.5%, average inventory value +13%. An ablation without predicted-risk supplier switching cuts total cost by 16.7%, so the forecasts deliver almost all the gain. Intervals reflect supplier-delay randomness only.

Full write-ups: `reports/forecast_comparison.md`, the model cards in `artifacts/*/<version>/model_card.md`, and the project report and slide deck.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # or: pip install torch
python -m pip install -r requirements-dev.txt
python -m pip install -e .

python -m pipelines.build_all --only data   # regenerate the synthetic data (seed 42; needed if you use the shipped artifacts)
python -m pipelines.build_all            # full run: data → baselines → LSTM → TFT → compare → anomaly → risk → snapshot → impact (~40 min CPU)
python -m pipelines.build_all --skip-tft # faster run without the TFT
python -m pipelines.build_all --only impact   # re-run one step
pytest -q                                # 32 tests
```

Run the agents on the snapshot date (31 Dec 2025):

```python
from supplychain_lens.agents.orchestrator import run_all
from supplychain_lens.tools.context import Context
r = run_all(Context())
print("\n".join(r.executive_summary))
for d in r.decisions[:10]:
    print(d.status, d.title)
```

Seed 42 throughout. Each model run writes a versioned folder under `artifacts/` with metrics, a model card and SHA-256 checksums.

## Interactive app

`app/` is a 13-page Streamlit application built for presenting the project. It uses the trained models through ONNX Runtime (no PyTorch needed).

| Section | Pages |
|---|---|
| Project | Project overview (problem, objectives, data, approach, outcomes) · Data explorer |
| Control tower | Executive control tower · AI decision room (approve / reject) · Ask the copilot |
| Deep learning models | Deep learning lab · Demand intelligence (live TFT forecast, model-vs-model replay) · Anomaly detection · Supplier risk (3 models) |
| Simulation & impact | Digital twin & scenarios (scenario builder, presets, OR-Tools transfers) · Supply network map · Business impact · Model evaluation |

Run it locally:

```bash
pip install -r requirements.txt
streamlit run app/Home.py
```

### Publish free on Streamlit Community Cloud

1. Create a GitHub repository (for example `supplychain-lens`) and upload this whole folder, **including `data/` and `artifacts/`** (the app reads the trained models from there).
2. Go to https://share.streamlit.io, sign in with GitHub, and choose **Create app → Deploy a public app from GitHub**.
3. Select the repository and branch, and set **Main file path** to `app/Home.py`.
4. Under **Advanced settings**, choose Python 3.12, then **Deploy**. The first start takes a few minutes while packages install; later visits load in seconds.

The world map on *Supply network* downloads its outlines from Plotly's CDN, so it needs an internet connection.

## Repository layout

```text
app/                  Streamlit dashboard shell (full dashboard: future work)
supplychain_lens/     Python package
  config.py           seed, fixed calendar, chronological split, paths
  data/ db/ features/ models/ training/ evaluation/ serving/
  monitoring/ risk/ simulation/ optimization/ tools/ agents/ llm/
scripts/smoke_test.py hardware and serving-path check
tests/                pytest suite (runs with or without PyTorch)
audit/                reproducible audit of NEXUS-SCCT + CPU feasibility benchmark
docs/                 architecture, ADRs, roadmap, feature audit, guides
pipelines/build_all.py end-to-end training and evaluation pipeline
reports/              data quality and forecast comparison reports
artifacts/            trained-model registry (ONNX, metrics, model cards, checksums)
requirements.txt      inference / dashboard (no PyTorch)
requirements-train.txt training (adds PyTorch, ONNX, statsmodels…)
requirements-dev.txt  + pytest, ruff
```

## Documentation

| Document | What it covers |
|---|---|
| [docs/roadmap.md](docs/roadmap.md) | All 16 phases (0–15) and their status |
| [docs/architecture.md](docs/architecture.md) | Architecture A (NEXUS as designed) and B (training vs inference) |
| [docs/change_impact_deep_learning.md](docs/change_impact_deep_learning.md) | Deep-learning upgrade: data readiness, model portfolio, compute benchmark |
| [docs/feature_audit.md](docs/feature_audit.md) | What NEXUS-SCCT advertises vs what exists |
| [audit/AUDIT_REPORT.md](audit/AUDIT_REPORT.md) | 20 reproducible checks on the reference repository |
| [docs/adr/](docs/adr/) | 10 architecture decision records |
| [docs/component_guide.md](docs/component_guide.md) | Every component explained, with interview questions |
| [docs/reuse_matrix.md](docs/reuse_matrix.md) | File-by-file plan for the reference code |
| [docs/original_vs_mine.md](docs/original_vs_mine.md) | What's different, tracked per phase |

## License and credits

Code in this repository: [Apache-2.0](LICENSE), see [NOTICE](NOTICE). Design ideas and the data model come from NEXUS-SCCT; no NEXUS-SCCT code is included ([ATTRIBUTION.md](ATTRIBUTION.md)).

Personal portfolio project. Not affiliated with SupplyLens™ (Lytica), Amazon Web Services, or SupplyChainLens.com.
