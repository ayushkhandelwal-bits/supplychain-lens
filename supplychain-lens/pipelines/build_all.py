"""Rebuild everything, in dependency order, from a clean checkout.

    python pipelines/build_all.py                 # full run (~45–60 min on a laptop CPU, mostly TFT)
    python pipelines/build_all.py --skip-tft      # reuse the stored TFT artifact (~10 min)
    python pipelines/build_all.py --only data     # just regenerate data + reports + database

Steps
  1. data       generate nexus_v2 (+ hidden truth), data-quality report, data dictionary, SQLite
  2. baselines  seasonal naive, Holt-Winters, LightGBM (quantile)
  3. lstm       PyTorch LSTM  -> ONNX
  4. tft        PyTorch TFT   -> ONNX            (skippable)
  5. compare    forecast comparison report, selects the served model on VALIDATION error
  6. anomaly    autoencoder vs robust z-score
  7. risk       neural classifier vs logistic regression vs XGBoost
  8. snapshot   today's forecasts, stockout probabilities, preset scenarios
  9. impact     Q4 2025 policy replay (business impact, SIMULATED)
 10. prune      keep only the latest version of each artifact
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STEPS = ["data", "baselines", "lstm", "tft", "compare", "anomaly", "risk", "snapshot", "impact", "prune"]


def step_data():
    from supplychain_lens.config import get_settings
    from supplychain_lens.data import dictionary, generator, quality
    from supplychain_lens.db.loader import build_sqlite

    s = get_settings()
    cfg = generator.GeneratorConfig.from_settings()
    generator.write(generator.generate_v2(cfg), s.data_dir / "nexus_v2", s.data_dir / "truth", cfg, "nexus_v2")
    generator.write(generator.generate_v1(cfg), s.data_dir / "nexus_v1", None, cfg, "nexus_v1")
    tables = quality.load(s.data_dir / "nexus_v2")
    checks = quality.run_checks(tables)
    s.reports_dir.mkdir(exist_ok=True)
    (s.reports_dir / "data_quality.md").write_text(quality.render(checks, tables), encoding="utf-8")
    (ROOT / "docs" / "data_dictionary.md").write_text(dictionary.render_markdown(s.data_dir / "nexus_v2", s.data_dir / "truth"), encoding="utf-8")
    build_sqlite(s.data_dir / "nexus_v2", s.data_dir / "supplychain_lens.db")
    failed = [c for c in checks if not c.passed]
    if failed:
        raise SystemExit(f"data-quality checks failed: {[c.name for c in failed]}")


def run(steps: list[str]) -> None:
    from supplychain_lens.evaluation import forecast_report
    from supplychain_lens.serving import registry
    from supplychain_lens.simulation import experiment, snapshot
    from supplychain_lens.training import anomaly, baselines, forecast, risk

    actions = {
        "data": step_data, "baselines": baselines.run, "lstm": lambda: forecast.run("lstm"),
        "tft": lambda: forecast.run("tft"), "compare": forecast_report.main, "anomaly": anomaly.run, "risk": risk.run,
        "snapshot": snapshot.run, "impact": experiment.run,
        "prune": lambda: [registry.prune(m) for m in ("forecast_baselines", "forecast_lstm", "forecast_tft", "forecast_comparison",
                                                       "anomaly_autoencoder", "risk_classifier", "snapshot", "business_impact")
                          if registry.latest_version(m)],
    }
    for name in steps:
        t0 = time.perf_counter()
        print(f"\n=== {name} ===", flush=True)
        actions[name]()
        print(f"=== {name} done in {time.perf_counter() - t0:.0f} s ===", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-tft", action="store_true")
    ap.add_argument("--only", choices=STEPS, nargs="+")
    a = ap.parse_args()
    chosen = a.only or [s for s in STEPS if not (a.skip_tft and s == "tft")]
    run(chosen)
