"""Run the three forecasting baselines and store their predictions as an artifact.

    python -m supplychain_lens.training.baselines
"""

from __future__ import annotations

import time

import pandas as pd

from supplychain_lens.evaluation.metrics import forecast_summary
from supplychain_lens.features.forecasting import evaluation_origins, load_panel, training_origins
from supplychain_lens.models import baselines as b
from supplychain_lens.serving import registry


def run(log=print) -> registry.Run:
    panel = load_panel()
    vo, to = evaluation_origins(panel, "val"), evaluation_origins(panel, "test")
    frames, timing = [], {}
    for name, fn in (("Seasonal naive", b.seasonal_naive), ("Holt-Winters (ETS)", b.holt_winters)):
        t0 = time.perf_counter()
        frames += [fn(panel, vo).assign(split="val"), fn(panel, to).assign(split="test")]
        timing[name] = round(time.perf_counter() - t0, 1)
        log(f"  {name}: {timing[name]} s")
    t0 = time.perf_counter()
    lgb = b.train_lightgbm(panel, training_origins(panel, stride=7), vo)
    frames += [b.predict_lightgbm(panel, lgb["models"], vo).assign(split="val"),
               b.predict_lightgbm(panel, lgb["models"], to).assign(split="test")]
    timing["LightGBM (quantile)"] = round(time.perf_counter() - t0, 1)
    log(f"  LightGBM: {timing['LightGBM (quantile)']} s  {lgb['info']['rounds']}")

    preds = pd.concat(frames, ignore_index=True)
    metrics = {m: {split: forecast_summary(g) for split, g in gm.groupby("split")} for m, gm in preds.groupby("model")}
    run_ = registry.start_run("forecast_baselines")
    preds.to_parquet(run_.path / "predictions.parquet", index=False)
    for q, m in lgb["models"].items():
        m.save_model(str(run_.path / f"lightgbm_q{int(q * 100):02d}.txt"), num_iteration=m.best_iteration)
    registry.write_json(run_, "metrics.json", {"models": metrics, "fit_seconds": timing, "lightgbm": lgb["info"],
                                               "lightgbm_features": b.LGB_FEATURES})
    registry.finalize(run_)
    for m, d in metrics.items():
        log(f"  {m:<22} val WAPE {d['val']['wape']:.4f}  test WAPE {d['test']['wape']:.4f}")
    return run_


if __name__ == "__main__":
    run()
