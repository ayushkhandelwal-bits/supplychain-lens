"""Precompute the "today" view (snapshot date) so the dashboard loads instantly.

    python -m supplychain_lens.simulation.snapshot

Writes artifact ``snapshot/<version>/``:
  forecast.parquet     28-day P10/P50/P90 from the selected model for every SKU × warehouse
  positions.parquet    on hand, expected receipts, days of supply, stockout probability (14 days),
                       expected lost units / margin, per SKU × warehouse
  scenarios.json       preset what-if scenarios, baseline vs AI-assisted policy (forward twin)
  meta.json            snapshot date, model versions, timing
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from supplychain_lens.config import get_settings
from supplychain_lens.features.forecasting import load_panel
from supplychain_lens.serving import registry
from supplychain_lens.serving.forecaster import Forecaster
from supplychain_lens.serving.risk import RiskScorer
from supplychain_lens.simulation.twin import Scenario, build_state, quantile_demand_paths, simulate

STOCKOUT_WINDOW = 14
MC_SAMPLES = 2000
SCENARIO_REPS = 40
SCENARIO_DAYS = 84  # 12 weeks: orders placed now (lead times 20–90 days) can land inside the window


def extend_quantiles(q: np.ndarray, days: int) -> np.ndarray:
    """Beyond the 28-day forecast, repeat its 4-week pattern (documented planning assumption)."""
    reps = int(np.ceil(days / q.shape[1]))
    return np.concatenate([q] * reps, axis=1)[:, :days]


def expected_receipts(state, days: int) -> np.ndarray:
    """(S, days) expected receipts from open orders: promised date + supplier's median recent delay."""
    out = np.zeros((len(state.keys), days))
    med = np.array([np.median(p) for p in state.delay_pool])
    for k in range(len(state.pipe_qty)):
        d = int(round(state.pipe_promised[k] + med[state.pipe_supplier[k]]))
        if 0 <= d < days:
            out[state.pipe_series[k], d] += state.pipe_qty[k]
        elif d < 0:  # overdue: assume it arrives tomorrow at the earliest
            out[state.pipe_series[k], 0] += state.pipe_qty[k]
    return out


def stockout_table(state, q: np.ndarray, seed: int = 42) -> pd.DataFrame:
    """Monte Carlo P(stockout within STOCKOUT_WINDOW days) per series from forecast quantiles."""
    W = STOCKOUT_WINDOW
    paths = quantile_demand_paths(q[:, :W], MC_SAMPLES, seed)            # (M, S, W)
    rec = expected_receipts(state, W)
    supply = state.on_hand0[None, :, None] + np.cumsum(rec, axis=1)[None]
    cum_d = np.cumsum(paths, axis=2)
    short = cum_d > supply
    shortfall = np.maximum(0, cum_d[:, :, -1] - supply[:, :, -1])
    k = state.keys
    margin = (k.list_price - k.unit_cost).to_numpy()
    p50 = q[..., 1]
    first_short = np.where(short.any(2), short.argmax(2), W)
    return pd.DataFrame({
        "sku_id": k.sku_id, "warehouse_id": k.warehouse_id, "category": k.category,
        "primary_supplier_id": k.primary_supplier_id, "backup_supplier_id": k.backup_supplier_id,
        "unit_cost": k.unit_cost, "list_price": k.list_price, "on_hand": state.on_hand0,
        "receipts_14d": rec.sum(1), "demand_p50_14d": p50[:, :W].sum(1), "demand_p50_28d": p50.sum(1),
        "demand_p90_14d": q[:, :W, 2].sum(1), "days_of_supply": state.on_hand0 / np.maximum(p50.mean(1), 1e-6),
        "stockout_probability_14d": short.any(2).mean(0),
        "median_days_to_stockout": np.median(first_short, axis=0),
        "expected_lost_units_14d": shortfall.mean(0), "expected_lost_margin_14d": shortfall.mean(0) * margin,
        "expected_lost_revenue_14d": shortfall.mean(0) * k.list_price.to_numpy(),
        "inventory_value": state.on_hand0 * k.unit_cost.to_numpy(),
    })


def preset_scenarios(top_supplier: str) -> list[Scenario]:
    return [
        Scenario("Baseline conditions"),
        Scenario("Demand +20%", demand_change_pct=20),
        Scenario("All supplier lead times +5 days", lead_time_add_days=5),
        Scenario(f"Highest-risk supplier offline ({top_supplier})", offline_supplier=top_supplier),
        Scenario("Supplier capacity −30%", supplier_capacity_pct=70),
        Scenario("Warehouse capacity −10%", warehouse_capacity_pct=90),
        Scenario("Transport delays +7 days", transport_delay_days=7),
    ]


def run_scenario(state, q, scenario: Scenario, risk_now: np.ndarray, threshold: float, reps: int, seed: int) -> dict:
    days, H = state.days, q.shape[1]
    q_long = extend_quantiles(q, days + H)
    demand = quantile_demand_paths(q_long[:, :days], reps, seed)
    # Each day the policy sees the next 28 days of the (extended) snapshot forecast.
    fq = np.stack([q_long[:, t : t + H] for t in range(days)])
    risk = np.repeat(risk_now[None], days, axis=0)
    base = simulate(state, demand, "baseline", scenario, seed=seed)
    ai = simulate(state, demand, "ai", scenario, forecasts=fq, risk=risk, risk_threshold=threshold, seed=seed)

    def summ(r):
        return {k: {"mean": float(np.mean(v)), "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95))}
                for k, v in r["per_rep"].items()}
    return {"scenario": scenario.to_dict(), "baseline": summ(base), "ai": summ(ai),
            "daily": {"baseline_fill": base["daily"]["fill_rate"].tolist(), "ai_fill": ai["daily"]["fill_rate"].tolist(),
                      "baseline_lost_revenue": base["daily"]["lost_revenue"].tolist(),
                      "ai_lost_revenue": ai["daily"]["lost_revenue"].tolist()}}


def run(log=print) -> registry.Run:
    s = get_settings()
    t0 = time.perf_counter()
    panel = load_panel()
    origin = len(panel.dates)
    fc = Forecaster.load(panel=panel)
    q_all = fc.predict([origin])[0]                                   # (S_panel, H, 3)
    forecast = fc.forecast_frame(origin)
    start = panel.dates[-1] + pd.Timedelta(days=1)
    state = build_state(start, SCENARIO_DAYS, known_arrivals=False)
    order = state.keys[["sku_id", "warehouse_id"]].merge(panel.keys[["sku_id", "warehouse_id"]].reset_index(names="pi"),
                                                         on=["sku_id", "warehouse_id"]).pi.to_numpy()
    q = q_all[order]
    positions = stockout_table(state, q, s.seed)  # uses the first 14 days only

    scorer = RiskScorer.load()
    scores = scorer.current_scores().set_index("supplier_id").loc[state.supplier_ids]
    risk_now = scores.probability.to_numpy()
    top_supplier = scores.probability.idxmax()
    scenarios = []
    for sc in preset_scenarios(top_supplier):
        scenarios.append(run_scenario(state, q, sc, risk_now, scorer.threshold, SCENARIO_REPS, s.seed))
        b, a = scenarios[-1]["baseline"], scenarios[-1]["ai"]
        log(f"  {sc.name:<45} fill baseline {b['fill_rate']['mean']:.3f}  AI {a['fill_rate']['mean']:.3f}")

    run_ = registry.start_run("snapshot")
    forecast.to_parquet(run_.path / "forecast.parquet", index=False)
    positions.to_parquet(run_.path / "positions.parquet", index=False)
    registry.write_json(run_, "scenarios.json", {"reps": SCENARIO_REPS, "start": str(start.date()),
                                                 "days": SCENARIO_DAYS, "results": scenarios,
                                                 "assumption": "days 29–84 repeat the 28-day forecast pattern",
                                                 "label": "SIMULATED — forward digital-twin scenarios from the snapshot"})
    registry.write_json(run_, "meta.json", {"snapshot_date": str(panel.dates[-1].date()), "forecast_model": fc.model_name,
                                            "forecast_artifact": fc.artifact, "forecast_version": fc.version,
                                            "forecast_latency_ms_all_series": round(fc.last_latency_ms, 1),
                                            "risk_model": scorer.selected, "risk_version": scorer.version,
                                            "risk_threshold": scorer.threshold, "top_risk_supplier": top_supplier,
                                            "stockout_window_days": STOCKOUT_WINDOW, "mc_samples": MC_SAMPLES,
                                            "seconds": round(time.perf_counter() - t0, 1)})
    registry.finalize(run_)
    log(f"Saved {run_.path}")
    return run_


if __name__ == "__main__":
    run()
