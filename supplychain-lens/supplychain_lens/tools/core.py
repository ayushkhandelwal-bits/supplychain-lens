"""The seven deterministic tools agents call. Every number an agent reports comes from here.

Each tool returns a plain dict: ``summary`` (key numbers), ``rows`` (records for tables),
``source`` (which artifact and version produced it). No tool calls an LLM.
"""

from __future__ import annotations

import pandas as pd

from supplychain_lens.tools.context import Context


def _source(ctx: Context, *models: str) -> dict:
    return {m: (ctx.run(m).version if ctx.run(m) else None) for m in models}


# 1 ─────────────────────────────────────────────────────────────────────────
def forecast_demand(ctx: Context, sku_id: str | None = None, warehouse_id: str | None = None, top_n: int = 10) -> dict:
    """28-day forecast vs the last 28 days of actual demand, per SKU × warehouse."""
    f = ctx.forecast
    agg = f.groupby(["sku_id", "warehouse_id"]).agg(p10=("p10", "sum"), p50=("p50", "sum"), p90=("p90", "sum")).reset_index()
    d = ctx.table("demand_daily")
    last = d[d.date > ctx.snapshot_date - pd.Timedelta(days=28)].groupby(["sku_id", "warehouse_id"]).demand_units.sum()
    agg = agg.merge(last.rename("last_28d").reset_index(), on=["sku_id", "warehouse_id"])
    agg["change_pct"] = 100 * (agg.p50 / agg.last_28d.clip(lower=1) - 1)
    agg["uncertainty_ratio"] = (agg.p90 - agg.p10) / agg.p50.clip(lower=1)
    if sku_id:
        agg = agg[agg.sku_id == sku_id]
    if warehouse_id:
        agg = agg[agg.warehouse_id == warehouse_id]
    rows = agg.sort_values("change_pct", ascending=False)
    return {"summary": {"series": int(len(agg)), "total_p50_28d": float(agg.p50.sum()), "total_last_28d": float(agg.last_28d.sum()),
                        "network_change_pct": float(100 * (agg.p50.sum() / max(agg.last_28d.sum(), 1) - 1)),
                        "model": ctx.snapshot_meta["forecast_model"], "horizon_days": int(ctx.forecast.h.max())},
            "rows": rows.head(top_n).to_dict("records"), "source": _source(ctx, "snapshot", ctx.snapshot_meta["forecast_artifact"])}


# 2 ─────────────────────────────────────────────────────────────────────────
EXPOSURE_DAYS = 14


def score_disruption_risk(ctx: Context, supplier_id: str | None = None, top_n: int = 10) -> dict:
    """Risk ranking: probability (classifier) × exposure (open orders + 14 days of dependent revenue)."""
    sc = ctx.risk_scores.copy()
    pos = ctx.positions
    rev_per_day = (pos.demand_p50_28d / 28 * pos.list_price).groupby(pos.primary_supplier_id).sum()
    dep = pos.groupby("primary_supplier_id").agg(dependent_series=("sku_id", "size"))
    sh = ctx.table("shipments")
    open_value = sh[sh.status.isin(["IN_TRANSIT", "AT_PORT", "DELAYED"])].groupby("supplier_id").value_usd.sum()
    sc["open_order_value_usd"] = sc.supplier_id.map(open_value).fillna(0)
    sc["revenue_14d_usd"] = sc.supplier_id.map(rev_per_day).fillna(0) * EXPOSURE_DAYS
    sc["exposure_usd"] = sc.open_order_value_usd + sc.revenue_14d_usd
    sc["expected_loss_usd"] = sc.probability * sc.exposure_usd
    sc["dependent_series"] = sc.supplier_id.map(dep.dependent_series).fillna(0).astype(int)
    sc["flagged"] = sc.probability >= sc.threshold
    active = ctx.table("disruptions")
    active = active[active.status == "ACTIVE"].merge(ctx.table("disruption_suppliers"), on="event_id")
    sc["active_recorded_event"] = sc.supplier_id.isin(active.supplier_id)
    med = sc[["late_share_28", "delay_mean_28", "overdue_open", "fill_ratio_28"]].median()
    drivers = []
    for r in sc.itertuples():
        d = []
        if r.late_share_28 > max(med.late_share_28 * 1.5, 0.2):
            d.append(f"{r.late_share_28:.0%} of deliveries late in the last 28 days (fleet median {med.late_share_28:.0%})")
        if r.delay_mean_28 > med.delay_mean_28 + 3:
            d.append(f"average delay {r.delay_mean_28:+.1f} days (fleet median {med.delay_mean_28:+.1f})")
        if r.overdue_open > 0:
            d.append(f"{int(r.overdue_open)} open orders past their promised date")
        if r.fill_ratio_28 < 0.95:
            d.append(f"shipping only {r.fill_ratio_28:.0%} of ordered quantity")
        if r.health_score < 50:
            d.append(f"low supplier health score ({int(r.health_score)})")
        drivers.append(d)
    sc["drivers"] = drivers
    if supplier_id:
        sc = sc[sc.supplier_id == supplier_id]
    cols = ["supplier_id", "name", "country", "category", "probability", "threshold", "flagged", "active_recorded_event",
            "exposure_usd", "expected_loss_usd", "dependent_series", "drivers", "late_share_28", "delay_mean_28",
            "overdue_open", "fill_ratio_28", "health_score"]
    rows = sc.sort_values("expected_loss_usd", ascending=False)[cols]
    m = ctx.risk_metrics["models"][ctx.risk_metrics["selected_model"]]["test"]
    return {"summary": {"suppliers_flagged": int(sc.flagged.sum()), "threshold": float(sc.threshold.iloc[0]) if len(sc) else None,
                        "total_expected_loss_usd": float(sc.expected_loss_usd.sum()), "model": ctx.risk_metrics["selected_model"],
                        "horizon_days": 14, "test_pr_auc": m["pr_auc"], "test_roc_auc": m["roc_auc"], "test_precision": m["precision"],
                        "test_recall": m["recall"], "positive_rate": m["positive_rate"]},
            "rows": rows.head(top_n).to_dict("records"), "source": _source(ctx, "risk_classifier", "snapshot")}


# 3 ─────────────────────────────────────────────────────────────────────────
def detect_anomalies(ctx: Context, days: int = 14, top_n: int = 15) -> dict:
    """Autoencoder alerts in the last ``days`` days before the snapshot ("investigation required")."""
    a = ctx.anomaly_alerts
    a = a[a.date > ctx.snapshot_date - pd.Timedelta(days=days)].copy()
    reasons = {"demand_z": "demand far from its recent pattern", "unexplained_inv": "inventory changed without sales or receipts",
               "receipt_delay": "receipts arriving much later than promised"}
    a["reason"] = a.top_feature_1.map(reasons).fillna(a.top_feature_1)
    a["status"] = "Investigation required"
    keep = ["date", "sku_id", "warehouse_id", "category", "supplier_id", "score", "threshold", "reason", "top_feature_1",
            "raw_demand_z", "raw_unexplained_inv", "raw_receipt_delay", "status"]
    m = ctx.run("anomaly_autoencoder").json("metrics.json")["models"]["Autoencoder (PyTorch)"]["test"]
    return {"summary": {"alerts": int(len(a)), "window_days": days, "threshold": float(m["threshold"]),
                        "test_precision_incl_recorded": m["precision_incl_recorded_events"], "test_event_recall": m["event_recall"]},
            "rows": a.sort_values("score", ascending=False)[keep].head(top_n).to_dict("records"),
            "source": _source(ctx, "anomaly_autoencoder")}


# 4 ─────────────────────────────────────────────────────────────────────────
def calculate_stockout_probability(ctx: Context, sku_id: str | None = None, warehouse_id: str | None = None,
                                   min_probability: float = 0.0, top_n: int = 15) -> dict:
    """Monte Carlo P(stockout in 14 days) from forecast quantiles, stock on hand and expected receipts."""
    p = ctx.positions
    if sku_id:
        p = p[p.sku_id == sku_id]
    if warehouse_id:
        p = p[p.warehouse_id == warehouse_id]
    p = p[p.stockout_probability_14d >= min_probability]
    rows = p.sort_values("expected_lost_margin_14d", ascending=False)
    allp = ctx.positions
    return {"summary": {"series_at_risk_50pct": int((allp.stockout_probability_14d >= 0.5).sum()),
                        "series_total": int(len(allp)), "expected_lost_revenue_14d": float(allp.expected_lost_revenue_14d.sum()),
                        "expected_lost_margin_14d": float(allp.expected_lost_margin_14d.sum()),
                        "window_days": int(ctx.snapshot_meta["stockout_window_days"]), "mc_samples": int(ctx.snapshot_meta["mc_samples"])},
            "rows": rows.head(top_n).to_dict("records"), "source": _source(ctx, "snapshot")}


# 5 ─────────────────────────────────────────────────────────────────────────
def simulate_supplier_disruption(ctx: Context, supplier_id: str, offline: bool = True, lead_time_add_days: int = 0,
                                 capacity_pct: float = 100.0, reps: int = 30) -> dict:
    """Forward digital twin (12 weeks): what happens to KPIs if this supplier is disrupted?"""
    from supplychain_lens.simulation.twin import Scenario

    sc = Scenario(name=f"{supplier_id} disruption", offline_supplier=supplier_id if offline else None,
                  affected_supplier=None if offline else supplier_id, lead_time_add_days=lead_time_add_days,
                  supplier_capacity_pct=capacity_pct)
    return compare_inventory_policies(ctx, sc, reps=reps)


# 6 ─────────────────────────────────────────────────────────────────────────
def compare_inventory_policies(ctx: Context, scenario=None, reps: int = 30) -> dict:
    """Baseline rule vs AI-assisted policy under a scenario, with the same random draws (paired)."""
    from supplychain_lens.simulation.snapshot import run_scenario
    from supplychain_lens.simulation.twin import Scenario

    scenario = scenario or Scenario()
    st = ctx.twin_state
    sc = ctx.risk_scores.set_index("supplier_id").loc[st.supplier_ids]
    res = run_scenario(st, ctx.twin_quantiles, scenario, sc.probability.to_numpy(), float(sc.threshold.iloc[0]), reps,
                       ctx.settings.seed)
    base_ref = run_scenario(st, ctx.twin_quantiles, Scenario(), sc.probability.to_numpy(), float(sc.threshold.iloc[0]), reps,
                            ctx.settings.seed)
    b, a, ref = res["baseline"], res["ai"], base_ref["baseline"]
    summary = {
        "scenario": scenario.name, "reps": reps, "days": int(ctx.twin_state.days),
        "fill_rate_baseline": b["fill_rate"]["mean"], "fill_rate_ai": a["fill_rate"]["mean"],
        "fill_rate_normal_conditions": ref["fill_rate"]["mean"],
        "lost_revenue_baseline": b["lost_revenue"]["mean"], "lost_revenue_ai": a["lost_revenue"]["mean"],
        "lost_revenue_normal_conditions": ref["lost_revenue"]["mean"],
        "total_cost_baseline": b["total_cost"]["mean"], "total_cost_ai": a["total_cost"]["mean"],
        "backup_orders_ai": a["backup_orders"]["mean"], "label": "SIMULATED",
    }
    return {"summary": summary, "rows": [], "detail": res, "source": _source(ctx, "snapshot", "risk_classifier")}


# 7 ─────────────────────────────────────────────────────────────────────────
def calculate_business_impact(ctx: Context) -> dict:
    """Q4 2025 replay: baseline vs AI-assisted policy (precomputed experiment, 95% intervals)."""
    imp = ctx.impact
    if imp is None:
        return {"summary": {"available": False}, "rows": [], "source": {}}
    rows = [{"metric": k, **v} for k, v in imp["metrics"].items()]
    m = imp["metrics"]
    return {"summary": {"available": True, "label": imp["label"], "period": imp["period"], "replications": imp["replications"],
                        "fill_rate_baseline": m["fill_rate"]["baseline_mean"], "fill_rate_ai": m["fill_rate"]["ai_mean"],
                        "total_cost_change": m["total_cost"]["diff"], "total_cost_change_pct": m["total_cost"]["relative_change"],
                        "total_cost_ci": [m["total_cost"]["ci_low"], m["total_cost"]["ci_high"]],
                        "lost_revenue_change": m["lost_revenue"]["diff"], "inventory_change_pct": m["avg_inventory_value"]["relative_change"]},
            "rows": rows, "source": _source(ctx, "business_impact")}


# Helpers used by the Inventory agent ─────────────────────────────────────────
def transfer_options(ctx: Context, sku_id: str) -> dict:
    """OR-Tools re-balancing plan for one SKU across its warehouses (14-day P90 cover)."""
    from supplychain_lens.optimization.transfers import plan_transfers

    p = ctx.positions[ctx.positions.sku_id == sku_id]
    pos = pd.DataFrame({"warehouse_id": p.warehouse_id, "on_hand": p.on_hand, "arriving": p.receipts_14d,
                        "demand_p90": p.demand_p90_14d})
    unit_cost, price = float(p.unit_cost.iloc[0]), float(p.list_price.iloc[0])
    plan = plan_transfers(sku_id, pos, ctx.table("warehouses"), unit_cost, price - unit_cost)
    return {"summary": {"sku_id": sku_id, "status": plan.solver_status, "transfer_cost_usd": plan.total_transfer_cost,
                        "units_moved": float(plan.moves.units.sum()) if len(plan.moves) else 0.0,
                        "unmet_before": plan.unmet_units_before, "unmet_after": plan.unmet_units_after,
                        "margin_protected_usd": plan.lost_margin_avoided, "net_benefit_usd": plan.net_benefit},
            "rows": plan.moves.to_dict("records"), "source": _source(ctx, "snapshot")}


TOOLS = {
    "forecast_demand": forecast_demand,
    "score_disruption_risk": score_disruption_risk,
    "detect_anomalies": detect_anomalies,
    "calculate_stockout_probability": calculate_stockout_probability,
    "simulate_supplier_disruption": simulate_supplier_disruption,
    "compare_inventory_policies": compare_inventory_policies,
    "calculate_business_impact": calculate_business_impact,
}
