# Agent roles inspired by NEXUS-SCCT's LangGraph agents (orchestrator.py); reimplemented as
# tool-grounded Python classes (ADR-0005). No agent invents a number: all come from tools.
"""The five specialist agents."""

from __future__ import annotations

import numpy as np
import pandas as pd

from supplychain_lens.agents.policy import risk_level_from_value
from supplychain_lens.agents.schemas import AgentDecision, AgentReport, Alternative
from supplychain_lens.optimization.routing import mode_options
from supplychain_lens.tools import core
from supplychain_lens.tools.context import Context


def money(x: float) -> str:
    x = float(x)
    for unit, div in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(x) >= div:
            return f"${x / div:,.1f}{unit}"
    return f"${x:,.0f}"


class DemandIntelligenceAgent:
    name = "Demand Intelligence"

    def run(self, ctx: Context, top_n: int = 3) -> AgentReport:
        fd = core.forecast_demand(ctx, top_n=400)
        s = fd["summary"]
        rows = pd.DataFrame(fd["rows"])
        pos = ctx.positions.set_index(["sku_id", "warehouse_id"])
        findings = [f"Network demand forecast for the next {s['horizon_days']} days is {s['total_p50_28d']:,.0f} units "
                    f"({s['network_change_pct']:+.1f}% vs the last 28 days), from {s['model']}."]
        wide = rows[rows.uncertainty_ratio > rows.uncertainty_ratio.quantile(0.9)]
        findings.append(f"{len(wide)} series have the widest forecast bands (P10–P90 width ≥ "
                        f"{rows.uncertainty_ratio.quantile(0.9):.0%} of P50): buffers matter most there.")
        surge = rows[(rows.change_pct >= 25)].copy()
        surge["exposure"] = [pos.loc[(r.sku_id, r.warehouse_id), "expected_lost_margin_14d"] for r in surge.itertuples()]
        decisions = []
        for r in surge.sort_values("exposure", ascending=False).head(top_n).itertuples():
            p = pos.loc[(r.sku_id, r.warehouse_id)]
            conf = float(np.clip(1 - r.uncertainty_ratio / 2, 0, 0.99))
            decisions.append(AgentDecision(
                agent=self.name, decision_type="ORDER_ADJUSTMENT",
                title=f"Demand rising for {r.sku_id} at {r.warehouse_id}",
                action=f"Raise the order-up-to level for {r.sku_id} @ {r.warehouse_id} to cover the forecast "
                       f"(P50 {r.p50:,.0f} units over 28 days) at the next review.",
                evidence=[f"Forecast P50 {r.p50:,.0f} units vs {r.last_28d:,.0f} in the last 28 days ({r.change_pct:+.0f}%)",
                          f"80% band {r.p10:,.0f}–{r.p90:,.0f} units",
                          f"On hand {p.on_hand:,.0f} units = {p.days_of_supply:.1f} days of supply",
                          f"P(stockout in 14 days) {p.stockout_probability_14d:.0%}"],
                confidence=conf, confidence_basis="1 − half the relative forecast-band width (narrow band = high confidence)",
                risk_level=risk_level_from_value(p.expected_lost_margin_14d, p.stockout_probability_14d),
                estimated_value_usd=float(p.expected_lost_margin_14d),
                expected_impact={"expected_lost_margin_14d_if_no_action": float(p.expected_lost_margin_14d)},
                affected={"sku_id": r.sku_id, "warehouse_id": r.warehouse_id},
                uncertainty=f"Forecast quantiles from {s['model']}; empirical P10–P90 coverage is reported on the Model Evaluation page.",
                sources=fd["source"]))
        return AgentReport(self.name, findings, decisions, ["forecast_demand"], {"summary": s})


class AnomalyInvestigationAgent:
    name = "Anomaly Investigation"

    def run(self, ctx: Context, top_n: int = 4) -> AgentReport:
        an = core.detect_anomalies(ctx, days=14, top_n=50)
        rows = pd.DataFrame(an["rows"])
        dis = ctx.table("disruptions")
        active = dis[dis.status == "ACTIVE"].merge(ctx.table("disruption_suppliers"), on="event_id")
        findings = [f"{an['summary']['alerts']} autoencoder alerts in the last 14 days (threshold "
                    f"{an['summary']['threshold']:.2f}). Alerts are model signals, not confirmed events."]
        decisions = []
        for r in rows.head(top_n).itertuples() if len(rows) else []:
            explained = active[active.supplier_id == r.supplier_id]
            ev = [f"Anomaly score {r.score:.2f} vs threshold {r.threshold:.2f} on {pd.Timestamp(r.date):%d %b %Y}",
                  f"Main signal: {r.reason}"]
            if r.top_feature_1 == "demand_z":
                ev.append(f"Demand was {r.raw_demand_z:+.1f} standard deviations from its 28-day pattern")
            if r.top_feature_1 == "unexplained_inv":
                ev.append(f"Unexplained stock change of {r.raw_unexplained_inv:+.1f}× average daily demand")
            if r.top_feature_1 == "receipt_delay":
                ev.append(f"Receipts arrived {r.raw_receipt_delay * 7:+.0f} days vs promise")
            if len(explained):
                ev.append(f"Supplier {r.supplier_id} has an active recorded event: {explained.title.iloc[0]} ({explained.event_id.iloc[0]})")
            decisions.append(AgentDecision(
                agent=self.name, decision_type="INVESTIGATION",
                title=f"Unusual pattern: {r.sku_id} at {r.warehouse_id}",
                action=("Check the recorded supplier event first; likely explained." if len(explained) else
                        "Investigate: verify the transaction records, a physical count and the supplier's confirmations."),
                evidence=ev, confidence=float(min(0.99, r.score / (2 * r.threshold))),
                confidence_basis="anomaly score relative to twice the threshold",
                risk_level="MEDIUM", estimated_value_usd=0.0,
                affected={"sku_id": r.sku_id, "warehouse_id": r.warehouse_id, "supplier_id": r.supplier_id},
                uncertainty=f"Detector test precision (incl. recorded events) {an['summary']['test_precision_incl_recorded']:.0%}, "
                            f"event recall {an['summary']['test_event_recall']:.0%} on synthetic labels.",
                sources=an["source"]))
        return AgentReport(self.name, findings, decisions, ["detect_anomalies"], {"summary": an["summary"]})


class SupplierRiskAgent:
    name = "Supplier Risk"

    def run(self, ctx: Context, top_n: int = 3, simulate_top: int = 1) -> AgentReport:
        rk = core.score_disruption_risk(ctx, top_n=50)
        rows = pd.DataFrame(rk["rows"])
        s = rk["summary"]
        findings = [f"{s['suppliers_flagged']} of 50 suppliers are above the {s['model']} threshold "
                    f"({s['threshold']:.2f}) for a supplier-originated disruption in the next 14 days.",
                    f"Total expected loss across suppliers: {money(s['total_expected_loss_usd'])} "
                    f"(probability × open orders and 14 days of dependent revenue).",
                    f"Model quality on the test quarter: PR-AUC {s['test_pr_auc']:.2f} vs base rate {s['positive_rate']:.2f}; "
                    f"precision {s['test_precision']:.0%}, recall {s['test_recall']:.0%}."]
        cand = rows[(rows.flagged) | (rows.active_recorded_event)].head(top_n)
        decisions = []
        for i, r in enumerate(cand.itertuples()):
            prod = ctx.table("products")
            skus = prod[prod.primary_supplier_id == r.supplier_id]
            backups = sorted(skus.backup_supplier_id.unique())
            sim = None
            if i < simulate_top:
                sim = core.simulate_supplier_disruption(ctx, r.supplier_id, offline=True, reps=20)["summary"]
            ev = [f"Model probability of a new supplier-originated failure: {r.probability:.0%} (threshold {r.threshold:.0%})",
                  f"Exposure {money(r.exposure_usd)}; expected loss {money(r.expected_loss_usd)}",
                  f"{r.dependent_series} SKU × warehouse series depend on it as primary source"]
            ev += list(r.drivers)
            if r.active_recorded_event:
                ev.append("A recorded disruption is already active for this supplier")
            # A recorded, active event is a fact (probability 1); otherwise use the classifier's probability.
            p_eff = 1.0 if r.active_recorded_event else float(r.probability)
            impact, value = {}, float(p_eff * r.exposure_usd) if r.active_recorded_event else float(r.expected_loss_usd)
            if sim:
                avoided = sim["lost_revenue_baseline"] - sim["lost_revenue_ai"]
                ev.append(f"Simulated 28 days if it goes offline: fill rate {sim['fill_rate_baseline']:.1%} with the current rule "
                          f"vs {sim['fill_rate_ai']:.1%} with backup sourcing (normal conditions {sim['fill_rate_normal_conditions']:.1%})")
                impact = {"fill_rate_if_offline_current_rule": sim["fill_rate_baseline"],
                          "fill_rate_if_offline_with_backup": sim["fill_rate_ai"],
                          "lost_revenue_avoided_if_offline": avoided, "simulation_reps": sim["reps"]}
                value = float(p_eff * avoided)
            decisions.append(AgentDecision(
                agent=self.name, decision_type="SUPPLIER_MITIGATION",
                title=(f"Active disruption: {r.name} ({r.supplier_id})" if r.active_recorded_event
                       else f"Elevated disruption risk: {r.name} ({r.supplier_id})"),
                action=f"Route the next orders for {len(skus)} SKUs to backup supplier(s) {', '.join(backups)} "
                       f"and confirm open orders with {r.supplier_id}.",
                evidence=ev, confidence=p_eff,
                confidence_basis=("recorded event in progress (certain)" if r.active_recorded_event else
                                  "classifier probability of a supplier-originated disruption within 14 days"),
                risk_level=risk_level_from_value(value, p_eff), estimated_value_usd=value,
                expected_impact=impact,
                alternatives=[Alternative("Monitor only", 0.0, 0.0, "Keep the primary supplier; review again next week"),
                              Alternative("Shift to backup", None, value, "Backup orders cost +8% per unit"),
                              Alternative("Expedite open orders by air", None, None, "See Supply Network for lane options")],
                affected={"supplier_id": r.supplier_id, "skus": list(skus.sku_id)[:20]},
                uncertainty="Synthetic supervised labels; probabilities are model scores and may be miscalibrated (see reliability chart).",
                sources=rk["source"]))
        return AgentReport(self.name, findings, decisions, ["score_disruption_risk", "simulate_supplier_disruption"],
                           {"summary": s, "rows": rows.head(10).to_dict("records")})


class InventoryAgent:
    name = "Inventory"

    def run(self, ctx: Context, top_n: int = 4) -> AgentReport:
        so = core.calculate_stockout_probability(ctx, min_probability=0.5, top_n=60)
        s = so["summary"]
        rows = pd.DataFrame(so["rows"])
        findings = [f"{s['series_at_risk_50pct']} of {s['series_total']} SKU × warehouse series have ≥ 50% probability of a "
                    f"stockout within {s['window_days']} days.",
                    f"Expected lost revenue over that window: {money(s['expected_lost_revenue_14d'])} "
                    f"(margin {money(s['expected_lost_margin_14d'])}), from {s['mc_samples']:,} Monte Carlo demand paths."]
        decisions, seen = [], set()
        sh = ctx.table("shipments")
        wh = ctx.table("warehouses").set_index("warehouse_id")
        sup = ctx.table("suppliers").set_index("supplier_id")
        for r in rows.itertuples() if len(rows) else []:
            if r.sku_id in seen or len(decisions) >= top_n:
                continue
            seen.add(r.sku_id)
            tr = core.transfer_options(ctx, r.sku_id)
            base_ev = [f"P(stockout in 14 days) {r.stockout_probability_14d:.0%}; median days to stockout {r.median_days_to_stockout:.0f}",
                       f"On hand {r.on_hand:,.0f} units, expected receipts {r.receipts_14d:,.0f}, P90 demand {r.demand_p90_14d:,.0f}",
                       f"Expected lost margin without action {money(r.expected_lost_margin_14d)}"]
            alternatives = [Alternative("Do nothing", 0.0, 0.0, f"Expected lost margin {money(r.expected_lost_margin_14d)}")]
            open_po = sh[(sh.sku_id == r.sku_id) & (sh.warehouse_id == r.warehouse_id) & sh.status.isin(["IN_TRANSIT", "DELAYED"])]
            if len(open_po):
                o = open_po.sort_values("eta").iloc[0]
                w, sp = wh.loc[r.warehouse_id], sup.loc[o.supplier_id]
                opts = mode_options(sp.country, w.lat, w.lng, w.region, sp.region, r.unit_cost, o.quantity_units)
                air = opts[opts["mode"] == "AIR"]
                if len(air):
                    alternatives.append(Alternative(f"Expedite {o.shipment_id} ({o.quantity_units:,.0f} units) by air",
                                                    float(air.freight_cost_usd.iloc[0]), None,
                                                    f"≈{air.transit_days.iloc[0]:.0f} days transit for the remaining leg"))
            if tr["summary"]["status"] == "OPTIMAL" and tr["summary"]["net_benefit_usd"] > 0:
                moves = pd.DataFrame(tr["rows"])
                desc = "; ".join(f"{m.units:,} units {m.from_wh} → {m.to_wh} ({m.transit_days} days)" for m in moves.itertuples())
                cost, protected = tr["summary"]["transfer_cost_usd"], tr["summary"]["margin_protected_usd"]
                exp_value = float(protected * r.stockout_probability_14d - cost)
                decisions.append(AgentDecision(
                    agent=self.name, decision_type="INVENTORY_TRANSFER",
                    title=f"Stockout risk: {r.sku_id} at {r.warehouse_id}",
                    action=f"Re-balance {r.sku_id}: {desc}.",
                    evidence=base_ev + [f"OR-Tools min-cost-flow plan: move {tr['summary']['units_moved']:,.0f} units for "
                                        f"{money(cost)}; covers {tr['summary']['unmet_before'] - tr['summary']['unmet_after']:,.0f} "
                                        f"of {tr['summary']['unmet_before']:,.0f} units of P90 shortfall"],
                    confidence=float(r.stockout_probability_14d),
                    confidence_basis="Monte Carlo probability that the shortage occurs without action",
                    risk_level=risk_level_from_value(r.expected_lost_margin_14d, r.stockout_probability_14d),
                    estimated_value_usd=exp_value,
                    expected_impact={"transfer_cost_usd": cost, "margin_protected_if_shortage_usd": protected,
                                     "expected_net_benefit_usd": exp_value},
                    alternatives=alternatives + [Alternative("Transfer between warehouses", cost, exp_value, "OR-Tools plan")],
                    affected={"sku_id": r.sku_id, "warehouses": sorted(set(moves.from_wh) | set(moves.to_wh))},
                    uncertainty="Demand paths sampled from forecast quantiles; receipts assume each supplier's median recent delay.",
                    sources={**so["source"], **tr["source"]}))
            else:
                b = sup.loc[r.backup_supplier_id]
                decisions.append(AgentDecision(
                    agent=self.name, decision_type="EMERGENCY_ORDER",
                    title=f"Stockout risk: {r.sku_id} at {r.warehouse_id}",
                    action=f"No sibling warehouse has surplus. Place an emergency order with backup supplier {b.name} "
                           f"({r.backup_supplier_id}, nominal lead time {b.avg_lead_time_days} days) and expedite open orders.",
                    evidence=base_ev + ["Transfer optimiser found no economical surplus in the SKU's other warehouses"],
                    confidence=float(r.stockout_probability_14d),
                    confidence_basis="Monte Carlo probability that the shortage occurs without action",
                    risk_level=risk_level_from_value(r.expected_lost_margin_14d, r.stockout_probability_14d),
                    estimated_value_usd=float(r.expected_lost_margin_14d), reversible=False,
                    alternatives=alternatives,
                    affected={"sku_id": r.sku_id, "warehouse_id": r.warehouse_id},
                    uncertainty=f"Backup lead time ({b.avg_lead_time_days} days) may exceed the 14-day risk window; value is an upper bound.",
                    sources=so["source"]))
        return AgentReport(self.name, findings, decisions, ["calculate_stockout_probability", "transfer_options"], {"summary": s})


class SimulationAgent:
    name = "Simulation"

    def run(self, ctx: Context) -> AgentReport:
        sc = ctx.scenarios
        findings, rows = [], []
        for res in sc["results"]:
            b, a = res["baseline"], res["ai"]
            rows.append({"scenario": res["scenario"]["name"], "fill_baseline": b["fill_rate"]["mean"], "fill_ai": a["fill_rate"]["mean"],
                         "lost_revenue_baseline": b["lost_revenue"]["mean"], "lost_revenue_ai": a["lost_revenue"]["mean"],
                         "cost_baseline": b["total_cost"]["mean"], "cost_ai": a["total_cost"]["mean"]})
        df = pd.DataFrame(rows)
        worst = df.sort_values("fill_baseline").iloc[0]
        normal = df.iloc[0]
        findings.append(f"Under normal conditions the next {sc['days']} days simulate to {normal.fill_baseline:.1%} fill rate with the "
                        f"current rule and {normal.fill_ai:.1%} with the AI-assisted policy ({sc['reps']} runs each).")
        findings.append(f"Most damaging preset: '{worst.scenario}' — fill rate {worst.fill_baseline:.1%} (current rule) vs "
                        f"{worst.fill_ai:.1%} (AI-assisted). SIMULATED.")
        d = AgentDecision(agent=self.name, decision_type="INFORMATION", title="Scenario stress test",
                          action="Review the scenario table on the Digital Twin page.", evidence=findings,
                          confidence=1.0, confidence_basis="deterministic simulation summary", risk_level="LOW",
                          estimated_value_usd=0.0, sources={"snapshot": ctx.run("snapshot").version})
        return AgentReport(self.name, findings, [d], ["compare_inventory_policies"], {"rows": rows})
