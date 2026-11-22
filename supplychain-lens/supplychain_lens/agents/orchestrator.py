"""Decision Orchestrator: runs the specialists, ranks their proposals and applies the approval policy.

Order: Demand Intelligence, Anomaly Investigation and Supplier Risk (independent) →
Inventory and Simulation → Decision Orchestrator (rank by expected value and risk, apply
policy.py, write the executive summary). Plain Python with a typed state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from supplychain_lens.agents.policy import apply_policy
from supplychain_lens.agents.schemas import AgentDecision, AgentReport
from supplychain_lens.agents.specialists import (
    AnomalyInvestigationAgent,
    DemandIntelligenceAgent,
    InventoryAgent,
    SimulationAgent,
    SupplierRiskAgent,
    money,
)
from supplychain_lens.tools.context import Context

RISK_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


@dataclass
class OrchestrationResult:
    reports: dict[str, AgentReport]
    decisions: list[AgentDecision]
    executive_summary: list[str]
    kpis: dict
    seconds: float
    trace: list[dict] = field(default_factory=list)


def run_all(ctx: Context) -> OrchestrationResult:
    t0 = time.perf_counter()
    trace = []
    reports = {}
    for agent in (DemandIntelligenceAgent(), AnomalyInvestigationAgent(), SupplierRiskAgent(), InventoryAgent(), SimulationAgent()):
        a0 = time.perf_counter()
        reports[agent.name] = agent.run(ctx)
        trace.append({"agent": agent.name, "tools": reports[agent.name].tool_calls,
                      "decisions": len(reports[agent.name].decisions), "seconds": round(time.perf_counter() - a0, 2)})
    decisions = [apply_policy(d) for r in reports.values() for d in r.decisions]
    decisions.sort(key=lambda d: (d.decision_type == "INFORMATION", RISK_ORDER.get(d.risk_level, 9), -abs(d.estimated_value_usd)))

    k = ctx.table("kpi_daily")
    last, prev = k.iloc[-1], k.iloc[-29:-1]
    kpis = {"date": str(pd.Timestamp(last.date).date()), "fill_rate_pct": float(last.fill_rate_pct),
            "fill_rate_28d_avg": float(prev.fill_rate_pct.mean()), "stockout_rate_pct": float(last.stockout_rate_pct),
            "otif_pct": float(last.otif_pct), "inventory_value_usd": float(last.inventory_value_usd),
            "active_disruptions": int(last.active_disruptions)}
    actionable = [d for d in decisions if d.decision_type not in ("INFORMATION", "INVESTIGATION")]
    pending = [d for d in actionable if d.requires_human_approval]
    summary = [
        f"On {kpis['date']} the network filled {kpis['fill_rate_pct']:.1f}% of demand (28-day average "
        f"{kpis['fill_rate_28d_avg']:.1f}%) with {kpis['active_disruptions']} active recorded disruptions.",
        reports["Inventory"].findings[0], reports["Supplier Risk"].findings[0],
        f"{len(actionable)} actions proposed ({len(pending)} need approval) and "
        f"{sum(d.decision_type == 'INVESTIGATION' for d in decisions)} anomalies to investigate; "
        f"value at stake in the proposed actions {money(sum(d.estimated_value_usd for d in actionable))}.",
    ]
    trace.append({"agent": "Decision Orchestrator", "tools": ["policy.apply_policy"], "decisions": len(decisions),
                  "seconds": round(time.perf_counter() - t0, 2)})
    return OrchestrationResult(reports, decisions, summary, kpis, round(time.perf_counter() - t0, 2), trace)
