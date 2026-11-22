# ADR-0005 — Six tool-grounded agents instead of eight LLM-only agents

- **Status:** Accepted (revised 2026-10-27: 5 → 6 agents after the deep-learning change request)
- **Date:** 2026-10-03

## Context

In NEXUS-SCCT, each of the 8 LangGraph nodes sends a role prompt plus the disruption to Claude and asks it to return a JSON decision (audit check A4). Consequences:

- Quantities, costs and values are invented by the LLM, not computed.
- `confidence` is the LLM's own guess, yet it decides auto-approval (≥ 0.85).
- The ML engines (risk model, forecaster, optimizer, router) are never called.
- The graph is a linear chain of 8 sequential LLM calls (check A5): slow and costly.
- Escalation is a fixed policy (value and severity thresholds) but is partly delegated to an LLM.

The deep-learning change request adds two needs: an agent that investigates **autoencoder anomalies**, and an agent that runs **digital-twin scenarios**.

## Decision

Agents are Python classes that **call deterministic tools** (which in turn use the trained models) and return the *same* `AgentDecision` schema NEXUS defined: decision type, action, reasoning, confidence, value, risk level, status, alternatives and approval flag. The governance concept is preserved.

| Agent | Replaces NEXUS agents | Distinct responsibility | Main tools |
|---|---|---|---|
| **Demand Intelligence** | Demand Forecasting | Read forecasts and their uncertainty; flag SKUs with elevated demand or wide intervals | `forecast_demand()` |
| **Anomaly Investigation** | *(new)* | Separate model alerts from verified events; gather the operational records behind an alert | `detect_anomalies()` |
| **Supplier Risk** | Supplier Risk | Rank suppliers by predicted disruption probability × exposure; trace affected SKUs and warehouses | `score_disruption_risk()`, cascade graph, mode/reroute options (absorbs the old Logistics agent) |
| **Inventory** | Inventory | Shortage risk and replenishment / transfer options with their trade-offs | `calculate_stockout_probability()`, `compare_inventory_policies()` |
| **Simulation** | *(new)* | Run baseline vs scenario in the digital twin; return KPI deltas with uncertainty | `simulate_supplier_disruption()` and the scenario engine |
| **Decision Orchestrator** | Planner + Escalation + Executive Summary + KPI Monitor | Combine the evidence, rank actions by expected value, apply the approval policy, write the summary | `calculate_business_impact()`, `policy.py`, narrator |

The seven tool names come from the change request and are kept exactly as written: `forecast_demand()`, `score_disruption_risk()`, `detect_anomalies()`, `calculate_stockout_probability()`, `simulate_supplier_disruption()`, `compare_inventory_policies()` and `calculate_business_impact()`.

**Confidence is computed, not self-reported.** It comes from model outputs, such as a calibrated probability or the width of the forecast interval. **Approval** is a rule in `policy.py`, unit-tested.

**Execution order:** Demand, Anomaly and Supplier Risk run first (they're independent), then Inventory and Simulation use their outputs, and the Orchestrator synthesizes. The orchestrator is plain Python with a typed state. LangGraph is free and could be swapped in later; it isn't needed to show the pattern.

## Consequences

- Recommendations are reproducible and auditable: same data and models → same decision.
- The system works with no LLM at all (ADR-0004).
- Each agent has a distinct responsibility. KPI monitoring, escalation and summarizing are a threshold, a rule and a formatter, so they live inside the Orchestrator rather than as separate "reasoners".
