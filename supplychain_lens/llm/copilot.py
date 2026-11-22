"""'Ask SupplyChain Lens' — question → tool → grounded answer.

Routing is deterministic (keywords and IDs in the question). The answer is built from the tool's
output by templates; if a local Ollama model is available, it may rephrase the same facts, and the
rephrasing is discarded unless every number in it appears in the facts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from supplychain_lens.agents.specialists import money
from supplychain_lens.llm import ollama_client
from supplychain_lens.tools import core
from supplychain_lens.tools.context import Context

SKU = re.compile(r"SKU-\d{4}", re.I)
SUP = re.compile(r"SUP-\d{4}", re.I)
WH = re.compile(r"WH-\d{3}", re.I)

EXAMPLES = [
    "Which suppliers are most at risk?",
    "Where are we likely to stock out in the next two weeks?",
    "What is the demand forecast for SKU-0007?",
    "Any anomalies I should look at?",
    "What happens if SUP-0012 goes offline?",
    "What is the business impact of the AI policy?",
]


@dataclass
class Answer:
    text: str
    tool: str
    facts: str
    used_llm: bool
    rows: list


def route(question: str) -> str:
    q = question.lower()
    if SUP.search(question) and any(w in q for w in ("offline", "what if", "what happens", "disrupt", "fail", "shut")):
        return "simulate_supplier_disruption"
    if any(w in q for w in ("impact", "saving", "benefit", "roi", "worth")):
        return "calculate_business_impact"
    if any(w in q for w in ("anomal", "unusual", "strange", "odd")):
        return "detect_anomalies"
    if any(w in q for w in ("stock out", "stockout", "shortage", "run out", "out of stock")):
        return "calculate_stockout_probability"
    if any(w in q for w in ("supplier", "risk", "disruption")) or SUP.search(question):
        return "score_disruption_risk"
    if any(w in q for w in ("forecast", "demand", "next month", "predict")) or SKU.search(question):
        return "forecast_demand"
    if any(w in q for w in ("policy", "compare", "scenario")):
        return "compare_inventory_policies"
    return "help"


def _facts(tool: str, out: dict, question: str) -> tuple[str, list]:
    s, rows = out["summary"], out.get("rows", [])
    if tool == "score_disruption_risk":
        lines = [f"{s['suppliers_flagged']} suppliers are above the risk threshold {s['threshold']:.2f} ({s['model']}, 14-day horizon)."]
        for r in rows[:5]:
            lines.append(f"{r['supplier_id']} {r['name']}: probability {r['probability']:.0%}, expected loss {money(r['expected_loss_usd'])}"
                         + (f"; {r['drivers'][0]}" if r["drivers"] else ""))
    elif tool == "calculate_stockout_probability":
        lines = [f"{s['series_at_risk_50pct']} of {s['series_total']} SKU-warehouse series have at least 50% chance of a stockout "
                 f"within {s['window_days']} days; expected lost revenue {money(s['expected_lost_revenue_14d'])}."]
        for r in rows[:5]:
            lines.append(f"{r['sku_id']} at {r['warehouse_id']}: {r['stockout_probability_14d']:.0%} chance, "
                         f"{r['days_of_supply']:.1f} days of supply, expected lost margin {money(r['expected_lost_margin_14d'])}")
    elif tool == "forecast_demand":
        lines = [f"Network forecast for the next {s['horizon_days']} days: {s['total_p50_28d']:,.0f} units "
                 f"({s['network_change_pct']:+.1f}% vs the last 28 days), model {s['model']}."]
        for r in rows[:5]:
            lines.append(f"{r['sku_id']} at {r['warehouse_id']}: P50 {r['p50']:,.0f} units (80% band {r['p10']:,.0f}–{r['p90']:,.0f}), "
                         f"{r['change_pct']:+.0f}% vs last 28 days")
    elif tool == "detect_anomalies":
        lines = [f"{s['alerts']} autoencoder alerts in the last {s['window_days']} days (threshold {s['threshold']:.2f}). "
                 "Alerts need investigation; they are not confirmed disruptions."]
        for r in rows[:5]:
            lines.append(f"{r['date']:%d %b}: {r['sku_id']} at {r['warehouse_id']} — {r['reason']} (score {r['score']:.2f})")
    elif tool == "simulate_supplier_disruption":
        lines = [f"Simulated scenario '{s['scenario']}' over {s['days']} days ({s['reps']} runs, SIMULATED):",
                 f"fill rate {s['fill_rate_baseline']:.1%} with the current rule vs {s['fill_rate_ai']:.1%} with the AI-assisted policy "
                 f"(normal conditions {s['fill_rate_normal_conditions']:.1%}).",
                 f"Lost revenue {money(s['lost_revenue_baseline'])} vs {money(s['lost_revenue_ai'])}."]
    elif tool == "calculate_business_impact":
        if not s.get("available"):
            lines = ["The business-impact experiment has not been run yet."]
        else:
            lines = [f"{s['label']}.",
                     f"Replay of {s['period']['start']} to {s['period']['end']} ({s['replications']} runs): fill rate "
                     f"{s['fill_rate_baseline']:.1%} (current rule) vs {s['fill_rate_ai']:.1%} (AI-assisted).",
                     f"Total cost change {money(s['total_cost_change'])} ({s['total_cost_change_pct']:+.1%}); "
                     f"95% interval {money(s['total_cost_ci'][0])} to {money(s['total_cost_ci'][1])}.",
                     f"Average inventory value change {s['inventory_change_pct']:+.1%}."]
    elif tool == "compare_inventory_policies":
        lines = [f"Next {s['days']} days, normal conditions ({s['reps']} runs, SIMULATED): fill rate {s['fill_rate_baseline']:.1%} "
                 f"(current rule) vs {s['fill_rate_ai']:.1%} (AI-assisted)."]
    else:
        lines = []
    return "\n".join(lines), rows


def ask(ctx: Context, question: str, allow_llm: bool = True) -> Answer:
    tool = route(question)
    if tool == "help":
        return Answer("I answer questions with the project's tools. Try: " + " · ".join(EXAMPLES), "help", "", False, [])
    kwargs = {}
    if tool == "forecast_demand" and SKU.search(question):
        kwargs["sku_id"] = SKU.search(question).group(0).upper()
        if WH.search(question):
            kwargs["warehouse_id"] = WH.search(question).group(0).upper()
    if tool in ("score_disruption_risk",) and SUP.search(question):
        kwargs["supplier_id"] = SUP.search(question).group(0).upper()
    if tool == "simulate_supplier_disruption":
        kwargs = {"supplier_id": SUP.search(question).group(0).upper(), "reps": 20}
    out = core.TOOLS[tool](ctx, **kwargs)
    facts, rows = _facts(tool, out, question)
    text = ollama_client.rewrite(facts, question) if allow_llm else None
    return Answer(text or facts, tool, facts, text is not None, rows)
