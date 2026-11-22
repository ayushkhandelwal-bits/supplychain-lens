# Auto-approval thresholds follow NEXUS-SCCT orchestrator.py (confidence ≥ 0.85, not CRITICAL,
# value < $5M); extended with a reversibility rule. Implemented as code, not as an LLM judgement.
"""Human-in-the-loop approval policy."""

from __future__ import annotations

from supplychain_lens.agents.schemas import AgentDecision

AUTO_APPROVE_MIN_CONFIDENCE = 0.85
AUTO_APPROVE_MAX_VALUE_USD = 5_000_000


def apply_policy(d: AgentDecision) -> AgentDecision:
    """Set status / requires_human_approval / approval_reason in place and return the decision."""
    if d.decision_type == "INVESTIGATION":
        d.status, d.requires_human_approval = "INVESTIGATION_REQUIRED", True
        d.approval_reason = "Model alert, not a confirmed event: a person must investigate."
        return d
    if d.decision_type == "INFORMATION":
        d.status, d.requires_human_approval, d.approval_reason = "INFORMATION", False, "No action proposed."
        return d
    reasons = []
    if d.confidence < AUTO_APPROVE_MIN_CONFIDENCE:
        reasons.append(f"confidence {d.confidence:.0%} is below {AUTO_APPROVE_MIN_CONFIDENCE:.0%}")
    if d.risk_level == "CRITICAL":
        reasons.append("risk level is CRITICAL")
    if abs(d.estimated_value_usd) >= AUTO_APPROVE_MAX_VALUE_USD:
        reasons.append(f"value at stake ≥ ${AUTO_APPROVE_MAX_VALUE_USD:,.0f}")
    if not d.reversible:
        reasons.append("action is not easily reversible")
    if reasons:
        d.status, d.requires_human_approval = "PENDING_APPROVAL", True
        d.approval_reason = "Needs approval: " + "; ".join(reasons) + "."
    else:
        d.status, d.requires_human_approval = "AUTO_APPROVED", False
        d.approval_reason = "Auto-approved: high confidence, reversible, below the value limit, not critical."
    return d


def risk_level_from_value(value_usd: float, probability: float) -> str:
    if value_usd >= 5_000_000 or probability >= 0.9:
        return "CRITICAL"
    if value_usd >= 1_000_000 or probability >= 0.6:
        return "HIGH"
    if value_usd >= 100_000 or probability >= 0.3:
        return "MEDIUM"
    return "LOW"
