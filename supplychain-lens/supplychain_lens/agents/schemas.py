# Decision schema and statuses adapted from NEXUS-SCCT orchestrator.py (AgentDecision);
# reimplemented — confidence here is computed from model/simulation outputs, never self-reported.
"""Structured outputs shared by all agents."""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field

_ids = itertools.count(1)

STATUSES = ("AUTO_APPROVED", "PENDING_APPROVAL", "APPROVED", "REJECTED", "INVESTIGATION_REQUIRED", "INFORMATION")
RISK_LEVELS = ("LOW", "MEDIUM", "HIGH", "CRITICAL")


@dataclass
class Alternative:
    action: str
    cost_usd: float | None = None
    expected_benefit_usd: float | None = None
    note: str = ""


@dataclass
class AgentDecision:
    agent: str
    decision_type: str              # e.g. INVENTORY_TRANSFER, SUPPLIER_MITIGATION, INVESTIGATION, ORDER_ADJUSTMENT
    title: str
    action: str
    evidence: list[str]             # every number here came from a tool
    confidence: float               # computed; meaning is stated in confidence_basis
    confidence_basis: str
    risk_level: str
    estimated_value_usd: float      # expected net benefit of the action (0 if not monetised)
    reversible: bool = True
    expected_impact: dict = field(default_factory=dict)
    alternatives: list[Alternative] = field(default_factory=list)
    affected: dict = field(default_factory=dict)
    uncertainty: str = ""
    sources: dict = field(default_factory=dict)
    status: str = "PENDING_APPROVAL"
    requires_human_approval: bool = True
    approval_reason: str = ""
    decision_id: str = field(default_factory=lambda: f"DEC-{next(_ids):04d}")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentReport:
    agent: str
    findings: list[str]
    decisions: list[AgentDecision]
    tool_calls: list[str]
    data: dict = field(default_factory=dict)
