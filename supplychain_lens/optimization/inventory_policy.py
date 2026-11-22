"""Classical inventory formulas used to explain and size replenishment decisions.

  eoq                  economic order quantity √(2·D·K / h): balances ordering vs holding cost
  base_stock_level     μ_cover + z·σ_cover — the order-up-to level the AI-assisted policy uses
  nexus_baseline_level trailing mean × (L + R) × 1.15 — the historical NEXUS-style rule
The simulation that compares the policies lives in supplychain_lens/simulation/twin.py.
"""

from __future__ import annotations

import math

from supplychain_lens.simulation.twin import (
    HOLDING_RATE_PER_YEAR,
    ORDER_COST_USD,
    REVIEW_PERIOD,
    SAFETY_FACTOR_BASELINE,
    SERVICE_Z_AI,
)


def eoq(annual_demand_units: float, unit_cost: float, order_cost: float = ORDER_COST_USD,
        holding_rate: float = HOLDING_RATE_PER_YEAR) -> float:
    h = unit_cost * holding_rate
    return math.sqrt(2 * annual_demand_units * order_cost / h) if h > 0 and annual_demand_units > 0 else 0.0


def base_stock_level(mean_daily: float, std_daily: float, lead_time: float, lead_time_std: float,
                     review_period: int = REVIEW_PERIOD, z: float = SERVICE_Z_AI) -> float:
    cover = lead_time + review_period
    return mean_daily * cover + z * math.sqrt(cover * std_daily**2 + mean_daily**2 * lead_time_std**2)


def nexus_baseline_level(trailing_mean: float, lead_time: float, review_period: int = REVIEW_PERIOD) -> float:
    return trailing_mean * (lead_time + review_period) * (1 + SAFETY_FACTOR_BASELINE)
