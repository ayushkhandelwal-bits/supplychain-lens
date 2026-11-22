"""Re-balance stock of one SKU between its warehouses with OR-Tools min-cost flow.

For each warehouse stocking the SKU:
  need     = P90 demand over the cover window − (on hand + receipts expected in the window)
  surplus  = on hand + expected receipts − P90 demand over the cover window − a keep-back buffer

The network sends surplus units to warehouses with need when that is cheaper than losing
the sale. Unmet need flows from a dummy node at a cost equal to the lost margin, and unused
surplus flows to a dummy sink at zero cost, so the solver maximises net benefit:
    minimise  Σ transfer cost  +  Σ lost margin on unmet need

Cost assumptions (documented, adjustable): transfer cost per unit = 1% of unit cost plus
$0.04 per 100 km; transfer time 2 days within a region, 6 days across regions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

COST_SHARE_OF_UNIT = 0.01
COST_PER_100KM = 0.04
DAYS_SAME_REGION, DAYS_CROSS_REGION = 2, 6


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


@dataclass
class TransferPlan:
    sku_id: str
    moves: pd.DataFrame          # from_wh, to_wh, units, unit_cost_usd, total_cost_usd, transit_days, distance_km
    total_transfer_cost: float
    unmet_units_before: float
    unmet_units_after: float
    lost_margin_avoided: float
    net_benefit: float
    solver_status: str


def plan_transfers(sku_id: str, positions: pd.DataFrame, warehouses: pd.DataFrame, unit_cost: float,
                   unit_margin: float, keep_back_share: float = 0.15) -> TransferPlan:
    """positions: one row per warehouse with columns warehouse_id, on_hand, arriving, demand_p90."""
    from ortools.graph.python import min_cost_flow

    wh = warehouses.set_index("warehouse_id")
    pos = positions.reset_index(drop=True).copy()
    avail = pos.on_hand + pos.arriving
    pos["need"] = np.maximum(0, np.round(pos.demand_p90 - avail)).astype(int)
    pos["surplus"] = np.maximum(0, np.round(avail - pos.demand_p90 * (1 + keep_back_share))).astype(int)
    pos["surplus"] = np.minimum(pos.surplus, np.maximum(0, pos.on_hand).astype(int))  # can only ship what is on hand
    unmet_before = float(pos.need.sum())
    empty = pd.DataFrame(columns=["from_wh", "to_wh", "units", "unit_cost_usd", "total_cost_usd", "transit_days", "distance_km"])
    if unmet_before == 0 or pos.surplus.sum() == 0:
        return TransferPlan(sku_id, empty, 0.0, unmet_before, unmet_before, 0.0, 0.0, "NOTHING_TO_MOVE")

    n = len(pos)
    dummy_src, dummy_sink = n, n + 1
    smcf = min_cost_flow.SimpleMinCostFlow()
    scale = 100  # costs in cents (solver needs integers)
    arc_info = {}
    for i in range(n):
        for j in range(n):
            if i == j or pos.surplus[i] == 0 or pos.need[j] == 0:
                continue
            a, b = wh.loc[pos.warehouse_id[i]], wh.loc[pos.warehouse_id[j]]
            km = haversine_km(a.lat, a.lng, b.lat, b.lng)
            per_unit = unit_cost * COST_SHARE_OF_UNIT + COST_PER_100KM * km / 100
            if per_unit >= unit_margin:
                continue  # never worth moving
            arc = smcf.add_arc_with_capacity_and_unit_cost(i, j, int(pos.surplus[i]), int(round(per_unit * scale)))
            arc_info[arc] = (i, j, per_unit, km, DAYS_SAME_REGION if a.region == b.region else DAYS_CROSS_REGION)
    for j in range(n):
        if pos.need[j]:
            smcf.add_arc_with_capacity_and_unit_cost(dummy_src, j, int(pos.need[j]), int(round(unit_margin * scale)))
    for i in range(n):
        if pos.surplus[i]:
            smcf.add_arc_with_capacity_and_unit_cost(i, dummy_sink, int(pos.surplus[i]), 0)
    # Lets the dummy source's unused supply drain away when real transfers cover the need.
    smcf.add_arc_with_capacity_and_unit_cost(dummy_src, dummy_sink, int(pos.need.sum()), 0)
    for i in range(n):
        smcf.set_node_supply(i, int(pos.surplus[i] - pos.need[i]))  # a node has either need or surplus
    smcf.set_node_supply(dummy_src, int(pos.need.sum()))
    smcf.set_node_supply(dummy_sink, -int(pos.surplus.sum()))
    status = smcf.solve()
    if status != smcf.OPTIMAL:
        return TransferPlan(sku_id, empty, 0.0, unmet_before, unmet_before, 0.0, 0.0, f"SOLVER_{status}")
    rows = []
    for arc, (i, j, per_unit, km, days) in arc_info.items():
        f = smcf.flow(arc)
        if f > 0:
            rows.append({"from_wh": pos.warehouse_id[i], "to_wh": pos.warehouse_id[j], "units": int(f),
                         "unit_cost_usd": round(per_unit, 2), "total_cost_usd": round(f * per_unit, 2),
                         "transit_days": days, "distance_km": round(km)})
    moves = pd.DataFrame(rows, columns=empty.columns)
    moved = float(moves.units.sum()) if len(moves) else 0.0
    cost = float(moves.total_cost_usd.sum()) if len(moves) else 0.0
    avoided = moved * unit_margin
    return TransferPlan(sku_id, moves, cost, unmet_before, unmet_before - moved, avoided, avoided - cost, "OPTIMAL")
