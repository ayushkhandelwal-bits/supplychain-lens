# Inventory cost parameters and the baseline rule follow NEXUS-SCCT rl_optimizer.py / seed_data.py
# (holding cost 25%/yr, 15% safety stock); reimplemented and extended.
"""Digital twin of the replenishment network, vectorised over Monte Carlo replications.

The same engine runs (a) the business-impact backtest over the test quarter with ACTUAL demand
and (b) forward what-if scenarios from the snapshot with demand sampled from forecast quantiles.

World model (calibrated from the observable data, identical for every policy):
  • pipeline orders that exist at the start arrive on their known (backtest) or estimated
    (forward: promised date + sampled delay) dates
  • a new order's lead time = supplier's nominal lead time + a delay drawn from that supplier's
    own recent delivery history + extra days during recorded disruptions; shipped quantity is
    cut by the disruption's capacity factor
  • lost sales are lost (no backorders)

Policies (decision makers — they only see information available on the review day):
  baseline  the historical NEXUS-style rule: weekly order-up-to trailing-28-day mean ×
            (nominal lead time + review period) × 1.15, always from the primary supplier
  ai        forecast-driven order-up-to: Σ P50 forecast over lead time + review period, plus
            safety stock z·√(Σσ²_forecast + d̄²·σ²_lead-time); switches to the backup supplier
            when the primary has a recorded active disruption or a predicted disruption risk at
            or above the classifier threshold (backup costs +8%)

Common random numbers: both policies see the same demand paths and delay draws per replication,
so differences come from the decisions, not from luck.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from supplychain_lens.config import get_settings

HOLDING_RATE_PER_YEAR = 0.25     # NEXUS rl_optimizer holding cost
ORDER_COST_USD = 500.0           # fixed cost per purchase order (assumption)
BACKUP_PREMIUM = 0.08            # backup supplier unit-cost premium (assumption)
OVERFLOW_COST_PER_UNIT_DAY = 0.001  # share of unit cost per day for stock above warehouse capacity
REVIEW_PERIOD = 7
SAFETY_FACTOR_BASELINE = 0.15    # NEXUS safety_stock_pct
SERVICE_Z_AI = 1.65              # ~95% cycle service level


@dataclass(frozen=True)
class Scenario:
    name: str = "Baseline conditions"
    demand_change_pct: float = 0.0
    lead_time_add_days: int = 0
    supplier_capacity_pct: float = 100.0
    warehouse_capacity_pct: float = 100.0
    transport_delay_days: int = 0
    offline_supplier: str | None = None
    affected_supplier: str | None = None  # None = lead-time/capacity changes apply to all suppliers

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TwinState:
    keys: pd.DataFrame                 # per series: sku_id, warehouse_id, category, primary/backup supplier, costs
    supplier_ids: np.ndarray
    start_date: pd.Timestamp
    days: int
    on_hand0: np.ndarray               # (S,)
    pipe_series: np.ndarray            # open orders at start
    pipe_day: np.ndarray               # relative arrival day (may be ≥ days)
    pipe_qty: np.ndarray
    pipe_supplier: np.ndarray
    pipe_promised: np.ndarray          # relative promised day
    demand_hist: np.ndarray            # (S, 28) demand before start
    nominal_lt: np.ndarray             # (n_sup,)
    delay_pool: list                   # per supplier: array of recent delays
    delay_std: np.ndarray              # (n_sup,)
    env_cap: np.ndarray                # (n_sup, days) capacity factor from recorded disruptions
    env_lt_add: np.ndarray             # (n_sup, days)
    known_disrupted: np.ndarray        # (n_sup, days) bool: recorded & detected by that day
    wh_capacity: np.ndarray            # (S,) capacity of the series' warehouse
    wh_index: np.ndarray               # (S,) warehouse index
    n_wh: int
    review_weekday: np.ndarray
    dows: np.ndarray                   # (days,)
    pipeline_known: bool = True        # True in the backtest (actual arrival dates)
    extra: dict = field(default_factory=dict)


def build_state(start: str | pd.Timestamp, days: int, data_dir: Path | None = None,
                known_arrivals: bool = True) -> TwinState:
    """State at the START of ``start`` (i.e. end of the previous day)."""
    data_dir = data_dir or get_settings().data_dir / "nexus_v2"
    start = pd.Timestamp(start)
    prev = start - pd.Timedelta(days=1)
    sw = pd.read_parquet(data_dir / "sku_warehouse.parquet")
    products = pd.read_parquet(data_dir / "products.parquet")
    sup = pd.read_parquet(data_dir / "suppliers.parquet")
    wh = pd.read_parquet(data_dir / "warehouses.parquet").reset_index(drop=True)
    inv = pd.read_parquet(data_dir / "inventory_daily.parquet", filters=[("date", "==", prev)])
    dem = pd.read_parquet(data_dir / "demand_daily.parquet", columns=["date", "sku_id", "warehouse_id", "demand_units"],
                          filters=[("date", ">", prev - pd.Timedelta(days=28)), ("date", "<=", prev)])
    po = pd.read_parquet(data_dir / "purchase_orders.parquet")
    sh = pd.read_parquet(data_dir / "shipments.parquet")
    dis = pd.read_parquet(data_dir / "disruptions.parquet")
    bridge = pd.read_parquet(data_dir / "disruption_suppliers.parquet")

    keys = sw.sort_values(["sku_id", "warehouse_id"]).reset_index(drop=True).merge(
        products[["sku_id", "category", "primary_supplier_id", "backup_supplier_id", "unit_cost", "list_price"]], on="sku_id")
    S = len(keys)
    sidx = {s: i for i, s in enumerate(sup.supplier_id)}
    widx = {w: i for i, w in enumerate(wh.warehouse_id)}
    key_idx = {(a, b): i for i, (a, b) in enumerate(zip(keys.sku_id, keys.warehouse_id, strict=True))}
    keys["primary_idx"] = keys.primary_supplier_id.map(sidx)
    keys["backup_idx"] = keys.backup_supplier_id.map(sidx)

    on_hand0 = np.zeros(S)
    for r in inv.itertuples():
        on_hand0[key_idx[(r.sku_id, r.warehouse_id)]] = r.on_hand_units
    hist = dem.pivot_table(index=["sku_id", "warehouse_id"], columns="date", values="demand_units").reindex(
        pd.MultiIndex.from_frame(keys[["sku_id", "warehouse_id"]])).to_numpy(float)

    # Open orders at the start: shipped before start, not arrived by prev.
    open_ = po.merge(sh[["po_id", "quantity_units", "actual_arrival"]], on="po_id")
    open_ = open_[(open_.order_date <= prev) & ((open_.actual_arrival.isna()) | (open_.actual_arrival > prev))
                  & (open_.quantity_units > 0)]
    rel = lambda d: (pd.to_datetime(d) - start).dt.days.to_numpy()  # noqa: E731
    arrival = open_.actual_arrival if known_arrivals else pd.Series(pd.NaT, index=open_.index)

    # Supplier delay history: 180 days before start (observable).
    dl = sh[(sh.status == "DELIVERED") & (pd.to_datetime(sh.actual_arrival) <= prev)
            & (pd.to_datetime(sh.actual_arrival) > prev - pd.Timedelta(days=180))]
    pool = [dl.loc[dl.supplier_id == s, "delay_days"].to_numpy(float) for s in sup.supplier_id]
    pool = [p if len(p) >= 5 else np.array([0.0, 0, 1, -1, 2]) for p in pool]

    env_cap, env_lt, known = np.ones((len(sup), days)), np.zeros((len(sup), days)), np.zeros((len(sup), days), bool)
    ev = bridge.merge(dis[["event_id", "detected_at", "resolved_at", "capacity_factor", "lead_time_add_days", "event_type"]],
                      on="event_id")
    for e in ev[ev.event_type != "DEMAND"].itertuples():
        a = (pd.Timestamp(e.detected_at) - start).days
        b = (pd.Timestamp(e.resolved_at) - start).days if pd.notna(e.resolved_at) else days - 1
        if b < 0 or a >= days:
            continue
        s, lo, hi = sidx[e.supplier_id], max(0, a), min(days - 1, b)
        env_cap[s, lo : hi + 1] = np.minimum(env_cap[s, lo : hi + 1], e.capacity_factor)
        env_lt[s, lo : hi + 1] = np.maximum(env_lt[s, lo : hi + 1], e.lead_time_add_days)
        known[s, lo : hi + 1] = True

    wh_cap = wh.set_index("warehouse_id").capacity_units
    return TwinState(
        keys=keys, supplier_ids=sup.supplier_id.to_numpy(), start_date=start, days=days, on_hand0=on_hand0,
        pipe_series=np.array([key_idx[(a, b)] for a, b in zip(open_.sku_id, open_.warehouse_id, strict=True)], dtype=int),
        pipe_day=np.where(arrival.notna(), rel(arrival.fillna(start)), -1),
        pipe_qty=open_.quantity_units.to_numpy(float), pipe_supplier=open_.supplier_id.map(sidx).to_numpy(),
        pipe_promised=rel(open_.promised_date), demand_hist=np.nan_to_num(hist), nominal_lt=sup.avg_lead_time_days.to_numpy(float),
        delay_pool=pool, delay_std=np.array([np.std(p) for p in pool]), env_cap=env_cap, env_lt_add=env_lt,
        known_disrupted=known, wh_capacity=keys.warehouse_id.map(wh_cap).to_numpy(float),
        wh_index=keys.warehouse_id.map(widx).to_numpy(), n_wh=len(wh), review_weekday=keys.review_weekday.to_numpy(),
        dows=pd.date_range(start, periods=days).dayofweek.to_numpy(), pipeline_known=known_arrivals)


def simulate(state: TwinState, demand: np.ndarray, policy: str, scenario: Scenario | None = None,
             forecasts: np.ndarray | None = None, risk: np.ndarray | None = None, risk_threshold: float = 1.0,
             seed: int = 42) -> dict:
    """Run one policy. ``demand`` is (R, S, days) units.

    forecasts: (days, S, H, 3) quantiles available at each day (required for policy 'ai')
    risk:      (days, n_sup) predicted disruption probability available at each day (optional)
    """
    scenario = scenario or Scenario()
    R, S, D = demand.shape
    rng = np.random.default_rng(seed)  # same seed for both policies -> common random numbers
    n_sup = len(state.supplier_ids)
    horizon_pad = D + 200
    demand = demand * (1 + scenario.demand_change_pct / 100)

    # Scenario modifiers on the environment
    affected = np.ones(n_sup, bool) if scenario.affected_supplier is None else (state.supplier_ids == scenario.affected_supplier)
    cap = state.env_cap.copy()
    cap[affected] *= scenario.supplier_capacity_pct / 100
    lt_add = state.env_lt_add.copy()
    lt_add[affected] += scenario.lead_time_add_days
    lt_add += scenario.transport_delay_days
    offline = state.supplier_ids == scenario.offline_supplier if scenario.offline_supplier else np.zeros(n_sup, bool)
    cap[offline] = 0.0
    known = state.known_disrupted | offline[:, None]
    wh_cap = state.wh_capacity * scenario.warehouse_capacity_pct / 100

    # Pre-draw delays (common random numbers: fixed by seed, independent of policy decisions)
    delay_u = rng.random((R, S, D))
    pipe_u = rng.random((R, len(state.pipe_qty)))

    arrivals = np.zeros((R, S, horizon_pad))
    on_order = np.zeros((R, S))
    for k in range(len(state.pipe_qty)):
        s, sup_k = state.pipe_series[k], state.pipe_supplier[k]
        if state.pipeline_known and state.pipe_day[k] >= 0:
            day = np.full(R, state.pipe_day[k])
        else:
            pool = state.delay_pool[sup_k]
            day = state.pipe_promised[k] + pool[(pipe_u[:, k] * len(pool)).astype(int)]
        day = day + scenario.transport_delay_days + (scenario.lead_time_add_days if affected[sup_k] else 0)
        if offline[sup_k]:
            day = np.full(R, horizon_pad - 1)  # never arrives within the horizon
        day = np.clip(np.round(day), 0, horizon_pad - 1).astype(int)
        arrivals[np.arange(R), s, day] += state.pipe_qty[k]
        on_order[:, s] += state.pipe_qty[k]

    on_hand = np.repeat(state.on_hand0[None], R, axis=0)
    hist = np.repeat(state.demand_hist[None], R, axis=0)          # rolling 28-day window for the baseline rule
    unit_cost, price = state.keys.unit_cost.to_numpy(), state.keys.list_price.to_numpy()
    primary, backup = state.keys.primary_idx.to_numpy(), state.keys.backup_idx.to_numpy()

    rec = {k: np.zeros((R, D)) for k in ("demand", "sales", "lost_units", "lost_revenue", "lost_margin", "holding",
                                          "ordering", "premium", "overflow", "inventory_value", "stockout_series")}
    orders_placed = np.zeros(R)
    backup_orders = np.zeros(R)
    for t in range(D):
        arr_t = arrivals[:, :, t]
        on_hand += arr_t
        on_order -= arr_t
        d = demand[:, :, t]
        sales = np.minimum(on_hand, d)
        lost = d - sales
        on_hand -= sales
        hist = np.concatenate([hist[:, :, 1:], d[:, :, None]], axis=2)

        due = np.flatnonzero(state.review_weekday == state.dows[t])
        if len(due):
            sup_choice = primary[due].copy()
            if policy == "baseline":
                rate = hist[:, due].mean(axis=2)
                target = rate * (state.nominal_lt[sup_choice] + REVIEW_PERIOD) * (1 + SAFETY_FACTOR_BASELINE)
            elif policy == "ai":
                p_risk = risk[t] if risk is not None else np.zeros(n_sup)
                switch = (known[primary[due], t] | (p_risk[primary[due]] >= risk_threshold)) & ~known[backup[due], t]
                sup_choice = np.where(switch, backup[due], primary[due])
                cover = (state.nominal_lt[sup_choice] + REVIEW_PERIOD).astype(int)
                q = forecasts[t][due]                                   # (n_due, H, 3)
                H = q.shape[1]
                p50, sig = q[..., 1], (q[..., 2] - q[..., 0]) / 2.563
                mean_tail, var_tail = p50.mean(1), (sig**2).mean(1)
                mu = p50.sum(1) + np.maximum(0, cover - H) * mean_tail
                var = (sig**2).sum(1) + np.maximum(0, cover - H) * var_tail
                var += (mu / np.maximum(cover, 1)) ** 2 * state.delay_std[sup_choice] ** 2
                target = np.broadcast_to(mu + SERVICE_Z_AI * np.sqrt(var), (R, len(due)))
            else:
                raise ValueError(policy)
            qty = np.maximum(0.0, np.round(target - (on_hand[:, due] + on_order[:, due])))
            placed = qty > 0
            shipped = qty * cap[sup_choice, t]
            pool_draw = np.stack([state.delay_pool[s_][(delay_u[:, i, t] * len(state.delay_pool[s_])).astype(int)]
                                  for i, s_ in zip(due, sup_choice, strict=True)], axis=1)
            lt = state.nominal_lt[sup_choice] + pool_draw + lt_add[sup_choice, t]
            arr_day = np.clip(t + np.round(lt).astype(int), t + 1, horizon_pad - 1)
            r_idx = np.repeat(np.arange(R), len(due))
            np.add.at(arrivals, (r_idx, np.tile(due, R), arr_day.reshape(-1)), shipped.reshape(-1))
            on_order[:, due] += shipped
            is_backup = sup_choice != primary[due]
            cost = qty * unit_cost[due]
            rec["ordering"][:, t] = placed.sum(1) * ORDER_COST_USD
            rec["premium"][:, t] = (cost * is_backup * BACKUP_PREMIUM).sum(1)
            orders_placed += placed.sum(1)
            backup_orders += (placed & is_backup).sum(1)

        wh_stock = np.zeros((R, state.n_wh))
        np.add.at(wh_stock, (slice(None), state.wh_index), on_hand)
        cap_by_wh = np.zeros(state.n_wh)
        cap_by_wh[state.wh_index] = wh_cap
        over_units = np.maximum(0, wh_stock - cap_by_wh)
        avg_cost_wh = np.bincount(state.wh_index, unit_cost, state.n_wh) / np.maximum(np.bincount(state.wh_index, minlength=state.n_wh), 1)
        rec["overflow"][:, t] = (over_units * avg_cost_wh * OVERFLOW_COST_PER_UNIT_DAY).sum(1)
        rec["demand"][:, t] = d.sum(1)
        rec["sales"][:, t] = sales.sum(1)
        rec["lost_units"][:, t] = lost.sum(1)
        rec["lost_revenue"][:, t] = (lost * price).sum(1)
        rec["lost_margin"][:, t] = (lost * (price - unit_cost)).sum(1)
        rec["inventory_value"][:, t] = (on_hand * unit_cost).sum(1)
        rec["holding"][:, t] = rec["inventory_value"][:, t] * HOLDING_RATE_PER_YEAR / 365
        rec["stockout_series"][:, t] = (lost > 0).sum(1)
    return summarise(rec, S, orders_placed, backup_orders)


def summarise(rec: dict, S: int, orders: np.ndarray, backups: np.ndarray) -> dict:
    total_cost = rec["holding"].sum(1) + rec["lost_margin"].sum(1) + rec["ordering"].sum(1) + rec["premium"].sum(1) + rec["overflow"].sum(1)
    fill_daily = rec["sales"] / np.maximum(rec["demand"], 1)
    roll = pd.DataFrame(fill_daily.T).rolling(7, min_periods=7).mean().to_numpy().T
    recovered = np.array([int(np.argmax(r >= 0.95)) if (r >= 0.95).any() else -1 for r in np.nan_to_num(roll)])
    per_rep = {
        "fill_rate": rec["sales"].sum(1) / np.maximum(rec["demand"].sum(1), 1),
        "stockout_rate": rec["stockout_series"].sum(1) / (S * rec["demand"].shape[1]),
        "lost_units": rec["lost_units"].sum(1), "lost_revenue": rec["lost_revenue"].sum(1),
        "holding_cost": rec["holding"].sum(1), "lost_margin": rec["lost_margin"].sum(1),
        "ordering_cost": rec["ordering"].sum(1), "backup_premium": rec["premium"].sum(1),
        "overflow_cost": rec["overflow"].sum(1), "total_cost": total_cost,
        "avg_inventory_value": rec["inventory_value"].mean(1), "end_inventory_value": rec["inventory_value"][:, -1],
        "orders": orders, "backup_orders": backups, "recovery_day": recovered,
    }
    daily = {k: rec[k].mean(0) for k in ("demand", "sales", "lost_revenue", "inventory_value")}
    daily["fill_rate"] = fill_daily.mean(0)
    return {"per_rep": per_rep, "daily": daily}


def quantile_demand_paths(q: np.ndarray, reps: int, seed: int = 42) -> np.ndarray:
    """Sample demand paths (R, S, H) from P10/P50/P90 using a lognormal fitted per day.

    A common per-replication level shock (σ = 5%) adds realistic correlation across series.
    """
    rng = np.random.default_rng(seed)
    p10, p50, p90 = (np.maximum(q[..., i], 1e-3) for i in range(3))
    sigma = np.clip((np.log(p90) - np.log(p10)) / 2.563, 0.01, 2.0)
    z = rng.standard_normal((reps,) + p50.shape)
    level = rng.normal(1, 0.05, (reps, 1, 1))
    return np.round(p50[None] * np.exp(sigma[None] * z) * level).clip(0)
