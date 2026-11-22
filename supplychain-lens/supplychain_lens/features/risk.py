"""Supplier × day features for disruption-risk prediction (observable data only, no look-ahead).

Each row describes a supplier on day t using only information available at the end of day t.
Deliveries count once they have ARRIVED (their delay is only known then); a purchase order's
shipped quantity counts once its ship date has passed; disruptions count once DETECTED.

Features (all trailing):
  delay_mean_28 / _90, late_share_28 / _90, delay_p90_90   delivery performance
  delay_trend                                             28-day minus 90-day mean delay
  lead_time_cv_90                                         lead-time variability
  overdue_open, max_overdue_days                          open orders past their promised date
  fill_ratio_28                                           shipped / ordered (capacity cuts)
  country_delay_14                                        peers in the same country (congestion)
  country_active_events, region_active_events             recorded disruptions nearby
  own_active_event, days_since_own_event                  supplier's own recorded history
  country_risk_base, tier, health_score, n_certifications static master data
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from supplychain_lens.config import get_settings
from supplychain_lens.data import reference as ref

FEATURES = ["delay_mean_28", "delay_mean_90", "late_share_28", "late_share_90", "delay_p90_90", "delay_trend",
            "lead_time_cv_90", "overdue_open", "max_overdue_days", "fill_ratio_28", "country_delay_14",
            "country_active_events", "region_active_events", "own_active_event", "days_since_own_event",
            "country_risk_base", "tier", "health_score", "n_certifications"]


@dataclass
class RiskPanel:
    suppliers: pd.DataFrame
    dates: pd.DatetimeIndex
    X: np.ndarray  # (n_suppliers, T, F)


def _rolling(values_by_day: np.ndarray, counts_by_day: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Trailing sums over n days ending today (inclusive) along axis 1."""
    def roll(a):
        c = np.cumsum(np.pad(a, ((0, 0), (1, 0))), axis=1)
        T = a.shape[1]
        return c[:, 1:] - c[:, np.clip(np.arange(T) + 1 - n, 0, None)]
    return roll(values_by_day), roll(counts_by_day)


def build_risk_panel(data_dir: Path | None = None) -> RiskPanel:
    data_dir = data_dir or get_settings().data_dir / "nexus_v2"
    sup = pd.read_parquet(data_dir / "suppliers.parquet").reset_index(drop=True)
    po = pd.read_parquet(data_dir / "purchase_orders.parquet")
    sh = pd.read_parquet(data_dir / "shipments.parquet").merge(po[["po_id", "order_date", "promised_date", "qty_ordered"]], on="po_id")
    dis = pd.read_parquet(data_dir / "disruptions.parquet")
    bridge = pd.read_parquet(data_dir / "disruption_suppliers.parquet")
    cal = pd.read_parquet(data_dir / "calendar.parquet")
    dates = pd.DatetimeIndex(cal.date)
    S, T = len(sup), len(dates)
    sidx = {s: i for i, s in enumerate(sup.supplier_id)}
    day = lambda col: (pd.to_datetime(col) - dates[0]).dt.days  # noqa: E731

    # Deliveries by arrival day
    dl = sh[sh.status == "DELIVERED"].copy()
    dl["s"], dl["t"] = dl.supplier_id.map(sidx), day(dl.actual_arrival)
    dl["lead_t"] = (pd.to_datetime(dl.actual_arrival) - pd.to_datetime(dl.order_date)).dt.days
    def grid(values):
        g = np.zeros((S, T))
        np.add.at(g, (dl.s.to_numpy(), dl.t.to_numpy()), values)
        return g
    n_del = grid(np.ones(len(dl)))
    delay_sum, late_sum = grid(dl.delay_days.to_numpy(float)), grid((dl.delay_days > 0).to_numpy(float))
    lt_sum, lt_sq = grid(dl.lead_t.to_numpy(float)), grid(dl.lead_t.to_numpy(float) ** 2)

    feats = {}
    for n in (28, 90):
        ds, c = _rolling(delay_sum, n_del, n)
        ls, _ = _rolling(late_sum, n_del, n)
        feats[f"delay_mean_{n}"] = np.where(c > 0, ds / np.maximum(c, 1), 0.0)
        feats[f"late_share_{n}"] = np.where(c > 0, ls / np.maximum(c, 1), 0.0)
    s1, c90 = _rolling(lt_sum, n_del, 90)
    s2, _ = _rolling(lt_sq, n_del, 90)
    mean_lt = s1 / np.maximum(c90, 1)
    feats["lead_time_cv_90"] = np.where(c90 > 1, np.sqrt(np.clip(s2 / np.maximum(c90, 1) - mean_lt**2, 0, None)) / np.maximum(mean_lt, 1), 0.0)
    feats["delay_trend"] = feats["delay_mean_28"] - feats["delay_mean_90"]

    # P90 of delays over 90 days (per supplier, computed on arrival-day lists)
    p90 = np.zeros((S, T))
    for s, g in dl.groupby("s"):
        t_arr, d_arr = g.t.to_numpy(), g.delay_days.to_numpy(float)
        order = np.argsort(t_arr)
        t_arr, d_arr = t_arr[order], d_arr[order]
        lo = np.searchsorted(t_arr, np.arange(T) - 89, side="left")
        hi = np.searchsorted(t_arr, np.arange(T), side="right")
        for t in range(T):
            if hi[t] > lo[t]:
                p90[s, t] = np.quantile(d_arr[lo[t] : hi[t]], 0.9)
    feats["delay_p90_90"] = p90

    # Open orders past promised date (known at t: promised < t and not yet arrived at t)
    overdue, max_over = np.zeros((S, T)), np.zeros((S, T))
    arr = pd.to_datetime(sh.actual_arrival).fillna(pd.Timestamp("2100-01-01"))
    sh_s, prom_t, arr_t = sh.supplier_id.map(sidx).to_numpy(), day(sh.promised_date).to_numpy(), day(arr).to_numpy()
    for s in range(S):
        m = sh_s == s
        p, a = prom_t[m], arr_t[m]
        for t in range(T):
            o = (p < t) & (a > t)
            if o.any():
                overdue[s, t] = o.sum()
                max_over[s, t] = (t - p[o]).max()
    feats["overdue_open"], feats["max_overdue_days"] = overdue, max_over

    # Fill ratio of orders whose ship date has passed, ordered in the last 28 days
    shipped_t = day(sh.ship_date).to_numpy()
    qs, qo = np.zeros((S, T)), np.zeros((S, T))
    np.add.at(qs, (sh_s, np.clip(shipped_t, 0, T - 1)), sh.quantity_units.to_numpy(float))
    np.add.at(qo, (sh_s, np.clip(shipped_t, 0, T - 1)), sh.qty_ordered.to_numpy(float))
    a28, b28 = _rolling(qs, qo, 28)
    feats["fill_ratio_28"] = np.where(b28 > 0, a28 / np.maximum(b28, 1), 1.0)

    # Country peers' recent delays (excluding self)
    country = sup.country.to_numpy()
    d14, c14 = _rolling(delay_sum, n_del, 14)
    cd = np.zeros((S, T))
    for c in np.unique(country):
        members = np.flatnonzero(country == c)
        tot_d, tot_c = d14[members].sum(0), c14[members].sum(0)
        for s in members:
            od, oc = tot_d - d14[s], tot_c - c14[s]
            cd[s] = np.where(oc > 0, od / np.maximum(oc, 1), 0.0)
    feats["country_delay_14"] = cd

    # Recorded disruptions (observable once detected)
    region = sup.region.to_numpy()
    own, c_ev, r_ev = np.zeros((S, T)), np.zeros((S, T)), np.zeros((S, T))
    since = np.full((S, T), 365.0)
    d_supp = bridge.merge(dis[["event_id", "detected_at", "resolved_at", "event_type"]], on="event_id")
    d_supp = d_supp[d_supp.event_type != "DEMAND"]
    for e in d_supp.itertuples():
        s = sidx[e.supplier_id]
        t0 = (pd.Timestamp(e.detected_at) - dates[0]).days
        t1 = (pd.Timestamp(e.resolved_at) - dates[0]).days if pd.notna(e.resolved_at) else T - 1
        own[s, t0 : t1 + 1] = 1
        for t in range(t1 + 1, min(T, t1 + 366)):
            since[s, t] = min(since[s, t], t - t1)
        peers_c = np.flatnonzero((country == country[s]) & (np.arange(S) != s))
        peers_r = np.flatnonzero((region == region[s]) & (np.arange(S) != s))
        c_ev[peers_c, t0 : t1 + 1] += 1
        r_ev[peers_r, t0 : t1 + 1] += 1
    since[own == 1] = 0
    feats.update(own_active_event=own, days_since_own_event=since, country_active_events=c_ev,
                 region_active_events=r_ev)

    static = {
        "country_risk_base": sup.country.map(lambda c: ref.COUNTRIES[c]["risk_base"]).to_numpy(float),
        "tier": sup.tier.to_numpy(float), "health_score": sup.health_score.to_numpy(float),
        "n_certifications": sup.certifications.str.count(",").to_numpy(float) + 1,
    }
    for k, v in static.items():
        feats[k] = np.repeat(v[:, None], T, axis=1)
    X = np.stack([feats[f] for f in FEATURES], axis=-1).astype(np.float32)
    return RiskPanel(sup, dates, X)
