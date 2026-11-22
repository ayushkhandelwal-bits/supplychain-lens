"""Statistics that drive decisions.

Every function here exists because a decision uses it:

  demand_profile       CV and ADI classify each series (smooth / erratic / intermittent / lumpy,
                       Syntetos-Boylan cut-offs) -> how hard it is to forecast, how much buffer it needs
  seasonality_strength STL decomposition -> is weekly seasonality worth modelling for this series?
  lead_time_stats      mean, standard deviation and P90 of supplier lead time -> safety stock
  wilson_interval      95% interval for an on-time rate -> don't punish a supplier for 3 bad shipments
  safety_stock         z · sqrt(L·σ_d² + d²·σ_L²) -> buffer that covers demand AND lead-time variability
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats as st


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion. Better than ±1.96·SE for small n or rates near 0/1."""
    if n == 0:
        return (0.0, 1.0)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def classify_demand(adi: float, cv2: float) -> str:
    """Syntetos-Boylan-Croston categories (ADI 1.32, CV² 0.49)."""
    if adi < 1.32:
        return "smooth" if cv2 < 0.49 else "erratic"
    return "intermittent" if cv2 < 0.49 else "lumpy"


def seasonality_strength(y: np.ndarray, period: int = 7) -> float:
    """Hyndman's F_S = max(0, 1 − Var(remainder) / Var(seasonal + remainder)) from STL."""
    from statsmodels.tsa.seasonal import STL

    res = STL(np.log1p(y), period=period, robust=True).fit()
    denom = np.var(res.seasonal + res.resid)
    return float(max(0.0, 1 - np.var(res.resid) / denom)) if denom > 0 else 0.0


def demand_profile(demand: pd.DataFrame, end: str | pd.Timestamp, days: int = 365) -> pd.DataFrame:
    """Per SKU × warehouse statistics over the ``days`` before ``end`` (inclusive)."""
    end = pd.Timestamp(end)
    d = demand[(demand.date > end - pd.Timedelta(days=days)) & (demand.date <= end)]
    rows = []
    for (sku, wh), g in d.groupby(["sku_id", "warehouse_id"]):
        y = g.sort_values("date").demand_units.to_numpy(float)
        nz = np.flatnonzero(y > 0)
        adi = float(np.mean(np.diff(nz))) if len(nz) > 1 else float(len(y))
        mean, sd = y.mean(), y.std(ddof=1)
        cv = sd / mean if mean > 0 else np.nan
        nzv = y[y > 0]
        cv2 = float((nzv.std(ddof=1) / nzv.mean()) ** 2) if len(nzv) > 1 else np.nan
        t = np.arange(len(y))
        slope = np.polyfit(t, np.log1p(y), 1)[0] * 365 if len(y) > 30 else np.nan
        rows.append({"sku_id": sku, "warehouse_id": wh, "mean": mean, "std": sd, "cv": cv,
                     "p50": float(np.median(y)), "p95": float(np.quantile(y, 0.95)),
                     "zero_share": float((y == 0).mean()), "adi": adi, "cv2": cv2,
                     "pattern": classify_demand(adi, cv2) if not np.isnan(cv2) else "insufficient",
                     "trend_pct_per_year": float(np.expm1(slope) * 100) if not np.isnan(slope) else np.nan,
                     "weekly_seasonality_strength": seasonality_strength(y) if len(y) >= 28 else np.nan})
    return pd.DataFrame(rows)


def lead_time_stats(shipments: pd.DataFrame, purchase_orders: pd.DataFrame, end: str | pd.Timestamp | None = None,
                    days: int | None = None) -> pd.DataFrame:
    """Per-supplier lead-time and on-time statistics from delivered shipments."""
    sh = shipments[shipments.status == "DELIVERED"].merge(purchase_orders[["po_id", "order_date"]], on="po_id")
    sh["actual_arrival"] = pd.to_datetime(sh.actual_arrival)
    if end is not None:
        end = pd.Timestamp(end)
        sh = sh[sh.actual_arrival <= end]
        if days:
            sh = sh[sh.actual_arrival > end - pd.Timedelta(days=days)]
    sh["lead_time"] = (sh.actual_arrival - pd.to_datetime(sh.order_date)).dt.days
    rows = []
    for sup, g in sh.groupby("supplier_id"):
        on_time = int((g.delay_days <= 0).sum())
        lo, hi = wilson_interval(on_time, len(g))
        rows.append({"supplier_id": sup, "deliveries": len(g), "lead_time_mean": g.lead_time.mean(),
                     "lead_time_std": g.lead_time.std(ddof=1), "lead_time_p90": g.lead_time.quantile(0.9),
                     "delay_mean": g.delay_days.mean(), "delay_p90": g.delay_days.quantile(0.9),
                     "on_time_rate": on_time / len(g), "on_time_ci_low": lo, "on_time_ci_high": hi})
    return pd.DataFrame(rows)


def safety_stock(z: float, mean_demand: float, std_demand: float, lead_time: float, std_lead_time: float) -> float:
    """Units of buffer covering both demand variability and lead-time variability."""
    return float(z * math.sqrt(lead_time * std_demand**2 + mean_demand**2 * std_lead_time**2))


def service_z(service_level: float) -> float:
    return float(st.norm.ppf(service_level))
