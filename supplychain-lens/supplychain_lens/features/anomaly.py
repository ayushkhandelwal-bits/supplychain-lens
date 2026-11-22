"""Daily operational features for anomaly detection (observable data only).

One feature vector per SKU × warehouse × day; the autoencoder reads 7-day windows.
Every statistic uses only data up to and including that day (trailing windows).

  demand_z          demand vs trailing 28-day mean / std (spikes)
  is_promo          planned promotion (explains some spikes)
  is_holiday        Black Friday / Cyber Monday / holiday week (explains others)
  unexplained_inv   (Δ on hand − receipts + sales) / trailing mean demand (shrink, write-offs)
  receipt_delay     mean delay of shipments received that day, in weeks (supplier problems)
  stockout          1 if any demand was lost that day
  days_of_supply    log(1 + on hand / trailing mean demand)
  pipeline_cover    log(1 + on order / trailing mean demand)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from supplychain_lens.config import get_settings

FEATURES = ["demand_z", "is_promo", "is_holiday", "unexplained_inv", "receipt_delay", "stockout",
            "days_of_supply", "pipeline_cover"]
WINDOW = 7


@dataclass
class OpsPanel:
    keys: pd.DataFrame
    dates: pd.DatetimeIndex
    x: np.ndarray            # (S, T, F) raw features
    receipts: np.ndarray     # (S, T)
    primary_supplier: pd.Series


def _trailing(a: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std over the previous n days (excluding today)."""
    c = np.cumsum(np.pad(a, ((0, 0), (1, 0))), axis=1)
    c2 = np.cumsum(np.pad(a**2, ((0, 0), (1, 0))), axis=1)
    T = a.shape[1]
    lo = np.clip(np.arange(T) - n, 0, None)
    cnt = np.maximum(np.arange(T) - lo, 1)
    mean = (c[:, np.arange(T)] - c[:, lo]) / cnt
    var = (c2[:, np.arange(T)] - c2[:, lo]) / cnt - mean**2
    return mean, np.sqrt(np.clip(var, 0, None))


def build_ops_panel(data_dir: Path | None = None) -> OpsPanel:
    data_dir = data_dir or get_settings().data_dir / "nexus_v2"
    d = pd.read_parquet(data_dir / "demand_daily.parquet").sort_values(["sku_id", "warehouse_id", "date"])
    inv = pd.read_parquet(data_dir / "inventory_daily.parquet").sort_values(["sku_id", "warehouse_id", "date"])
    products = pd.read_parquet(data_dir / "products.parquet")
    sh = pd.read_parquet(data_dir / "shipments.parquet")
    keys = d[["sku_id", "warehouse_id"]].drop_duplicates().reset_index(drop=True)
    keys = keys.merge(products[["sku_id", "category", "primary_supplier_id"]], on="sku_id", how="left")
    dates = pd.DatetimeIndex(sorted(d.date.unique()))
    S, T = len(keys), len(dates)
    shape = lambda col, df: df[col].to_numpy(float).reshape(S, T)  # noqa: E731

    demand, sales, lost = shape("demand_units", d), shape("sales_units", d), shape("lost_sales_units", d)
    on_hand, on_order, receipts = shape("on_hand_units", inv), shape("on_order_units", inv), shape("receipts_units", inv)
    promo = shape("is_promo", d)
    holiday = np.maximum.reduce([shape(c, d) for c in ("is_black_friday", "is_cyber_monday", "is_holiday_week")])

    mean28, sd28 = _trailing(demand, 28)
    base = np.maximum(mean28, 1.0)
    demand_z = np.clip((demand - mean28) / np.maximum(sd28, 0.1 * base), -10, 10)
    prev_on_hand = np.concatenate([on_hand[:, :1], on_hand[:, :-1]], axis=1)
    unexplained = np.clip((on_hand - prev_on_hand - receipts + sales) / base, -20, 20)

    # mean delay (weeks) of shipments received that day, per series
    key_idx = {(r.sku_id, r.warehouse_id): i for i, r in enumerate(keys.itertuples())}
    delivered = sh[sh.status == "DELIVERED"].copy()
    delivered["s"] = [key_idx[(a, b)] for a, b in zip(delivered.sku_id, delivered.warehouse_id, strict=True)]
    delivered["t"] = (pd.to_datetime(delivered.actual_arrival) - dates[0]).dt.days
    agg = delivered.groupby(["s", "t"]).delay_days.mean()
    receipt_delay = np.zeros((S, T))
    receipt_delay[agg.index.get_level_values(0), agg.index.get_level_values(1)] = agg.to_numpy() / 7.0

    x = np.stack([demand_z, promo, holiday, unexplained, receipt_delay, (lost > 0).astype(float),
                  np.log1p(on_hand / base), np.log1p(on_order / base)], axis=-1).astype(np.float32)
    x[:, :28] = 0.0  # warm-up period without full trailing statistics
    return OpsPanel(keys, dates, x, receipts, keys.primary_supplier_id)


def windows(x: np.ndarray, pairs: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """(N, WINDOW * F) standardised windows ending on day t for (series, t) pairs."""
    s, t = pairs[:, 0], pairs[:, 1]
    idx = t[:, None] + np.arange(-WINDOW + 1, 1)[None]
    w = (x[s[:, None], idx] - mean) / std
    return w.reshape(len(pairs), -1).astype(np.float32)
