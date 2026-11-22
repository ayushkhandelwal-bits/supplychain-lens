# Calendar/holiday feature ideas adapted from NEXUS-SCCT tft_model.py; reimplemented.
"""Leakage-safe inputs for demand forecasting, shared by every forecasting model.

The panel is a set of dense arrays (series × day). A forecast made at *origin* day ``o``
may use observed values up to ``o - 1`` and *known-future* inputs (calendar, planned
promotions and planned prices) for ``o … o + H - 1``. Nothing else.

Scaling: each series is divided by its mean demand over the TRAIN period only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from supplychain_lens.config import Settings, get_settings

PAST_FEATURES = ["demand_scaled", "is_promo", "price_ratio", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
                 "is_black_friday", "is_cyber_monday", "is_holiday_week"]
FUTURE_FEATURES = PAST_FEATURES[1:]  # everything except the target itself is known in advance
STATIC_CATEGORICALS = ["sku", "warehouse", "category"]


@dataclass
class Panel:
    keys: pd.DataFrame          # one row per series: sku_id, warehouse_id, category, codes
    dates: pd.DatetimeIndex
    y: np.ndarray               # (S, T) demand units
    promo: np.ndarray           # (S, T) 0/1
    price_ratio: np.ndarray     # (S, T) price / list price
    calendar: np.ndarray        # (T, 7) dow sin/cos, doy sin/cos, BF, CM, holiday week
    scale: np.ndarray           # (S,) train-period mean demand
    settings: Settings

    @property
    def n_series(self) -> int:
        return self.y.shape[0]

    def idx(self, day) -> int:
        return int((pd.Timestamp(day) - self.dates[0]).days)

    @property
    def train_end_idx(self) -> int:
        return self.idx(self.settings.train_end)

    @property
    def val_end_idx(self) -> int:
        return self.idx(self.settings.val_end)

    def features(self) -> np.ndarray:
        """(S, T, F) array in PAST_FEATURES order. The first channel is the scaled target."""
        S, T = self.y.shape
        cal = np.broadcast_to(self.calendar[None], (S, T, self.calendar.shape[1]))
        return np.concatenate([(self.y / self.scale[:, None])[..., None], self.promo[..., None],
                               self.price_ratio[..., None], cal], axis=-1).astype(np.float32)


def load_panel(data_dir: Path | None = None, settings: Settings | None = None) -> Panel:
    s = settings or get_settings()
    data_dir = data_dir or s.data_dir / "nexus_v2"
    d = pd.read_parquet(data_dir / "demand_daily.parquet",
                        columns=["date", "sku_id", "warehouse_id", "demand_units", "price", "is_promo"])
    products = pd.read_parquet(data_dir / "products.parquet")
    cal = pd.read_parquet(data_dir / "calendar.parquet")
    keys = d[["sku_id", "warehouse_id"]].drop_duplicates().sort_values(["sku_id", "warehouse_id"]).reset_index(drop=True)
    keys = keys.merge(products[["sku_id", "category", "list_price", "unit_cost"]], on="sku_id", how="left")
    for col, code in (("sku_id", "sku"), ("warehouse_id", "warehouse"), ("category", "category")):
        keys[f"{code}_code"] = keys[col].astype("category").cat.codes.astype(np.int64)

    dates = pd.DatetimeIndex(sorted(d.date.unique()))
    d = d.sort_values(["sku_id", "warehouse_id", "date"])
    S, T = len(keys), len(dates)
    y = d.demand_units.to_numpy(dtype=np.float64).reshape(S, T)
    promo = d.is_promo.to_numpy(dtype=np.float64).reshape(S, T)
    price = d.price.to_numpy(dtype=np.float64).reshape(S, T)
    price_ratio = price / keys.list_price.to_numpy()[:, None]

    cal = cal.set_index("date").loc[dates]
    dow, doy = cal.dow.to_numpy(), cal.day_of_year.to_numpy()
    calendar = np.stack([np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
                         np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
                         cal.is_black_friday.to_numpy(float), cal.is_cyber_monday.to_numpy(float),
                         cal.is_holiday_week.to_numpy(float)], axis=1)
    train_end = int((pd.Timestamp(s.train_end) - dates[0]).days)
    scale = np.maximum(y[:, : train_end + 1].mean(axis=1), 1.0)
    return Panel(keys, dates, y, promo, price_ratio, calendar, scale, s)


def evaluation_origins(panel: Panel, split: str) -> list[int]:
    """Forecast origins (day indices) for 'val' or 'test', each followed by a full horizon.

    Validation: 1 Jul 2025 + every 28 days while the horizon stays within validation.
    Test:       1 Oct 2025 + every 28 days while the horizon stays within the calendar
                (3 origins covering 1 Oct – 23 Dec, including Black Friday).
    """
    H = panel.settings.horizon_days
    if split == "val":
        start, last = panel.train_end_idx + 1, panel.val_end_idx
    elif split == "test":
        start, last = panel.val_end_idx + 1, len(panel.dates) - 1
    else:
        raise ValueError(split)
    return [o for o in range(start, last + 1, H) if o + H - 1 <= last]


def training_origins(panel: Panel, stride: int = 1) -> list[int]:
    """Origins whose whole horizon ends on or before the last training day."""
    L, H = panel.settings.lookback_days, panel.settings.horizon_days
    return list(range(L, panel.train_end_idx - H + 2, stride))


def window_arrays(panel: Panel, feats: np.ndarray, pairs: np.ndarray):
    """Build model inputs for (series, origin) pairs.

    Returns past (N, L, F), future (N, H, F-1), static codes (N, 3), log-scale (N, 1), target (N, H).
    """
    L, H = panel.settings.lookback_days, panel.settings.horizon_days
    s, o = pairs[:, 0], pairs[:, 1]
    t_past = o[:, None] + np.arange(-L, 0)[None]
    t_fut = o[:, None] + np.arange(H)[None]
    past = feats[s[:, None], t_past]
    fut = feats[s[:, None], t_fut][..., 1:]
    target = feats[s[:, None], t_fut][..., 0]
    codes = panel.keys[["sku_code", "warehouse_code", "category_code"]].to_numpy()[s]
    log_scale = np.log1p(panel.scale[s])[:, None].astype(np.float32)
    return past, fut, codes, log_scale, target
