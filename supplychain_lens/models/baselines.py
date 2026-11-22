"""Forecasting baselines every neural model must beat.

1. Seasonal naive  — repeat the last observed week. Zero parameters; the honest floor.
2. Holt-Winters    — exponential smoothing with weekly seasonality (statsmodels), fitted per
                     series on log1p(demand) over the year before each origin. Ignores promotions.
3. LightGBM        — one global gradient-boosted model, direct multi-horizon, with lags, rolling
                     statistics, known-future promotions/prices/calendar, and quantile objectives
                     for P10/P50/P90. A strong non-deep benchmark.

All functions return a long predictions frame with columns
    model, series, origin_idx, h (1..H), t_idx, y, p10, p50, p90
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from supplychain_lens.features.forecasting import Panel


def _frame(model: str, panel: Panel, origins: list[int], p50: np.ndarray, p10=None, p90=None) -> pd.DataFrame:
    """p50 shape (n_origins, S, H)."""
    H = panel.settings.horizon_days
    n_o, S = len(origins), panel.n_series
    o = np.repeat(np.array(origins), S * H)
    s = np.tile(np.repeat(np.arange(S), H), n_o)
    h = np.tile(np.arange(1, H + 1), n_o * S)
    t = o + h - 1
    df = pd.DataFrame({"model": model, "series": s, "origin_idx": o, "h": h, "t_idx": t, "y": panel.y[s, t],
                       "p50": p50.reshape(-1)})
    df["p10"] = p10.reshape(-1) if p10 is not None else np.nan
    df["p90"] = p90.reshape(-1) if p90 is not None else np.nan
    return df


# ── 1. Seasonal naive ────────────────────────────────────────────────────────
def seasonal_naive(panel: Panel, origins: list[int]) -> pd.DataFrame:
    H = panel.settings.horizon_days
    preds = np.stack([panel.y[:, o - 7 + (np.arange(H) % 7)] for o in origins])
    return _frame("Seasonal naive", panel, origins, preds)


# ── 2. Holt-Winters ─────────────────────────────────────────────────────────
def holt_winters(panel: Panel, origins: list[int], history_days: int = 365) -> pd.DataFrame:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    H = panel.settings.horizon_days
    preds = np.zeros((len(origins), panel.n_series, H))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k, o in enumerate(origins):
            for s in range(panel.n_series):
                hist = np.log1p(panel.y[s, max(0, o - history_days) : o])
                try:
                    fit = ExponentialSmoothing(hist, trend="add", damped_trend=True, seasonal="add",
                                               seasonal_periods=7, initialization_method="estimated").fit()
                    preds[k, s] = np.expm1(fit.forecast(H)).clip(0)
                except Exception:  # extremely rare numerical failure -> fall back to seasonal naive
                    preds[k, s] = panel.y[s, o - 7 + (np.arange(H) % 7)]
    return _frame("Holt-Winters (ETS)", panel, origins, preds)


# ── 3. LightGBM direct multi-horizon ────────────────────────────────────────
LGB_FEATURES = (
    [f"lag_{k}" for k in range(1, 8)]
    + ["mean_7", "mean_28", "mean_56", "std_28", "same_wd_last", "same_wd_mean4", "promo_share_28",
       "h", "promo_t", "price_ratio_t", "dow_sin_t", "dow_cos_t", "doy_sin_t", "doy_cos_t", "bf_t", "cm_t",
       "holiday_t", "sku_code", "warehouse_code", "category_code", "log_scale"]
)


def lgb_design(panel: Panel, origins: list[int]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """One row per (series, origin, h). Returns X, target (scaled), and (series, origin, h) index."""
    H = panel.settings.horizon_days
    ys = panel.y / panel.scale[:, None]
    S = panel.n_series
    blocks, targets, index = [], [], []
    codes = panel.keys[["sku_code", "warehouse_code", "category_code"]].to_numpy()
    log_scale = np.log1p(panel.scale)
    for o in origins:
        lags = ys[:, o - 7 : o][:, ::-1]                      # lag_1..lag_7
        m7, m28, m56 = ys[:, o - 7 : o].mean(1), ys[:, o - 28 : o].mean(1), ys[:, o - 56 : o].mean(1)
        sd28 = ys[:, o - 28 : o].std(1)
        promo28 = panel.promo[:, o - 28 : o].mean(1)
        for h in range(1, H + 1):
            t = o + h - 1
            k = (h - 1) // 7 + 1                              # most recent observed same weekday
            same_last = ys[:, t - 7 * k]
            same_mean4 = np.mean([ys[:, t - 7 * (k + j)] for j in range(4)], axis=0)
            cal = panel.calendar[t]
            blocks.append(np.column_stack([
                lags, m7, m28, m56, sd28, same_last, same_mean4, promo28, np.full(S, h),
                panel.promo[:, t], panel.price_ratio[:, t], np.tile(cal, (S, 1)), codes, log_scale,
            ]))
            targets.append(ys[:, t])
            index.append(np.column_stack([np.arange(S), np.full(S, o), np.full(S, h)]))
    X = pd.DataFrame(np.vstack(blocks).astype(np.float32), columns=LGB_FEATURES)
    for c in ("sku_code", "warehouse_code", "category_code"):
        X[c] = X[c].astype(int).astype("category")
    return X, np.concatenate(targets).astype(np.float32), np.vstack(index)


def train_lightgbm(panel: Panel, train_origins: list[int], val_origins: list[int], seed: int = 42,
                   quantiles=(0.1, 0.5, 0.9), max_rounds: int = 800) -> dict:
    import lightgbm as lgb

    Xtr, ytr, _ = lgb_design(panel, train_origins)
    Xva, yva, _ = lgb_design(panel, val_origins)
    models, info = {}, {"train_rows": len(Xtr), "val_rows": len(Xva), "rounds": {}, "seconds": {}}
    for q in quantiles:
        t0 = time.perf_counter()
        params = {"objective": "quantile", "alpha": q, "learning_rate": 0.05, "num_leaves": 63,
                  "min_data_in_leaf": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
                  "seed": seed, "verbose": -1, "num_threads": 0}
        dtr = lgb.Dataset(Xtr, ytr)
        dva = lgb.Dataset(Xva, yva, reference=dtr)
        m = lgb.train(params, dtr, max_rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        models[q] = m
        info["rounds"][str(q)] = m.best_iteration
        info["seconds"][str(q)] = round(time.perf_counter() - t0, 1)
    return {"models": models, "info": info}


def predict_lightgbm(panel: Panel, models: dict, origins: list[int]) -> pd.DataFrame:
    X, _, idx = lgb_design(panel, origins)
    H, S = panel.settings.horizon_days, panel.n_series
    out = {}
    for q, m in models.items():
        pred = m.predict(X, num_iteration=m.best_iteration).clip(0) * panel.scale[idx[:, 0]]
        # rows are ordered (origin, h, series) -> reshape to (origin, series, h)
        out[q] = pred.reshape(len(origins), H, S).transpose(0, 2, 1)
    p10, p50, p90 = np.sort(np.stack([out[0.1], out[0.5], out[0.9]]), axis=0)  # fix rare quantile crossing
    return _frame("LightGBM (quantile)", panel, origins, p50, p10, p90)
