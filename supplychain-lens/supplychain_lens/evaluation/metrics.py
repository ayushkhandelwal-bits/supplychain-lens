"""Forecast, classification and detection metrics — each with a business meaning.

Forecasting (computed on demand units, after un-scaling):
  WAPE      Σ|y − ŷ| / Σy. Error as a share of volume; robust to zero-demand days. Headline metric.
  MAE/RMSE  Units; RMSE punishes big misses (stockout-sized errors) more.
  Bias      Σ(ŷ − y) / Σy. Positive = over-forecast (excess stock), negative = under (stockouts).
  Pinball   Quantile loss for P10/P50/P90 — rewards well-placed uncertainty bands.
  Coverage  Share of actuals inside [P10, P90]; ~80% if the band is calibrated.
MAPE is not used as a headline: it explodes on low-volume days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def wape(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.abs(y - yhat).sum() / max(np.abs(y).sum(), 1e-9))


def bias(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float((yhat - y).sum() / max(np.abs(y).sum(), 1e-9))


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(yhat, float))))


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(yhat, float)) ** 2)))


def pinball(y, q_pred, q: float) -> float:
    d = np.asarray(y, float) - np.asarray(q_pred, float)
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def coverage(y, lo, hi) -> float:
    y = np.asarray(y, float)
    return float(np.mean((y >= np.asarray(lo, float)) & (y <= np.asarray(hi, float))))


def forecast_summary(df: pd.DataFrame) -> dict:
    """Metrics for a predictions frame with columns y, p50 and optionally p10/p90."""
    out = {"wape": wape(df.y, df.p50), "mae": mae(df.y, df.p50), "rmse": rmse(df.y, df.p50),
           "bias": bias(df.y, df.p50), "n": int(len(df))}
    if {"p10", "p90"} <= set(df.columns) and df.p10.notna().all():
        scale = max(float(df.y.abs().mean()), 1e-9)  # pinball scaled by mean demand -> comparable across models
        out["pinball_scaled"] = float(np.mean([pinball(df.y, df[c], q) for c, q in
                                               (("p10", 0.1), ("p50", 0.5), ("p90", 0.9))]) / scale)
        out["coverage_p10_p90"] = coverage(df.y, df.p10, df.p90)
    return out


def by_group(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = [{col: k, **forecast_summary(g)} for k, g in df.groupby(col, observed=True)]
    return pd.DataFrame(rows)


def horizon_bucket(h: pd.Series) -> pd.Series:
    return pd.cut(h, [0, 7, 14, 21, 28], labels=["days 1–7", "days 8–14", "days 15–21", "days 22–28"])


# ── Classification ──────────────────────────────────────────────────────────
def classification_summary(y, p, threshold: float) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    y, p = np.asarray(y, int), np.asarray(p, float)
    pred = p >= threshold
    tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)), "threshold": float(threshold),
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "positive_rate": float(y.mean()), "alerts": int(pred.sum()), "n": int(len(y)),
    }


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y, int), "p": np.asarray(p, float)})
    df["bin"] = pd.qcut(df.p.rank(method="first"), bins, labels=False)
    return df.groupby("bin").agg(mean_predicted=("p", "mean"), observed_rate=("y", "mean"), n=("y", "size")).reset_index()


def threshold_for_alert_budget(p, budget_share: float) -> float:
    """Threshold that flags the top ``budget_share`` of rows (an analyst-capacity rule)."""
    return float(np.quantile(np.asarray(p, float), 1 - budget_share))
