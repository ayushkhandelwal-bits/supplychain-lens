# Idea adapted from NEXUS-SCCT tft_model.py (DriftDetector: Mann-Whitney U per SKU, retrain if
# > 10% of SKUs drift). Extended here with false-discovery-rate control and PSI.
"""Model monitoring: has the forecast error distribution shifted?

NEXUS tested raw demand per SKU with Mann-Whitney U at p < 0.05. With 400 series that
gives ~20 false alarms by chance, and raw demand is seasonal, so peak season always
"drifts". We instead test the selected model's **forecast residuals** (actual − P50,
scaled per series), recent window vs validation period, and:

  • control the false discovery rate with Benjamini-Hochberg (q = 0.05)
  • add the Population Stability Index (PSI) as an effect size (> 0.25 = major shift)
  • keep NEXUS's system-level rule: retraining is recommended if > 10% of series drift
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


def benjamini_hochberg(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Boolean mask of discoveries at false discovery rate q."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    thresh = q * np.arange(1, n + 1) / n
    passed = p[order] <= thresh
    k = np.max(np.flatnonzero(passed)) + 1 if passed.any() else 0
    mask = np.zeros(n, bool)
    mask[order[:k]] = True
    return mask


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index using reference deciles."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, edges)[0] / len(reference)
    c = np.histogram(current, edges)[0] / len(current)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def residual_drift(predictions: pd.DataFrame, scale: np.ndarray, recent_days: int = 28,
                   q: float = 0.05, system_threshold: float = 0.10) -> dict:
    """predictions: one model's rows with split, series, t_idx, y, p50 (val and test)."""
    df = predictions.assign(res=(predictions.y - predictions.p50) / scale[predictions.series.to_numpy()])
    ref = df[df.split == "val"]
    cur = df[df.split == "test"]
    cur = cur[cur.t_idx > cur.t_idx.max() - recent_days]
    rows = []
    for s, g in cur.groupby("series"):
        r = ref[ref.series == s].res.to_numpy()
        c = g.res.to_numpy()
        p = mannwhitneyu(r, c, alternative="two-sided").pvalue
        rows.append({"series": s, "p_value": p, "psi": psi(r, c), "median_shift": float(np.median(c) - np.median(r))})
    out = pd.DataFrame(rows)
    out["raw_flag"] = out.p_value < 0.05
    out["drift"] = benjamini_hochberg(out.p_value.to_numpy(), q)
    share = float(out.drift.mean())
    return {"table": out, "share_drifting": share, "share_raw_p05": float(out.raw_flag.mean()),
            "major_psi_share": float((out.psi > 0.25).mean()), "retrain_recommended": share > system_threshold,
            "rule": f"retrain if > {system_threshold:.0%} of series drift after Benjamini-Hochberg (q = {q})"}
