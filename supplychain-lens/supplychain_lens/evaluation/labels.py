"""Evaluation labels built from hidden truth.

This is the ONLY module outside the generator allowed to read ``data/truth`` (enforced by
tests/test_data.py). Labels are used to train the supervised risk model and to score the
anomaly detector — never as model inputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from supplychain_lens.config import get_settings


def truth_dir() -> Path:
    return get_settings().data_dir / "truth"


SUPPLIER_ORIGINATED = ("SUPPLIER",)


def disruption_labels(suppliers: list[str], dates: pd.DatetimeIndex, horizon_days: int = 14,
                      event_types: tuple[str, ...] = SUPPLIER_ORIGINATED) -> np.ndarray:
    """(n_suppliers, n_days) array: 1 if a supplier-originated disruption STARTS in (t, t + horizon].

    Only SUPPLIER-type events (factory shutdown, quality recall, capacity reduction, financial
    distress) are targets: they arise from the supplier's own condition, so supplier data can
    anticipate them. Weather, geopolitical, port and logistics shocks are exogenous; they are
    handled by the recorded-events feed and the scenario simulator, not by this classifier.
    """
    ev = pd.read_parquet(truth_dir() / "all_disruption_events.parquet")
    ev = ev[ev.event_type.isin(event_types)]
    idx = {s: i for i, s in enumerate(suppliers)}
    y = np.zeros((len(suppliers), len(dates)), dtype=np.int8)
    for e in ev.itertuples():
        if e.supplier_id not in idx:
            continue
        t = (pd.Timestamp(e.start_date) - dates[0]).days
        y[idx[e.supplier_id], max(0, t - horizon_days) : max(0, t)] = 1
    return y


def injected_anomalies() -> pd.DataFrame:
    a = pd.read_parquet(truth_dir() / "injected_anomalies.parquet")
    a["start_date"] = pd.to_datetime(a.start_date)
    a["end_date"] = pd.to_datetime(a.end_date)
    return a


def anomaly_day_labels(keys: pd.DataFrame, dates: pd.DatetimeIndex, primary_supplier: pd.Series,
                       receipts: np.ndarray, burst_tail_days: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """Series × day anomaly labels and the matching anomaly row index (-1 = none).

    demand_spike / inventory_shrink: the injected days of that series.
    supplier_delay_burst: days with a receipt, from the burst start until ``burst_tail_days`` after
    it ends, for every series whose primary supplier is affected (late shipments land then).
    """
    a = injected_anomalies()
    S, T = len(keys), len(dates)
    lab = np.zeros((S, T), dtype=np.int8)
    which = np.full((S, T), -1, dtype=np.int32)
    key_idx = {f"{r.sku_id}@{r.warehouse_id}": i for i, r in enumerate(keys.itertuples())}
    sup_series = primary_supplier.reset_index(drop=True)
    for k, r in enumerate(a.itertuples()):
        t0, t1 = (r.start_date - dates[0]).days, (r.end_date - dates[0]).days
        if r.entity_type == "series":
            s = key_idx[r.entity_id]
            lab[s, t0 : t1 + 1], which[s, t0 : t1 + 1] = 1, k
        else:
            for s in np.flatnonzero(sup_series.to_numpy() == r.entity_id):
                span = np.arange(t0, min(T, t1 + burst_tail_days + 1))
                hit = span[receipts[s, span] > 0]
                lab[s, hit], which[s, hit] = 1, k
    return lab, which


def recorded_event_days(keys: pd.DataFrame, dates: pd.DatetimeIndex, primary_supplier: pd.Series,
                        data_dir: Path | None = None) -> np.ndarray:
    """Series × day mask of RECORDED events (disruptions of the primary supplier, category demand events).

    Alerts during these are explainable by information the business already has; we report
    precision both with and without counting them as true signals.
    """
    data_dir = data_dir or get_settings().data_dir / "nexus_v2"
    dis = pd.read_parquet(data_dir / "disruptions.parquet")
    bridge = pd.read_parquet(data_dir / "disruption_suppliers.parquet")
    S, T = len(keys), len(dates)
    mask = np.zeros((S, T), dtype=bool)
    end_default = dates[-1]
    for e in dis.itertuples():
        t0 = (pd.Timestamp(e.detected_at) - dates[0]).days
        t1 = (pd.Timestamp(e.resolved_at if pd.notna(e.resolved_at) else end_default) - dates[0]).days + 30
        if e.event_type == "DEMAND":
            series = np.flatnonzero(keys.category.to_numpy() == e.affected_category)
        else:
            sups = set(bridge.loc[bridge.event_id == e.event_id, "supplier_id"])
            series = np.flatnonzero(primary_supplier.reset_index(drop=True).isin(sups).to_numpy())
        mask[np.ix_(series, np.arange(max(0, t0), min(T, t1 + 1)))] = True
    return mask
