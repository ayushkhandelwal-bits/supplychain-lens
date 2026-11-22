"""Inference-time demand forecasting (no PyTorch needed).

Loads the model selected by the forecast comparison report and produces P10/P50/P90 for any
origin day — including origins AFTER the snapshot date, where known-future inputs are
calendar features only (no promotions are planned in the data beyond the snapshot, so none
are assumed; prices are assumed at list).

Neural models run through ONNX Runtime; LightGBM through its own booster files.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from supplychain_lens.data.generator import calendar_frame
from supplychain_lens.features.forecasting import Panel, load_panel, window_arrays
from supplychain_lens.serving import registry


def extended_features(panel: Panel, extra_days: int) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Feature array with ``extra_days`` of known-future rows appended after the last date.

    Demand in the appended rows is unknown and set to NaN; nothing reads it as an input.
    """
    feats = panel.features()
    if extra_days <= 0:
        return feats, panel.dates
    start = panel.dates[-1] + pd.Timedelta(days=1)
    cal = calendar_frame(start.date(), (start + pd.Timedelta(days=extra_days - 1)).date())
    dow, doy = cal.dow.to_numpy(), cal.day_of_year.to_numpy()
    calf = np.stack([np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7), np.sin(2 * np.pi * doy / 365.25),
                     np.cos(2 * np.pi * doy / 365.25), cal.is_black_friday.to_numpy(float),
                     cal.is_cyber_monday.to_numpy(float), cal.is_holiday_week.to_numpy(float)], axis=1)
    S = panel.n_series
    future = np.concatenate([np.full((S, extra_days, 1), np.nan), np.zeros((S, extra_days, 1)),  # demand, promo
                             np.ones((S, extra_days, 1)),                                        # price ratio
                             np.broadcast_to(calf[None], (S, extra_days, calf.shape[1]))], axis=-1)
    dates = panel.dates.append(pd.DatetimeIndex(cal.date))
    return np.concatenate([feats, future.astype(np.float32)], axis=1), dates


@dataclass
class Forecaster:
    model_name: str
    artifact: str
    version: str
    panel: Panel
    _session: object = field(default=None, repr=False)
    _boosters: dict = field(default=None, repr=False)
    last_latency_ms: float = 0.0
    batch_size: int = 1000

    @classmethod
    def load(cls, model_artifact: str | None = None, panel: Panel | None = None) -> Forecaster:
        """Load the selected model (from the comparison report) or a specific artifact."""
        panel = panel or load_panel()
        if model_artifact is None:
            cmp_run = registry.load_run("forecast_comparison")
            if cmp_run is None:
                raise FileNotFoundError("No forecast comparison found; run the training pipeline first.")
            cmp = cmp_run.json("comparison.json")
            model_artifact, name = cmp["selected_artifact"], cmp["selected_model"]
        else:
            name = {"forecast_lstm": "LSTM (PyTorch)", "forecast_tft": "TFT (PyTorch)",
                    "forecast_baselines": "LightGBM (quantile)"}[model_artifact]
        run = registry.load_run(model_artifact)
        if run is None:
            raise FileNotFoundError(f"artifact {model_artifact} not found")
        f = cls(name, model_artifact, run.version, panel)
        if model_artifact == "forecast_baselines":
            import lightgbm as lgb

            f._boosters = {q: lgb.Booster(model_file=str(run.file(f"lightgbm_q{int(q * 100):02d}.txt")))
                           for q in (0.1, 0.5, 0.9)}
        else:
            import onnxruntime as ort

            f._session = ort.InferenceSession(str(run.file("model.onnx")), providers=["CPUExecutionProvider"])
        return f

    def predict(self, origins: list[int], series: np.ndarray | None = None) -> np.ndarray:
        """Quantiles in demand units, shape (len(origins), n_series, H, 3)."""
        H = self.panel.settings.horizon_days
        series = np.arange(self.panel.n_series) if series is None else np.asarray(series)
        extra = max(0, max(origins) + H - len(self.panel.dates))
        t0 = time.perf_counter()
        if self._session is not None:
            feats, _ = extended_features(self.panel, extra)
            pairs = np.array([(s, o) for o in origins for s in series])
            chunks = []
            for i in range(0, len(pairs), self.batch_size):  # bounded memory (TFT attention is heavy)
                past, fut, codes, ls, _ = window_arrays(self.panel, feats, pairs[i : i + self.batch_size])
                chunks.append(self._session.run(None, {"past": past, "future": np.ascontiguousarray(fut),
                                                       "codes": codes.astype(np.int64), "log_scale": ls})[0])
            q = np.concatenate(chunks) * self.panel.scale[pairs[:, 0], None, None]
            out = q.reshape(len(origins), len(series), H, 3)
        else:
            out = self._predict_lightgbm(origins, series, extra)
        self.last_latency_ms = (time.perf_counter() - t0) * 1000
        return np.clip(out, 0, None)

    def _predict_lightgbm(self, origins, series, extra) -> np.ndarray:
        from supplychain_lens.models.baselines import lgb_design

        if extra > 0:  # extend the panel's arrays with known-future rows
            feats, dates = extended_features(self.panel, extra)
            p = self.panel
            ext = Panel(p.keys, dates, np.concatenate([p.y, np.full((p.n_series, extra), np.nan)], 1),
                        np.concatenate([p.promo, np.zeros((p.n_series, extra))], 1),
                        np.concatenate([p.price_ratio, np.ones((p.n_series, extra))], 1),
                        np.concatenate([p.calendar, feats[0, -extra:, 3:]], 0), p.scale, p.settings)
        else:
            ext = self.panel
        X, _, idx = lgb_design(ext, origins)
        H, S = self.panel.settings.horizon_days, self.panel.n_series
        qs = [self._boosters[q].predict(X).clip(0) * self.panel.scale[idx[:, 0]] for q in (0.1, 0.5, 0.9)]
        arr = np.sort(np.stack(qs, -1), axis=-1).reshape(len(origins), H, S, 3).transpose(0, 2, 1, 3)
        return arr[:, series]

    def forecast_frame(self, origin: int, series: np.ndarray | None = None) -> pd.DataFrame:
        series = np.arange(self.panel.n_series) if series is None else np.asarray(series)
        q = self.predict([origin], series)[0]
        H = q.shape[1]
        start = self.panel.dates[0] + pd.Timedelta(days=origin)
        dates = pd.date_range(start, periods=H)
        k = self.panel.keys.iloc[series]
        return pd.DataFrame({"sku_id": np.repeat(k.sku_id.to_numpy(), H), "warehouse_id": np.repeat(k.warehouse_id.to_numpy(), H),
                             "date": np.tile(dates, len(series)), "h": np.tile(np.arange(1, H + 1), len(series)),
                             "p10": q[..., 0].reshape(-1), "p50": q[..., 1].reshape(-1), "p90": q[..., 2].reshape(-1)})
