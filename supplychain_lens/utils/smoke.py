"""Hardware smoke test: a tiny, real deep-learning round trip on your machine.

What it proves in about 30 seconds:
  1. PyTorch is installed and can train (loss goes down on a learnable task).
  2. Your CPU's training speed, compared with the reference machine used for sizing.
  3. Export to ONNX works and ONNX Runtime gives the same answers as PyTorch
     at different batch sizes (the serving path from ADR-0009).

The model has the same shape as the planned demand model A (seq2seq LSTM with
quantile outputs and a series embedding) so the timing transfers; the data is a
synthetic seasonal series, NOT project data.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Measured on the reference machine (2 vCPU cloud workspace, 7 GB RAM, no GPU,
# PyTorch 2.14.1) with SmokeConfig() defaults: seconds per 20,000 training windows.
REFERENCE_SEC_PER_20K = 5.3
REFERENCE_MACHINE = "2 vCPU cloud workspace, 7 GB RAM, PyTorch 2.14.1"
# Phase 0b feasibility benchmark at full planned scale on the same machine:
# epoch time at stride 7 for the LSTM prototype and for the compact-TFT prototype.
REFERENCE_LSTM_EPOCH_SEC = 20.0
REFERENCE_TFT_EPOCH_SEC = 68.0


@dataclass(frozen=True)
class SmokeConfig:
    n_series: int = 64
    n_days: int = 400
    lookback: int = 56
    horizon: int = 28
    hidden: int = 64
    batch_size: int = 256
    train_windows: int = 20_000
    seed: int = 42


def make_windows(cfg: SmokeConfig, rng: np.random.Generator):
    """Build (past, future, series_id, target) windows from synthetic seasonal series.

    Past features: [scaled demand, day-of-week sin, day-of-week cos].
    Future (known) features: [day-of-week sin, day-of-week cos].
    """
    t = np.arange(cfg.n_days)
    level = rng.uniform(50, 500, (cfg.n_series, 1))
    weekly = rng.uniform(0.1, 0.4, (cfg.n_series, 1))
    y = level * (1 + weekly * np.sin(2 * np.pi * t / 7)) * rng.lognormal(0, 0.1, (cfg.n_series, cfg.n_days))
    y = y / level  # per-series scaling (stand-in for "fit scaler on train only")

    dow = np.stack([np.sin(2 * np.pi * t / 7), np.cos(2 * np.pi * t / 7)], -1)  # (T, 2)
    span = cfg.lookback + cfg.horizon
    starts = np.arange(cfg.n_days - span + 1)

    past_y = np.lib.stride_tricks.sliding_window_view(y, span, axis=1)[:, starts]  # (S, W, span)
    s_idx = np.repeat(np.arange(cfg.n_series), len(starts))
    windows = past_y.reshape(-1, span)

    dow_win = np.lib.stride_tricks.sliding_window_view(dow, span, axis=0)[starts]  # (W, 2, span)
    dow_win = np.transpose(dow_win, (0, 2, 1))[np.tile(np.arange(len(starts)), cfg.n_series)]  # (N, span, 2)

    past = np.concatenate([windows[:, : cfg.lookback, None], dow_win[:, : cfg.lookback]], -1)
    fut = dow_win[:, cfg.lookback :]
    target = windows[:, cfg.lookback :]
    return (
        past.astype(np.float32),
        fut.astype(np.float32),
        s_idx.astype(np.int64),
        target.astype(np.float32),
    )


def build_model(cfg: SmokeConfig, n_quantiles: int = 3):
    import torch
    import torch.nn as nn

    class TinyForecaster(nn.Module):
        """Seq2seq LSTM: encoder reads the past, decoder reads known future inputs."""

        def __init__(self) -> None:
            super().__init__()
            self.series_emb = nn.Embedding(cfg.n_series, 8)
            self.encoder = nn.LSTM(3 + 8, cfg.hidden, batch_first=True)
            self.decoder = nn.LSTM(2, cfg.hidden, batch_first=True)
            self.head = nn.Linear(cfg.hidden, n_quantiles)

        def forward(self, past, fut, series_id):
            emb = self.series_emb(series_id).unsqueeze(1).expand(-1, past.shape[1], -1)
            _, state = self.encoder(torch.cat([past, emb], dim=-1))
            out, _ = self.decoder(fut, state)
            return self.head(out)  # (batch, horizon, n_quantiles)

    return TinyForecaster()


def pinball_loss(pred, target, quantiles=(0.1, 0.5, 0.9)):
    """Quantile (pinball) loss averaged over quantiles, horizon and batch."""
    import torch

    q = torch.tensor(quantiles, dtype=pred.dtype)
    diff = target.unsqueeze(-1) - pred
    return torch.maximum(q * diff, (q - 1) * diff).mean()


def train(model, data, cfg: SmokeConfig, n_windows: int, rng: np.random.Generator):
    """Train for ``n_windows`` samples; return (seconds, first_loss, last_loss)."""
    import torch

    past, fut, sid, tgt = (torch.from_numpy(a) for a in data)
    order = rng.choice(len(past), size=min(n_windows, len(past)), replace=False)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    model.train()
    losses = []
    t0 = time.perf_counter()
    for i in range(0, len(order), cfg.batch_size):
        b = torch.from_numpy(order[i : i + cfg.batch_size])
        loss = pinball_loss(model(past[b], fut[b], sid[b]), tgt[b])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    seconds = time.perf_counter() - t0
    k = max(1, len(losses) // 10)  # average the first/last 10% of steps to smooth noise
    return seconds, float(np.mean(losses[:k])), float(np.mean(losses[-k:]))


def export_and_check_onnx(model, data, path: Path, batch_sizes=(1, 7, 64)) -> dict:
    """Export with a dynamic batch dimension and compare ONNX Runtime with PyTorch."""
    import contextlib
    import io
    import logging
    import warnings

    import onnxruntime as ort
    import torch

    model.eval()
    past, fut, sid, _ = (torch.from_numpy(a[:8]) for a in data)
    names = ["past", "future", "series_id"]
    torch_logger = logging.getLogger("torch")
    previous_level = torch_logger.level
    torch_logger.setLevel(logging.ERROR)  # hide exporter progress and "torchvision not installed" notices
    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            try:  # modern exporter (PyTorch >= 2.5)
                batch = torch.export.Dim("batch", min=1, max=1_000_000)
                torch.onnx.export(
                    model,
                    (past, fut, sid),
                    str(path),
                    input_names=names,
                    output_names=["quantiles"],
                    dynamo=True,
                    verbose=False,
                    external_data=False,
                    dynamic_shapes={"past": {0: batch}, "fut": {0: batch}, "series_id": {0: batch}},
                )
                exporter = "dynamo"
            except Exception:  # older PyTorch: legacy TorchScript exporter
                axes = {n: {0: "batch"} for n in [*names, "quantiles"]}
                torch.onnx.export(
                    model,
                    (past, fut, sid),
                    str(path),
                    input_names=names,
                    output_names=["quantiles"],
                    dynamic_axes=axes,
                    dynamo=False,
                )
                exporter = "torchscript"
    finally:
        torch_logger.setLevel(previous_level)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    max_diff = 0.0
    for bs in batch_sizes:
        p, f, s = (torch.from_numpy(a[:bs]) for a in data[:3])
        ort_out = session.run(None, {"past": p.numpy(), "future": f.numpy(), "series_id": s.numpy()})[0]
        with torch.no_grad():
            max_diff = max(max_diff, float(np.abs(ort_out - model(p, f, s).numpy()).max()))

    p, f, s = (a[:64] for a in data[:3])
    session.run(None, {"past": p, "future": f, "series_id": s})  # warm-up
    t0 = time.perf_counter()
    session.run(None, {"past": p, "future": f, "series_id": s})
    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "exporter": exporter,
        "file_kb": round(path.stat().st_size / 1024, 1),
        "batch_sizes_checked": list(batch_sizes),
        "max_abs_diff": max_diff,
        "ort_latency_ms_batch64": round(latency_ms, 2),
    }


def sizing_recommendation(sec_per_20k: float) -> dict:
    """Translate measured speed into training-time estimates for Phases 5–6."""
    factor = sec_per_20k / REFERENCE_SEC_PER_20K  # > 1 means slower than the reference
    lstm_30 = REFERENCE_LSTM_EPOCH_SEC * factor * 30 / 60
    tft_30 = REFERENCE_TFT_EPOCH_SEC * factor * 30 / 60
    if tft_30 <= 60:
        preset, note = "standard", "Train both LSTM and TFT at full planned size."
    elif tft_30 <= 180:
        preset, note = "standard", "Full size works; run TFT training in the background or overnight."
    else:
        preset, note = "light", "Use the light preset (smaller hidden size, stride 14) for TFT."
    return {
        "relative_to_reference": round(factor, 2),
        "est_lstm_30_epochs_min": math.ceil(lstm_30),
        "est_tft_30_epochs_min": math.ceil(tft_30),
        "preset": preset,
        "note": note,
    }
