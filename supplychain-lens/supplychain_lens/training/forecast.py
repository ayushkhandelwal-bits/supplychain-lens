"""Train, evaluate and export the neural forecasters (models A and B).

    python -m supplychain_lens.training.forecast --model lstm
    python -m supplychain_lens.training.forecast --model tft

Protocol (identical for both models and the baselines)
  • chronological split from config: train ≤ 2025-06-30 < validation ≤ 2025-09-30 < test (Q4 2025)
  • scalers fitted on the training period only
  • training windows: every origin whose 28-day horizon ends inside the training period
  • early stopping on validation pinball loss (10 weekly validation origins × 400 series)
  • test = 3 origins (1 Oct, 29 Oct, 26 Nov 2025) × 400 series × 28 days, never seen in training
  • best checkpoint exported to ONNX and checked against PyTorch at batch sizes 1 and N
"""

from __future__ import annotations

import argparse
import copy
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from supplychain_lens.config import get_settings
from supplychain_lens.evaluation.metrics import forecast_summary
from supplychain_lens.features.forecasting import (
    FUTURE_FEATURES,
    PAST_FEATURES,
    Panel,
    evaluation_origins,
    load_panel,
    training_origins,
    window_arrays,
)
from supplychain_lens.models.lstm import LSTMConfig, LSTMForecaster
from supplychain_lens.models.tft import TemporalFusionTransformer, TFTConfig
from supplychain_lens.serving import registry
from supplychain_lens.utils.seed import set_global_seed

QUANTILES = torch.tensor([0.1, 0.5, 0.9])
DISPLAY_NAMES = {"lstm": "LSTM (PyTorch)", "tft": "TFT (PyTorch)"}

TRAINING_DEFAULTS = {
    "lstm": {"lr": 2e-3, "batch_size": 256, "windows_per_epoch": 40_000, "max_epochs": 30, "patience": 5},
    "tft": {"lr": 1.5e-3, "batch_size": 256, "windows_per_epoch": 30_000, "max_epochs": 25, "patience": 5},
}


def pinball_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = target.unsqueeze(-1) - pred
    return torch.maximum(QUANTILES * diff, (QUANTILES - 1) * diff).mean()


def build_model(kind: str, panel: Panel) -> torch.nn.Module:
    n = dict(n_past=len(PAST_FEATURES), n_future=len(FUTURE_FEATURES),
             n_sku=int(panel.keys.sku_code.max()) + 1, n_warehouse=int(panel.keys.warehouse_code.max()) + 1,
             n_category=int(panel.keys.category_code.max()) + 1)
    if kind == "lstm":
        return LSTMForecaster(replace(LSTMConfig(), **n))
    if kind == "tft":
        return TemporalFusionTransformer(replace(TFTConfig(), **n))
    raise ValueError(kind)


def _tensors(arrs):
    past, fut, codes, log_scale, target = arrs
    return (torch.from_numpy(np.ascontiguousarray(past)), torch.from_numpy(np.ascontiguousarray(fut)),
            torch.from_numpy(codes.astype(np.int64)), torch.from_numpy(log_scale),
            torch.from_numpy(np.ascontiguousarray(target)))


def validation_pairs(panel: Panel) -> np.ndarray:
    H = panel.settings.horizon_days
    origins = range(panel.train_end_idx + 1, panel.val_end_idx - H + 2, 7)
    return np.array([(s, o) for o in origins for s in range(panel.n_series)])


@torch.no_grad()
def evaluate_loss(model, panel, feats, pairs, batch_size=1024) -> float:
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(pairs), batch_size):
        p, f, c, ls, y = _tensors(window_arrays(panel, feats, pairs[i : i + batch_size]))
        total += pinball_loss(model(p, f, c, ls), y).item() * len(y)
        n += len(y)
    return total / n


def train(kind: str, panel: Panel, seed: int, log=print, **overrides):
    hp = {**TRAINING_DEFAULTS[kind], **overrides}
    rng = set_global_seed(seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    feats = panel.features()
    train_pairs = np.array([(s, o) for o in training_origins(panel) for s in range(panel.n_series)])
    val_pairs = validation_pairs(panel)
    model = build_model(kind, panel)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    best, best_state, bad, curve = float("inf"), None, 0, []
    t_start = time.perf_counter()
    for epoch in range(1, hp["max_epochs"] + 1):
        model.train()
        t0 = time.perf_counter()
        sel = train_pairs[rng.choice(len(train_pairs), hp["windows_per_epoch"], replace=False)]
        losses = []
        for i in range(0, len(sel), hp["batch_size"]):
            p, f, c, ls, y = _tensors(window_arrays(panel, feats, sel[i : i + hp["batch_size"]]))
            loss = pinball_loss(model(p, f, c, ls), y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        val = evaluate_loss(model, panel, feats, val_pairs)
        sched.step(val)
        curve.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val,
                      "lr": opt.param_groups[0]["lr"], "seconds": round(time.perf_counter() - t0, 1)})
        log(f"  epoch {epoch:>2}  train {np.mean(losses):.4f}  val {val:.4f}  ({curve[-1]['seconds']} s)")
        if val < best - 1e-5:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= hp["patience"]:
                log(f"  early stop: no improvement for {hp['patience']} epochs")
                break
    model.load_state_dict(best_state)
    info = {"best_val_loss": best, "epochs_run": len(curve), "best_epoch": int(np.argmin([c["val_loss"] for c in curve]) + 1),
            "train_seconds": round(time.perf_counter() - t_start, 1), "hyperparameters": hp,
            "train_windows_available": int(len(train_pairs)), "val_windows": int(len(val_pairs))}
    return model, pd.DataFrame(curve), info


@torch.no_grad()
def predict(model, panel: Panel, origins: list[int], name: str) -> pd.DataFrame:
    model.eval()
    feats = panel.features()
    H = panel.settings.horizon_days
    pairs = np.array([(s, o) for o in origins for s in range(panel.n_series)])
    p, f, c, ls, _ = _tensors(window_arrays(panel, feats, pairs))
    q = model(p, f, c, ls).numpy() * panel.scale[pairs[:, 0], None, None]
    q = q.clip(0)
    s, o = np.repeat(pairs[:, 0], H), np.repeat(pairs[:, 1], H)
    h = np.tile(np.arange(1, H + 1), len(pairs))
    t = o + h - 1
    return pd.DataFrame({"model": name, "series": s, "origin_idx": o, "h": h, "t_idx": t, "y": panel.y[s, t],
                         "p10": q[..., 0].reshape(-1), "p50": q[..., 1].reshape(-1), "p90": q[..., 2].reshape(-1)})


@torch.no_grad()
def measure_latency(model, panel: Panel, origin: int, repeats: int = 5) -> float:
    model.eval()
    pairs = np.array([(s, origin) for s in range(panel.n_series)])
    p, f, c, ls, _ = _tensors(window_arrays(panel, panel.features(), pairs))
    model(p, f, c, ls)
    t0 = time.perf_counter()
    for _ in range(repeats):
        model(p, f, c, ls)
    return (time.perf_counter() - t0) / repeats * 1000


def export_onnx(model, panel: Panel, path) -> dict:
    """Export with a dynamic batch axis; verify ONNX Runtime == PyTorch at batch 1 and 400."""
    import contextlib
    import io
    import logging
    import warnings

    import onnxruntime as ort

    model.eval()
    feats = panel.features()
    origin = evaluation_origins(panel, "test")[0]
    pairs = np.array([(s, origin) for s in range(panel.n_series)])
    p, f, c, ls, _ = _tensors(window_arrays(panel, feats, pairs))
    names = ["past", "future", "codes", "log_scale"]
    logging.getLogger("torch").setLevel(logging.ERROR)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        batch = torch.export.Dim("batch", min=1, max=100_000)
        torch.onnx.export(model, (p[:8], f[:8], c[:8], ls[:8]), str(path), input_names=names,
                          output_names=["quantiles"], dynamo=True, external_data=False,
                          dynamic_shapes={"past": {0: batch}, "future": {0: batch}, "codes": {0: batch},
                                          "log_scale": {0: batch}})
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    max_diff = 0.0
    for n in (1, len(pairs)):
        feed = {"past": p[:n].numpy(), "future": f[:n].numpy(), "codes": c[:n].numpy(), "log_scale": ls[:n].numpy()}
        out = sess.run(None, feed)[0]
        with torch.no_grad():
            max_diff = max(max_diff, float(np.abs(out - model(p[:n], f[:n], c[:n], ls[:n]).numpy()).max()))
    feed = {"past": p.numpy(), "future": f.numpy(), "codes": c.numpy(), "log_scale": ls.numpy()}
    sess.run(None, feed)
    t0 = time.perf_counter()
    for _ in range(5):
        sess.run(None, feed)
    ort_ms = (time.perf_counter() - t0) / 5 * 1000
    return {"max_abs_diff_vs_pytorch": max_diff, "batch_sizes_checked": [1, len(pairs)],
            "onnxruntime_latency_ms_400_series": round(ort_ms, 1), "file_kb": round(path.stat().st_size / 1024, 1)}


@torch.no_grad()
def tft_interpretation(model, panel: Panel, origins: list[int]) -> dict:
    """Average variable-selection and attention weights over the test windows."""
    feats = panel.features()
    pairs = np.array([(s, o) for o in origins for s in range(panel.n_series)])
    p, f, c, ls, _ = _tensors(window_arrays(panel, feats, pairs))
    model.eval()
    _, w = model(p, f, c, ls, return_weights=True)
    L = panel.settings.lookback_days
    attn = w["attention"].mean(0).numpy()  # (H, L + H)
    return {
        "static_variables": dict(zip(["SKU", "warehouse", "category", "demand scale"], w["static"].mean(0).tolist(), strict=True)),
        "past_variables": dict(zip(PAST_FEATURES, w["past"].mean((0, 1)).tolist(), strict=True)),
        "future_variables": dict(zip(FUTURE_FEATURES, w["future"].mean((0, 1)).tolist(), strict=True)),
        "attention_over_past_days": attn[:, :L].mean(0).tolist(),  # lag -L … -1
        "attention_matrix_first_week": attn[:7].tolist(),
        "note": "Weights are associational (what the model used), not causal effects.",
    }


def model_card(kind: str, run, metrics: dict, info: dict, panel: Panel, onnx_info: dict, n_params: int) -> str:
    s = panel.settings
    t, v = metrics["test"], metrics["val"]
    return f"""# Model card — {DISPLAY_NAMES[kind]}

**Version:** `{run.version}` · **Task:** 28-day daily demand forecast per SKU × warehouse with P10/P50/P90
**Data:** synthetic (NEXUS-SCCT data model, nexus_v2), 400 series × {len(panel.dates)} days

## Training
- Split: train ≤ {s.train_end} · validation ≤ {s.val_end} · test {s.test_start} → {s.calendar_end}
- Inputs: past {s.lookback_days} days of {", ".join(PAST_FEATURES)}; known future {", ".join(FUTURE_FEATURES)};
  static SKU / warehouse / category embeddings and log demand scale
- Loss: pinball (quantile) loss; scaling by train-period mean per series
- Parameters: {n_params:,} · epochs run: {info['epochs_run']} (best {info['best_epoch']}) · training time {info['train_seconds']} s on CPU
- Hyperparameters: `{info['hyperparameters']}`

## Results (demand units)
| Split | WAPE | MAE | RMSE | Bias | P10–P90 coverage |
|---|---|---|---|---|---|
| Validation | {v['wape']:.3f} | {v['mae']:.1f} | {v['rmse']:.1f} | {v['bias']:+.3f} | {v.get('coverage_p10_p90', float('nan')):.1%} |
| Test (Q4 2025) | {t['wape']:.3f} | {t['mae']:.1f} | {t['rmse']:.1f} | {t['bias']:+.3f} | {t.get('coverage_p10_p90', float('nan')):.1%} |

Comparison with baselines and the other neural model: `reports/forecast_comparison.md`.

## Serving
- ONNX file {onnx_info['file_kb']} KB; max |ONNX − PyTorch| = {onnx_info['max_abs_diff_vs_pytorch']:.1e}
- ONNX Runtime latency for all 400 series: {onnx_info['onnxruntime_latency_ms_400_series']} ms (CPU)

## Limitations
- Trained and evaluated on synthetic data only; results are not evidence of real-world accuracy.
- P10/P90 are forecast quantiles; their empirical coverage is reported above rather than assumed.
- No promotions are known after the snapshot date, so live forecasts assume none.
"""


def run(kind: str, seed: int | None = None, log=print, **overrides) -> registry.Run:
    s = get_settings()
    seed = s.seed if seed is None else seed
    panel = load_panel()
    log(f"Training {DISPLAY_NAMES[kind]} on {panel.n_series} series …")
    model, curve, info = train(kind, panel, seed, log=log, **overrides)
    val = predict(model, panel, evaluation_origins(panel, "val"), DISPLAY_NAMES[kind])
    test = predict(model, panel, evaluation_origins(panel, "test"), DISPLAY_NAMES[kind])
    metrics = {"val": forecast_summary(val), "test": forecast_summary(test),
               "pytorch_latency_ms_400_series": round(measure_latency(model, panel, evaluation_origins(panel, "test")[0]), 1)}
    run_ = registry.start_run(f"forecast_{kind}")
    onnx_info = export_onnx(model, panel, run_.path / "model.onnx")
    metrics["onnx"] = onnx_info
    n_params = sum(p.numel() for p in model.parameters())
    torch.save(model.state_dict(), run_.path / "model.pt")
    registry.write_json(run_, "config.json", {"kind": kind, "display_name": DISPLAY_NAMES[kind], "seed": seed,
                                              "model_config": model.cfg.to_dict(), "parameters": n_params,
                                              "lookback": s.lookback_days, "horizon": s.horizon_days,
                                              "past_features": PAST_FEATURES, "future_features": FUTURE_FEATURES,
                                              "split": {"train_end": s.train_end, "val_end": s.val_end, "test_start": s.test_start},
                                              **info})
    registry.write_json(run_, "scaler.json", {"method": "divide by train-period mean demand per series",
                                              "series": panel.keys[["sku_id", "warehouse_id"]].assign(scale=panel.scale).to_dict("records")})
    registry.write_json(run_, "metrics.json", metrics)
    curve.to_csv(run_.path / "loss_curve.csv", index=False)
    pd.concat([val.assign(split="val"), test.assign(split="test")]).to_parquet(run_.path / "predictions.parquet", index=False)
    if kind == "tft":
        registry.write_json(run_, "interpretation.json", tft_interpretation(model, panel, evaluation_origins(panel, "test")))
    (run_.path / "model_card.md").write_text(model_card(kind, run_, metrics, info, panel, onnx_info, n_params), encoding="utf-8")
    registry.finalize(run_)
    log(f"Saved {run_.path}  test WAPE {metrics['test']['wape']:.4f}")
    return run_


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lstm", "tft"], required=True)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--windows-per-epoch", type=int, default=None)
    a = ap.parse_args()
    over = {k: v for k, v in {"max_epochs": a.max_epochs, "windows_per_epoch": a.windows_per_epoch}.items() if v}
    run(a.model, **over)
