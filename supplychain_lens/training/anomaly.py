"""Train and evaluate the anomaly detector (model C) against a robust z-score baseline.

    python -m supplychain_lens.training.anomaly

Protocol
  • autoencoder trained on windows ending in the training period (unsupervised — no labels)
  • threshold: chosen on the VALIDATION quarter as the score that maximises F1 against the
    injected synthetic anomalies, subject to an alert budget (≤ 0.5% of series-days);
    the same rule is applied to the baseline
  • test quarter (Q4 2025): precision, recall (event level), false-positive rate (day level)

"Anomaly detected" means "investigation required", never "confirmed disruption".
"""

from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import torch

from supplychain_lens.config import get_settings
from supplychain_lens.evaluation import labels as lab
from supplychain_lens.features.anomaly import FEATURES, WINDOW, build_ops_panel, windows
from supplychain_lens.models.autoencoder import Autoencoder
from supplychain_lens.serving import registry
from supplychain_lens.utils.seed import set_global_seed

ALERT_BUDGET = 0.005
# Contextual scoring: every feature is an input, but only behavioural signals on the latest day are
# scored. Promotions, holidays, stock cover and pipeline explain behaviour; they are not anomalies.
SCORED = ["demand_z", "unexplained_inv", "receipt_delay"]
MATCHED_ALERTS = 100  # head-to-head: each detector's top-100 test days
SCORED_IDX = [FEATURES.index(f) for f in SCORED]


def _pairs(S: int, t_range: range) -> np.ndarray:
    return np.array([(s, t) for t in t_range for s in range(S)])


def ae_scores(model, x, pairs, mean, std, batch=8192) -> tuple[np.ndarray, np.ndarray]:
    """Score = mean squared reconstruction error of the scored features on the window's last day.

    Also returns the last-day error of every feature (for explanations)."""
    model.eval()
    scores, per_feat = [], []
    with torch.no_grad():
        for i in range(0, len(pairs), batch):
            w = torch.from_numpy(windows(x, pairs[i : i + batch], mean, std))
            last = ((model(w) - w) ** 2).view(len(w), WINDOW, -1)[:, -1, :].numpy()
            scores.append(last[:, SCORED_IDX].mean(1))
            per_feat.append(last)
    return np.concatenate(scores), np.concatenate(per_feat)


def robust_z_scores(x, pairs, median, mad) -> np.ndarray:
    """Baseline: largest |robust z| of the scored features on the window's last day."""
    s, t = pairs[:, 0], pairs[:, 1]
    z = np.abs((x[s, t][:, SCORED_IDX] - median[SCORED_IDX]) / mad[SCORED_IDX])
    return z.max(1)


def choose_threshold(scores, y_day, budget=ALERT_BUDGET) -> float:
    """Validation threshold: best F1 among thresholds that flag ≤ budget of days."""
    cand = np.quantile(scores, np.linspace(1 - budget, 0.99999, 60))
    best, best_f1 = cand[0], -1
    for th in cand:
        pred = scores >= th
        tp = (pred & (y_day == 1)).sum()
        p, r = tp / max(pred.sum(), 1), tp / max(y_day.sum(), 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        if f1 > best_f1:
            best, best_f1 = th, f1
    return float(best)


def evaluate(scores, threshold, pairs, y_day, which, recorded, n_events_in_period) -> dict:
    pred = scores >= threshold
    tp_days = (pred & (y_day == 1)).sum()
    detected = set(which[pred & (which >= 0)].tolist())
    explained = pred & (y_day == 0) & recorded
    return {
        "threshold": float(threshold), "alerts": int(pred.sum()), "alert_rate": float(pred.mean()),
        "precision_injected": float(tp_days / max(pred.sum(), 1)),
        "precision_incl_recorded_events": float((tp_days + explained.sum()) / max(pred.sum(), 1)),
        "event_recall": float(len(detected) / max(n_events_in_period, 1)),
        "events_detected": len(detected), "events_in_period": int(n_events_in_period),
        "false_positive_rate_days": float((pred & (y_day == 0)).sum() / max((y_day == 0).sum(), 1)),
    }


def run(seed: int | None = None, log=print, epochs: int = 15) -> registry.Run:
    s = get_settings()
    seed = s.seed if seed is None else seed
    rng = set_global_seed(seed)
    ops = build_ops_panel()
    S, T = len(ops.keys), len(ops.dates)
    tr_end = (pd.Timestamp(s.train_end) - ops.dates[0]).days
    va_end = (pd.Timestamp(s.val_end) - ops.dates[0]).days
    y_day, which = lab.anomaly_day_labels(ops.keys, ops.dates, ops.primary_supplier, ops.receipts)
    recorded = lab.recorded_event_days(ops.keys, ops.dates, ops.primary_supplier)
    anomalies = lab.injected_anomalies()

    train_pairs = _pairs(S, range(35, tr_end + 1))
    val_pairs, test_pairs = _pairs(S, range(tr_end + 1, va_end + 1)), _pairs(S, range(va_end + 1, T))
    flat = ops.x[:, 35 : tr_end + 1].reshape(-1, len(FEATURES))
    mean, std = flat.mean(0), flat.std(0) + 1e-6
    median = np.median(flat, 0)
    mad = 1.4826 * np.median(np.abs(flat - median), 0)
    mad = np.where(mad < 1e-3, std, mad)  # binary features have MAD 0 -> fall back to std

    model = Autoencoder(WINDOW * len(FEATURES))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    curve, best, best_state = [], np.inf, None
    sample_val = val_pairs[rng.choice(len(val_pairs), 20000, replace=False)]
    t_start = time.perf_counter()
    # Rare contexts (holiday weeks, promotions) are oversampled so the model learns that they are normal.
    ctx = ops.x[train_pairs[:, 0], train_pairs[:, 1]][:, [FEATURES.index("is_holiday"), FEATURES.index("is_promo")]].max(1) > 0
    rare = train_pairs[ctx]
    for epoch in range(1, epochs + 1):
        model.train()
        sel = np.concatenate([train_pairs[rng.choice(len(train_pairs), 100_000, replace=False)],
                              rare[rng.choice(len(rare), min(len(rare), 40_000), replace=False)]])
        sel = sel[rng.permutation(len(sel))]
        losses = []
        for i in range(0, len(sel), 512):
            w = torch.from_numpy(windows(ops.x, sel[i : i + 512], mean, std))
            loss = ((model(w) - w) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        val_loss = float(ae_scores(model, ops.x, sample_val, mean, std)[0].mean())
        curve.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss})
        log(f"  epoch {epoch:>2}  train {np.mean(losses):.4f}  val {val_loss:.4f}")
        if val_loss < best:
            best, best_state = val_loss, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    train_seconds = round(time.perf_counter() - t_start, 1)

    def labels_for(pairs):
        return y_day[pairs[:, 0], pairs[:, 1]], which[pairs[:, 0], pairs[:, 1]], recorded[pairs[:, 0], pairs[:, 1]]

    def n_events(lo, hi):
        return int(((anomalies.start_date >= ops.dates[lo]) & (anomalies.start_date <= ops.dates[hi])).sum())

    res = {}
    for name, score_fn in (("Autoencoder (PyTorch)", lambda p: ae_scores(model, ops.x, p, mean, std)[0]),
                           ("Robust z-score", lambda p: robust_z_scores(ops.x, p, median, mad))):
        sv, st_ = score_fn(val_pairs), score_fn(test_pairs)
        yv, wv, rv = labels_for(val_pairs)
        yt, wt, rt = labels_for(test_pairs)
        th = choose_threshold(sv, yv)
        k_th = np.sort(st_)[-MATCHED_ALERTS]  # same number of alerts for every detector -> fair comparison
        res[name] = {"val": evaluate(sv, th, val_pairs, yv, wv, rv, n_events(tr_end + 1, va_end)),
                     "test": evaluate(st_, th, test_pairs, yt, wt, rt, n_events(va_end + 1, T - 1)),
                     "test_matched_alerts": evaluate(st_, k_th, test_pairs, yt, wt, rt, n_events(va_end + 1, T - 1))}
        log(f"  {name:<22} test precision {res[name]['test']['precision_injected']:.2f} "
            f"recall {res[name]['test']['event_recall']:.2f} alerts {res[name]['test']['alerts']}")
        if name.startswith("Autoencoder"):
            ae_test_scores, ae_threshold = st_, th
            _, per_feat = ae_scores(model, ops.x, test_pairs, mean, std)

    # Alerts table (test quarter) for the dashboard. Truth columns are for the evaluation page only.
    flagged = ae_test_scores >= ae_threshold
    fp = test_pairs[flagged]
    top_scored = np.argsort(-per_feat[flagged][:, SCORED_IDX], axis=1)[:, :2]
    top = np.array(SCORED_IDX)[top_scored]
    y_t, w_t, r_t = labels_for(fp)
    alerts = pd.DataFrame({
        "sku_id": ops.keys.sku_id.to_numpy()[fp[:, 0]], "warehouse_id": ops.keys.warehouse_id.to_numpy()[fp[:, 0]],
        "category": ops.keys.category.to_numpy()[fp[:, 0]], "supplier_id": ops.primary_supplier.to_numpy()[fp[:, 0]],
        "date": ops.dates[fp[:, 1]], "score": ae_test_scores[flagged], "threshold": ae_threshold,
        "top_feature_1": np.array(FEATURES)[top[:, 0]], "top_feature_2": np.array(FEATURES)[top[:, 1]],
        "eval_matches_injected": y_t.astype(bool),
        "eval_injected_type": [anomalies.type.iloc[k] if k >= 0 else "" for k in w_t],
        "eval_during_recorded_event": r_t,
    })
    for j, f in enumerate(FEATURES):
        alerts[f"err_{f}"] = per_feat[flagged][:, j]
        alerts[f"raw_{f}"] = ops.x[fp[:, 0], fp[:, 1], j]
    alerts = alerts.sort_values("score", ascending=False)

    run_ = registry.start_run("anomaly_autoencoder")
    torch.save(model.state_dict(), run_.path / "model.pt")
    import contextlib
    import io
    import logging
    import warnings
    logging.getLogger("torch").setLevel(logging.ERROR)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        sample = torch.from_numpy(windows(ops.x, test_pairs[:16], mean, std))
        torch.onnx.export(model, (sample,), str(run_.path / "model.onnx"), input_names=["window"],
                          output_names=["reconstruction"], dynamo=True, external_data=False,
                          dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=1_000_000)}})
    import onnxruntime as ort
    sess = ort.InferenceSession(str(run_.path / "model.onnx"), providers=["CPUExecutionProvider"])
    w = windows(ops.x, test_pairs[:400], mean, std)
    with torch.no_grad():
        diff = float(np.abs(sess.run(None, {"window": w})[0] - model(torch.from_numpy(w)).numpy()).max())
    registry.write_json(run_, "scaler.json", {"features": FEATURES, "scored_features": SCORED, "window": WINDOW, "mean": mean.tolist(),
                                              "std": std.tolist(), "median": median.tolist(), "mad": mad.tolist()})
    registry.write_json(run_, "config.json", {"model": "Autoencoder", "inputs": WINDOW * len(FEATURES), "hidden": 64,
                                              "bottleneck": 8, "epochs": epochs, "train_seconds": train_seconds,
                                              "alert_budget": ALERT_BUDGET, "seed": seed,
                                              "parameters": sum(p.numel() for p in model.parameters())})
    registry.write_json(run_, "metrics.json", {"models": res, "threshold": ae_threshold,
                                               "onnx_max_abs_diff": diff, "label_note":
                                               "Labels are injected synthetic anomalies (hidden truth), used only for evaluation."})
    pd.DataFrame(curve).to_csv(run_.path / "loss_curve.csv", index=False)
    alerts.to_parquet(run_.path / "test_alerts.parquet", index=False)
    (run_.path / "model_card.md").write_text(_card(run_, res, ae_threshold, train_seconds), encoding="utf-8")
    registry.finalize(run_)
    log(f"Saved {run_.path}")
    return run_


def _card(run_, res, threshold, seconds) -> str:
    ae, bl = res["Autoencoder (PyTorch)"]["test"], res["Robust z-score"]["test"]
    return f"""# Model card — Autoencoder anomaly detector

**Version:** `{run_.version}` · **Task:** flag unusual SKU × warehouse weeks for investigation
**Data:** synthetic (nexus_v2); 7-day windows of {len(FEATURES)} daily features ({", ".join(FEATURES)})

## Method
- Dense autoencoder 56 → 64 → 8 → 64 → 56, trained without labels on windows ending ≤ train end ({seconds} s on CPU)
- Score = mean squared reconstruction error of the behavioural features ({", ".join(SCORED)}) on the
  window's last day; promotions, holidays, stock cover and pipeline are context inputs only. Threshold {threshold:.3f} chosen on the validation quarter
  (best F1 against injected anomalies within an alert budget of {ALERT_BUDGET:.1%} of series-days)

## Test quarter (Q4 2025)
| Detector | Alerts | Precision (injected) | Precision (incl. recorded events) | Event recall | Day-level FPR |
|---|---|---|---|---|---|
| Autoencoder | {ae['alerts']} | {ae['precision_injected']:.2f} | {ae['precision_incl_recorded_events']:.2f} | {ae['event_recall']:.2f} ({ae['events_detected']}/{ae['events_in_period']}) | {ae['false_positive_rate_days']:.4f} |
| Robust z-score | {bl['alerts']} | {bl['precision_injected']:.2f} | {bl['precision_incl_recorded_events']:.2f} | {bl['event_recall']:.2f} ({bl['events_detected']}/{bl['events_in_period']}) | {bl['false_positive_rate_days']:.4f} |

## Head-to-head at the same alert volume (top {MATCHED_ALERTS} test days each)
| Detector | Precision (injected) | Precision (incl. recorded events) | Event recall |
|---|---|---|---|
| Autoencoder | {res['Autoencoder (PyTorch)']['test_matched_alerts']['precision_injected']:.2f} | {res['Autoencoder (PyTorch)']['test_matched_alerts']['precision_incl_recorded_events']:.2f} | {res['Autoencoder (PyTorch)']['test_matched_alerts']['event_recall']:.2f} |
| Robust z-score | {res['Robust z-score']['test_matched_alerts']['precision_injected']:.2f} | {res['Robust z-score']['test_matched_alerts']['precision_incl_recorded_events']:.2f} | {res['Robust z-score']['test_matched_alerts']['event_recall']:.2f} |

Only {ae['events_in_period']} injected anomalies start in the test quarter, so these rates are uncertain (one event = {1 / max(ae['events_in_period'], 1):.0%} of recall).

## Limitations
- Evaluated against injected synthetic anomalies; real anomalies are more varied.
- An alert means "investigation required", not a confirmed disruption.
- Per-feature reconstruction errors show which inputs looked unusual; they are not causes.
"""


if __name__ == "__main__":
    run()
