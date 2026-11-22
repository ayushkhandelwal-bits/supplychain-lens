# Supervised supplier-risk idea adapted from NEXUS-SCCT supplier_risk_model.py (XGBoost, 90-day label);
# reimplemented on operational features with a 14-day label and a neural model.
"""Model D — supplier disruption-risk classifier: neural network vs logistic regression vs XGBoost.

    python -m supplychain_lens.training.risk

Task:   P(a disruption affecting supplier s STARTS within the next 14 days | data up to day t)
Labels: built from the generator's event log (hidden truth) — synthetic supervised learning,
        NOT validated on real company outcomes.
Split:  purged chronological split — train rows t ≤ train_end − 14 days (their labels end inside
        training), validation train_end < t ≤ val_end − 14, test val_end < t ≤ snapshot − 14.
Choice: the model with the best VALIDATION PR-AUC is selected; its threshold maximises
        validation F1. Test metrics are reported for all models.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from supplychain_lens.config import get_settings
from supplychain_lens.evaluation import labels as lab
from supplychain_lens.evaluation.metrics import calibration_table, classification_summary
from supplychain_lens.features.risk import FEATURES, build_risk_panel
from supplychain_lens.serving import registry
from supplychain_lens.utils.seed import set_global_seed

HORIZON = 14


class RiskMLP(nn.Module):
    def __init__(self, n: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden // 2, 1))

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


def best_f1_threshold(y, p) -> float:
    cand = np.unique(np.quantile(p, np.linspace(0.5, 0.995, 100)))
    f1 = []
    for th in cand:
        pred = p >= th
        tp = (pred & (y == 1)).sum()
        prec, rec = tp / max(pred.sum(), 1), tp / max(y.sum(), 1)
        f1.append(2 * prec * rec / max(prec + rec, 1e-9))
    return float(cand[int(np.argmax(f1))])


def split_rows(panel, y, s):
    T = len(panel.dates)
    idx = lambda d: (pd.Timestamp(d) - panel.dates[0]).days  # noqa: E731
    tr_end, va_end = idx(s.train_end), idx(s.val_end)
    ranges = {"train": range(90, tr_end - HORIZON + 1), "val": range(tr_end + 1, va_end - HORIZON + 1),
              "test": range(va_end + 1, T - HORIZON)}
    out = {}
    for name, r in ranges.items():
        pairs = np.array([(si, t) for t in r for si in range(len(panel.suppliers))])
        out[name] = (panel.X[pairs[:, 0], pairs[:, 1]], y[pairs[:, 0], pairs[:, 1]].astype(int), pairs)
    return out


def train_mlp(Xtr, ytr, Xva, yva, seed, log, epochs=60, patience=8):
    set_global_seed(seed)
    model = RiskMLP(Xtr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.float32))
    Xv, yv = torch.from_numpy(Xva), torch.from_numpy(yva.astype(np.float32))
    loss_fn = nn.BCELoss()
    best, state, bad, curve = np.inf, None, 0, []
    g = torch.Generator().manual_seed(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        losses = []
        for i in range(0, len(perm), 512):
            b = perm[i : i + 512]
            loss = loss_fn(model(Xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xv), yv).item()
        curve.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": vl})
        if vl < best - 1e-5:
            best, state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    log(f"  MLP: {len(curve)} epochs, best val BCE {best:.4f}")
    return model, pd.DataFrame(curve)


def permutation_importance(predict, X, y, n_repeats=5, seed=42) -> pd.DataFrame:
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(seed)
    base = average_precision_score(y, predict(X))
    rows = []
    for j, f in enumerate(FEATURES):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(base - average_precision_score(y, predict(Xp)))
        rows.append({"feature": f, "pr_auc_drop": float(np.mean(drops)), "std": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("pr_auc_drop", ascending=False)


def run(seed: int | None = None, log=print) -> registry.Run:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    s = get_settings()
    seed = s.seed if seed is None else seed
    t0 = time.perf_counter()
    panel = build_risk_panel()
    y = lab.disruption_labels(list(panel.suppliers.supplier_id), panel.dates, HORIZON)
    parts = split_rows(panel, y, s)
    (Xtr, ytr, _), (Xva, yva, _), (Xte, yte, pte) = parts["train"], parts["val"], parts["test"]
    log(f"  rows train {len(ytr):,} ({ytr.mean():.1%} positive) · val {len(yva):,} ({yva.mean():.1%}) · "
        f"test {len(yte):,} ({yte.mean():.1%})")
    scaler = StandardScaler().fit(Xtr)
    Ztr, Zva, Zte = (scaler.transform(a).astype(np.float32) for a in (Xtr, Xva, Xte))

    lr = LogisticRegression(max_iter=2000, C=1.0).fit(Ztr, ytr)
    xgb = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                        min_child_weight=5, eval_metric="aucpr", early_stopping_rounds=40, random_state=seed,
                        n_jobs=2).fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    mlp, curve = train_mlp(Ztr, ytr, Zva, yva, seed, log)

    def mlp_predict(Z):
        with torch.no_grad():
            return mlp(torch.from_numpy(Z.astype(np.float32))).numpy()

    predictors = {
        "Neural network (PyTorch MLP)": (mlp_predict, "scaled"),
        "Logistic regression": (lambda Z: lr.predict_proba(Z)[:, 1], "scaled"),
        "XGBoost": (lambda X: xgb.predict_proba(X)[:, 1], "raw"),
    }
    data = {"scaled": (Zva, Zte), "raw": (Xva, Xte)}
    results, test_probs = {}, {}
    for name, (fn, kind) in predictors.items():
        pv, pt = fn(data[kind][0]), fn(data[kind][1])
        th = best_f1_threshold(yva, pv)
        results[name] = {"val": classification_summary(yva, pv, th), "test": classification_summary(yte, pt, th),
                         "calibration_test": calibration_table(yte, pt).to_dict("records")}
        test_probs[name] = pt
        log(f"  {name:<30} val PR-AUC {results[name]['val']['pr_auc']:.3f}  test PR-AUC "
            f"{results[name]['test']['pr_auc']:.3f}  ROC-AUC {results[name]['test']['roc_auc']:.3f}")
    selected = max(results, key=lambda m: results[m]["val"]["pr_auc"])
    sel_fn, sel_kind = predictors[selected]
    importance = permutation_importance(sel_fn, data[sel_kind][1], yte)

    # Current scores at the snapshot (what the dashboard shows "today").
    t_snap = len(panel.dates) - 1
    X_now = panel.X[:, t_snap]
    Z_now = scaler.transform(X_now).astype(np.float32)
    now = panel.suppliers[["supplier_id", "name", "country", "region", "category", "tier"]].copy()
    for name, (fn, kind) in predictors.items():
        now[f"p_{name}"] = fn(Z_now if kind == "scaled" else X_now)
    now["probability"] = now[f"p_{selected}"]
    now["threshold"] = results[selected]["test"]["threshold"]
    for j, f in enumerate(FEATURES):
        now[f] = X_now[:, j]

    run_ = registry.start_run("risk_classifier")
    torch.save(mlp.state_dict(), run_.path / "model.pt")
    import contextlib
    import io
    import logging
    import warnings
    logging.getLogger("torch").setLevel(logging.ERROR)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        torch.onnx.export(mlp, (torch.from_numpy(Zte[:8]),), str(run_.path / "model.onnx"), input_names=["features"],
                          output_names=["probability"], dynamo=True, external_data=False,
                          dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=1_000_000)}})
    xgb.save_model(str(run_.path / "xgboost.json"))
    registry.write_json(run_, "scaler.json", {"features": FEATURES, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()})
    registry.write_json(run_, "logistic_regression.json", {"features": FEATURES, "coef": lr.coef_[0].tolist(),
                                                           "intercept": float(lr.intercept_[0]), "inputs": "standardised"})
    registry.write_json(run_, "config.json", {"horizon_days": HORIZON, "selected_model": selected, "seed": seed,
                                              "selection_rule": "best validation PR-AUC; threshold = best validation F1",
                                              "mlp": {"hidden": 64, "parameters": sum(p.numel() for p in mlp.parameters())},
                                              "xgboost_best_iteration": int(xgb.best_iteration),
                                              "rows": {"train": len(ytr), "val": len(yva), "test": len(yte)},
                                              "positive_rate": {"train": float(ytr.mean()), "val": float(yva.mean()), "test": float(yte.mean())},
                                              "seconds": round(time.perf_counter() - t0, 1)})
    registry.write_json(run_, "metrics.json", {"models": results, "selected_model": selected,
                                               "label_note": "Synthetic labels from the generator's event log; not real outcomes."})
    curve.to_csv(run_.path / "loss_curve.csv", index=False)
    importance.to_csv(run_.path / "permutation_importance.csv", index=False)
    now.to_parquet(run_.path / "current_scores.parquet", index=False)
    pd.DataFrame({"supplier_id": panel.suppliers.supplier_id.to_numpy()[pte[:, 0]], "date": panel.dates[pte[:, 1]],
                  "y": yte, **{f"p_{k}": v for k, v in test_probs.items()}}).to_parquet(run_.path / "test_predictions.parquet", index=False)
    (run_.path / "model_card.md").write_text(_card(run_, results, selected, importance, ytr, yte), encoding="utf-8")
    registry.finalize(run_)
    log(f"Selected {selected}. Saved {run_.path}")
    return run_


def _card(run_, res, selected, imp, ytr, yte) -> str:
    rows = "\n".join(f"| {m} | {r['val']['pr_auc']:.3f} | {r['test']['pr_auc']:.3f} | {r['test']['roc_auc']:.3f} | "
                     f"{r['test']['brier']:.4f} | {r['test']['precision']:.2f} | {r['test']['recall']:.2f} | {r['test']['f1']:.2f} |"
                     for m, r in res.items())
    top = "\n".join(f"- {r.feature}: PR-AUC drop {r.pr_auc_drop:+.3f}" for r in imp.head(6).itertuples())
    return f"""# Model card — Supplier disruption-risk classifier

**Version:** `{run_.version}` · **Selected:** {selected} (best validation PR-AUC)
**Task:** probability that a disruption affecting a supplier starts within the next {HORIZON} days.
**Labels:** synthetic, from the generator's event log — this is synthetic supervised learning, not validated on real outcomes.
**Class balance:** {ytr.mean():.1%} positive in training, {yte.mean():.1%} in test.

| Model | Val PR-AUC | Test PR-AUC | Test ROC-AUC | Test Brier | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
{rows}

Precision / recall / F1 use each model's threshold chosen on validation (best F1). PR-AUC is the headline
metric because positives are rare; a random model's PR-AUC equals the positive rate ({yte.mean():.3f}).

## What drives the selected model (permutation importance on test, associational)
{top}

## Limitations
- Weather events are random in the generator and cannot be predicted from supplier data; part of the
  positive class is therefore unpredictable by design.
- Feature importance shows association with predictions, not causes of disruptions.
"""


if __name__ == "__main__":
    run()
