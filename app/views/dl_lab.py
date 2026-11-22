import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from lib import data, ui

ui.header("Deep learning lab", "Four neural networks, how they work and how they trained",
          "All four were built and trained in PyTorch on a laptop CPU, then exported to ONNX for fast serving.")

tft_c, lstm_c = data.art_json("forecast_tft", "config.json"), data.art_json("forecast_lstm", "config.json")
ae_c, risk_c = data.art_json("anomaly_autoencoder", "config.json"), data.art_json("risk_classifier", "config.json")
ui.kpis([("Temporal Fusion Transformer", f"{tft_c['parameters']:,}", "parameters · demand forecasting"),
         ("Seq2seq LSTM", f"{lstm_c['parameters']:,}", "parameters · demand forecasting"),
         ("Autoencoder", f"{ae_c.get('parameters', 8384):,}", "parameters · anomaly detection"),
         ("Neural network (MLP)", f"{risk_c.get('parameters', 3393):,}", "parameters · supplier risk")])
st.write("")


def curve(model: str, title: str, best: int | None = None):
    lc = data.art_csv(model, "loss_curve.csv")
    tr = "train_loss" if "train_loss" in lc else lc.columns[1]
    va = "val_loss" if "val_loss" in lc else (lc.columns[2] if lc.shape[1] > 2 else None)
    fig = go.Figure()
    fig.add_scatter(x=lc.epoch, y=lc[tr], name="Training loss", mode="lines+markers", line=dict(color=ui.NAVY))
    if va:
        fig.add_scatter(x=lc.epoch, y=lc[va], name="Validation loss", mode="lines+markers", line=dict(color=ui.CORAL))
    if best:
        fig.add_vline(x=best, line_dash="dot", annotation_text=f"best epoch {best}")
    fig.update_layout(title=title, height=340, xaxis_title="Epoch")
    st.plotly_chart(fig, width="stretch")


t1, t2, t3, t4, t5 = st.tabs(["Temporal Fusion Transformer", "LSTM", "Autoencoder", "Risk neural network", "Serving (ONNX)"])

with t1:
    c1, c2 = st.columns([1, 1.25])
    with c1:
        st.markdown("#### Why the TFT")
        st.markdown("The strongest demand drivers here (promotions, prices, Black Friday, Cyber Monday) are **known in advance**. "
                    "The TFT feeds them straight into its decoder, shares learning across all 400 series through static context, "
                    "and outputs a **range** (P10, P50, P90) plus **explanations**.")
        for n, (t, b) in enumerate([("Variable selection networks", "learn how much each input matters at each step"),
                                    ("Static covariate encoders", "turn SKU, warehouse, category and size into context vectors"),
                                    ("LSTM encoder–decoder", "reads 56 past days and 28 known future days"),
                                    ("Interpretable attention (4 heads)", "lets each future day look back at any earlier day"),
                                    ("Quantile head", "P10 ≤ P50 ≤ P90 for each of 28 days, trained with pinball loss")]):
            st.markdown(f'<div class="step"><div class="n">{n + 1}</div><div><b>{t}</b>: {b}</div></div>', unsafe_allow_html=True)
    with c2:
        curve("forecast_tft", "TFT training: pinball loss by epoch", tft_c.get("best_epoch"))
        h = tft_c["hyperparameters"]
        st.caption(f"AdamW, learning rate {h['lr']}, batch {h['batch_size']}, {h['windows_per_epoch']:,} windows per epoch "
                   f"(from {tft_c['train_windows_available']:,}), early stopping; best epoch {tft_c['best_epoch']} of {tft_c['epochs_run']}, "
                   f"{tft_c['train_seconds'] / 60:.0f} minutes on CPU.")
    st.markdown("#### What the TFT learned to look at")
    it = data.art_json("forecast_tft", "interpretation.json")
    cols = st.columns(3)
    for c, (key, title, col) in zip(cols, [("static_variables", "Static inputs", ui.GREY), ("past_variables", "Past inputs", ui.TEAL), ("future_variables", "Known-future inputs", ui.NAVY)], strict=False):
        s = pd.Series(it[key]).sort_values()
        c.plotly_chart(go.Figure(go.Bar(x=s.values * 100, y=s.index.str.replace("_", " "), orientation="h", marker_color=col,
                                        text=[f"{v * 100:.0f}%" for v in s.values], textposition="outside"))
                       .update_layout(title=title, height=360, xaxis_title="Weight (%)", xaxis_range=[0, s.max() * 130]), width="stretch")
    att = np.array(it["attention_over_past_days"])
    lags = np.arange(-len(att), 0)
    st.plotly_chart(go.Figure(go.Bar(x=lags, y=att * 100, marker_color=[ui.CORAL if lag % 7 == 0 else ui.NAVY for lag in lags]))
                    .update_layout(title="Attention over the 56-day history (coral = same weekday)", height=300, xaxis_title="Days before forecast origin", yaxis_title="%"),
                    width="stretch")

with t2:
    c1, c2 = st.columns([1, 1.25])
    with c1:
        st.markdown("#### Sequence-to-sequence LSTM")
        st.markdown("An **encoder LSTM** reads the 56-day history; its final state, initialised from product embeddings, starts a "
                    "**decoder LSTM** that reads the 28 known-future days; a small two-layer head outputs P10, P50 and P90.\n\n"
                    "Same inputs and loss as the TFT, but **no variable selection or attention**, so comparing the two isolates the value of those parts.")
        st.metric("Test WAPE", ui.pct(data.headline()["cmp"]["overall"]["LSTM (PyTorch)"]["test"]["wape"]),
                  f"TFT {ui.pct(data.headline()['cmp']['overall']['TFT (PyTorch)']['test']['wape'])}", delta_color="off")
    with c2:
        curve("forecast_lstm", "LSTM training: pinball loss by epoch", lstm_c.get("best_epoch"))

with t3:
    c1, c2 = st.columns([1, 1.25])
    with c1:
        st.markdown("#### Autoencoder for anomalies")
        st.markdown("A symmetric network compresses **7 days × 8 operational signals (56 inputs)** through 64 units to an **8-unit bottleneck** and rebuilds them. "
                    "It learns what normal looks like; windows it cannot rebuild are unusual.\n\n"
                    "Only signals that context cannot explain are scored (demand z-score, unexplained stock change, receipt delay), "
                    "so promotions and holidays are not flagged. The first version without this flagged 859 alerts at 1% precision.")
    with c2:
        curve("anomaly_autoencoder", "Autoencoder training: reconstruction loss")

with t4:
    c1, c2 = st.columns([1, 1.25])
    with c1:
        st.markdown("#### Neural network for supplier risk")
        st.markdown("A multilayer perceptron (**19 → 64 → 32 → 1**, ReLU, dropout 0.2) reads 19 trailing signals per supplier per day "
                    "and outputs the probability of a supplier-originated disruption in the next **14 days**. Trained with binary "
                    "cross-entropy; the alert threshold is chosen on validation data because positives are rare (2.7%).")
        st.warning("On the test quarter, simpler models beat it (see **Supplier risk**). We report this rather than hide it.")
    with c2:
        curve("risk_classifier", "Risk network training: binary cross-entropy")

with t5:
    meta = data.snapshot_meta()
    st.markdown("Trained networks are exported to **ONNX** and run with **ONNX Runtime**, so this app needs no PyTorch and runs on a free server.")
    rows = []
    for m, name in [("forecast_tft", "TFT"), ("forecast_lstm", "LSTM")]:
        on = data.art_json(m, "metrics.json").get("onnx", {})
        rows.append({"Model": name, "Max difference vs PyTorch": on.get("max_abs_diff_vs_pytorch"), "Latency, all 400 series (ms)": on.get("onnxruntime_latency_ms_400_series"),
                     "File size (KB)": on.get("file_kb"),
                     "Version": data.art_version(m)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if st.button("Run the TFT live on all 400 series now", type="primary"):
        fc = data.forecaster("forecast_tft")
        q = fc.predict([len(fc.panel.dates)])
        st.success(f"Forecast {q.shape[1]} series × {q.shape[2]} days × 3 quantiles in {fc.last_latency_ms:.0f} ms on this server.")
    st.caption(f"Snapshot forecast used across the app: {meta['forecast_model']} {meta['forecast_version']}.")

ui.footer()
