import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from lib import data, ui

ui.header("Demand intelligence", "28-day demand forecasts with a range",
          "Pick any product and warehouse to see the Temporal Fusion Transformer's forecast, or replay the test quarter and compare models.")

pn = data.panel()
keys = pn.keys.reset_index(drop=True)
dates = pn.dates
skus = sorted(keys.sku_id.unique())

c1, c2 = st.columns(2)
sku = c1.selectbox("Product", skus, index=skus.index("SKU-0057") if "SKU-0057" in skus else 0)
whs = keys[keys.sku_id == sku].warehouse_id.tolist()
wh = c2.selectbox("Warehouse", whs, index=whs.index("WH-008") if "WH-008" in whs else 0)
s_idx = int(keys[(keys.sku_id == sku) & (keys.warehouse_id == wh)].index[0])
prod = data.table("products").set_index("sku_id").loc[sku]
st.caption(f"{prod['name']} · {prod['category']} · primary supplier {prod['primary_supplier_id']} · price ${prod['list_price']:,.0f}")

t1, t2, t3 = st.tabs(["Forecast next 28 days (live)", "Replay the test quarter: model vs model", "Network outlook"])

with t1:
    fc = data.forecaster("forecast_tft")
    origin = len(dates)
    f = fc.forecast_frame(origin, np.array([s_idx]))
    hist = pd.Series(pn.y[s_idx, -84:], index=dates[-84:])
    fig = go.Figure()
    fig.add_scatter(x=hist.index, y=hist.values, name="History", line=dict(color=ui.GREY, width=1.6))
    fig.add_scatter(x=list(f.date) + list(f.date[::-1]), y=list(f.p90) + list(f.p10[::-1]), fill="toself", fillcolor="rgba(42,157,143,.18)",
                    line=dict(width=0), name="P10–P90 range")
    fig.add_scatter(x=f.date, y=f.p50, name="TFT forecast (P50)", line=dict(color=ui.TEAL, width=3))
    fig.add_vline(x=dates[-1], line_dash="dot")
    fig.update_layout(title=f"{sku} at {wh}: last 12 weeks and next 28 days", height=420, yaxis_title="Units per day")
    st.plotly_chart(fig, width="stretch")
    last28 = pn.y[s_idx, -28:].sum()
    ui.kpis([("Forecast next 28 days (P50)", f"{f.p50.sum():,.0f}", f"{(f.p50.sum() / last28 - 1) * 100:+.1f}% vs last 28 days", "up" if f.p50.sum() >= last28 else "down"),
             ("Low case (P10)", f"{f.p10.sum():,.0f}", None), ("High case (P90)", f"{f.p90.sum():,.0f}", None),
             ("Inference time", f"{fc.last_latency_ms:.0f} ms", "ONNX Runtime, this server")])
    st.caption("Future inputs after 31 Dec 2025 are the calendar only (no promotions are planned in the data beyond that date; prices assumed at list).")

with t2:
    origins = {"1 Oct 2025": 1004, "29 Oct 2025": 1032, "26 Nov 2025 (Black Friday window)": 1060}
    o_name = st.radio("Forecast made on", list(origins), index=2, horizontal=True)
    o = origins[o_name]
    preds = []
    for m in ["forecast_tft", "forecast_lstm", "forecast_baselines"]:
        p = data.art_parquet(m, "predictions.parquet")
        if "model" not in p:
            p = p.assign(model={"forecast_tft": "TFT (PyTorch)", "forecast_lstm": "LSTM (PyTorch)"}[m])
        preds.append(p[(p.series == s_idx) & (p.origin_idx == o)])
    P = pd.concat(preds)
    P["model"] = P.model.map(lambda x: ui.SHORT.get(x, x))
    show = st.multiselect("Models", ["TFT", "LSTM", "LightGBM", "Holt-Winters", "Seasonal naive"], default=["TFT", "LightGBM", "Holt-Winters"])
    fd = pd.Timestamp(dates[0]) + pd.to_timedelta(np.arange(o - 28, o + 28), unit="D")
    fig = go.Figure()
    fig.add_scatter(x=fd, y=pn.y[s_idx, o - 28: o + 28], name="Actual demand", line=dict(color="#111827", width=2.2))
    rows = []
    for m in show:
        d = P[P.model == m].sort_values("h")
        if d.empty:
            continue
        x = pd.Timestamp(dates[0]) + pd.to_timedelta(d.t_idx, unit="D")
        fig.add_scatter(x=x, y=d.p50, name=m, line=dict(color=ui.MODEL_COLORS[m], width=2.4 if m == "TFT" else 1.6, dash=None if m in ("TFT", "LSTM") else "dash"))
        rows.append({"Model": m, "WAPE (this series)": (d.y - d.p50).abs().sum() / d.y.sum(), "Forecast total": d.p50.sum(), "Actual total": d.y.sum()})
    fig.add_vline(x=pd.Timestamp(dates[0]) + pd.Timedelta(days=o), line_dash="dot", annotation_text="forecast made")
    fig.update_layout(title=f"{sku} at {wh}: forecasts vs what actually happened", height=420, yaxis_title="Units per day")
    st.plotly_chart(fig, width="stretch")
    if rows:
        st.dataframe(pd.DataFrame(rows).sort_values("WAPE (this series)"), column_config={"WAPE (this series)": st.column_config.NumberColumn(format="%.3f"),
                     "Forecast total": st.column_config.NumberColumn(format="%.0f"), "Actual total": st.column_config.NumberColumn(format="%.0f")},
                     hide_index=True, width="stretch")
    st.caption("One series can go either way; the overall comparison over all 400 series is on **Model evaluation**.")

with t3:
    ctx = data.ctx()
    fr = ctx.forecast.merge(keys.reset_index(names="s")[["sku_id", "warehouse_id", "s"]], on=["sku_id", "warehouse_id"])
    fr = fr.merge(data.table("products")[["sku_id", "category"]], on="sku_id") if "category" not in fr else fr
    nxt = fr.groupby("category").p50.sum()
    kc = keys.reset_index(names="s")
    last = pd.Series({c: pn.y[kc.loc[kc.category == c, "s"].to_numpy(), -28:].sum() for c in nxt.index})
    df = pd.DataFrame({"Last 28 days": last, "Next 28 days (P50)": nxt}).sort_values("Next 28 days (P50)")
    fig = go.Figure()
    fig.add_bar(y=df.index, x=df["Last 28 days"], name="Last 28 days", orientation="h", marker_color=ui.GREY)
    fig.add_bar(y=df.index, x=df["Next 28 days (P50)"], name="Next 28 days (P50)", orientation="h", marker_color=ui.NAVY)
    fig.update_layout(title="Demand by category: last 28 days vs forecast", height=440, barmode="group")
    st.plotly_chart(fig, width="stretch")
    tot = df.sum()
    st.metric("Network forecast, next 28 days", f"{tot['Next 28 days (P50)']:,.0f} units", f"{(tot['Next 28 days (P50)'] / tot['Last 28 days'] - 1) * 100:+.1f}% vs last 28 days")
    st.caption("The last 28 days include Black Friday, Cyber Monday and Christmas promotions, so a lower January is expected.")

ui.footer()
