import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from lib import data, ui

ui.header("Anomaly detection", "Unusual operations, flagged for investigation",
          "The autoencoder learns what normal looks like for each product and warehouse and scores how far each day is from it.")

al = data.ctx().anomaly_alerts.copy()
m = data.art_json("anomaly_autoencoder", "metrics.json")["models"]
A, Z = m["Autoencoder (PyTorch)"], m["Robust z-score"]

ui.kpis([("Alerts in the test quarter", f"{len(al)}", "Oct–Dec 2025"),
         ("Alert threshold", f"{A['test']['threshold']:.2f}", "chosen on validation"),
         ("Alerts linked to a real problem", ui.pct(A["test"]["precision_incl_recorded_events"], 0), "planted or recorded event"),
         ("Planted events caught", f"{A['test']['events_detected']} of {A['test']['events_in_period']}", None)])
st.write("")
t1, t2, t3 = st.tabs(["Alert list", "Investigate one alert", "Autoencoder vs z-score"])

with t1:
    reasons = {"receipt_delay": "Receipts arriving much later than promised", "unexplained_inv": "Stock changed without a sale or receipt",
               "demand_z": "Demand far from normal for this context"}
    al["likely cause"] = al.top_feature_1.map(reasons).fillna(al.top_feature_1)
    al["matched a known problem"] = al.eval_matches_injected | al.eval_during_recorded_event
    cats = st.multiselect("Category", sorted(al.category.unique()), default=sorted(al.category.unique()))
    v = al[al.category.isin(cats)].sort_values("score", ascending=False)
    st.dataframe(v[["date", "sku_id", "warehouse_id", "category", "supplier_id", "score", "likely cause", "matched a known problem"]],
                 column_config={"score": st.column_config.ProgressColumn("Anomaly score", format="%.1f", min_value=0, max_value=float(al.score.max()))},
                 hide_index=True, width="stretch", height=420)
    st.caption("'Matched a known problem' uses the hidden answer key (planted anomalies and recorded disruptions); it is shown only to grade the model.")
    by = al.groupby(al.date.dt.to_period("W").dt.start_time).size()
    st.plotly_chart(go.Figure(go.Bar(x=by.index, y=by.values, marker_color=ui.CORAL)).update_layout(title="Alerts per week", height=280), width="stretch")

with t2:
    lab = al.sort_values("score", ascending=False).apply(lambda r: f"{r.date:%d %b} · {r.sku_id} @ {r.warehouse_id} · score {r.score:.1f}", axis=1)
    pick = st.selectbox("Alert", lab.index, format_func=lambda i: lab[i])
    r = al.loc[pick]
    dem = data.table("demand_daily")
    inv = data.table("inventory_daily")
    w0, w1 = r.date - pd.Timedelta(days=45), r.date + pd.Timedelta(days=15)
    d = dem[(dem.sku_id == r.sku_id) & (dem.warehouse_id == r.warehouse_id) & dem.date.between(w0, w1)]
    i = inv[(inv.sku_id == r.sku_id) & (inv.warehouse_id == r.warehouse_id) & inv.date.between(w0, w1)]
    fig = go.Figure()
    fig.add_scatter(x=d.date, y=d.demand_units, name="Demand", line=dict(color=ui.NAVY))
    fig.add_scatter(x=i.date, y=i.on_hand_units, name="Stock on hand", line=dict(color=ui.TEAL), yaxis="y2")
    fig.add_bar(x=i.date, y=i.receipts_units, name="Receipts", marker_color=ui.GOLD, opacity=0.5, yaxis="y2")
    fig.add_vline(x=r.date, line_color=ui.CORAL, line_width=2, annotation_text="alert")
    fig.update_layout(title=f"{r.sku_id} at {r.warehouse_id} around the alert", height=400, yaxis=dict(title="Demand"),
                      yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Stock / receipts"))
    st.plotly_chart(fig, width="stretch")
    errs = {c[4:]: r[c] for c in al.columns if c.startswith("err_")}
    e = pd.Series(errs).sort_values()
    st.plotly_chart(go.Figure(go.Bar(x=e.values, y=e.index.str.replace("_", " "), orientation="h", marker_color=[ui.CORAL if k in ("demand_z", "unexplained_inv", "receipt_delay") else ui.GREY for k in e.index]))
                    .update_layout(title="Reconstruction error by signal (coral = scored signals)", height=320), width="stretch")

with t3:
    rows = []
    for name, x in [("Autoencoder", A), ("Robust z-score", Z)]:
        for setting, key in [("own validation threshold", "test"), ("100 alerts each", "test_matched_alerts")]:
            s = x[key]
            rows.append({"Model": name, "Setting": setting, "Alerts": s["alerts"], "Precision (planted)": s["precision_injected"],
                         "Precision (incl. recorded events)": s["precision_incl_recorded_events"], "Planted events caught": f"{s['events_detected']} of {s['events_in_period']}"})
    st.dataframe(pd.DataFrame(rows), column_config={"Precision (planted)": st.column_config.NumberColumn(format="%.3f"),
                 "Precision (incl. recorded events)": st.column_config.NumberColumn(format="%.3f")}, hide_index=True, width="stretch")
    ui.card("Verdict: a draw", "With the same 100-alert budget the autoencoder is twice as precise on planted anomalies (10% vs 5%) and catches one more event. "
            "But a z-score at its own threshold raises only 23 alerts at 22% precision. Choose by how many alerts the team can investigate.", ui.GOLD)

ui.footer()
