import plotly.graph_objects as go
import streamlit as st
from lib import data, ui
from lib.agents import results

ctx = data.ctx()
res = results()
k = res.kpis
meta = data.snapshot_meta()

ui.header("Executive control tower", f"Network status on {k['date']}",
          "What is happening, what is at risk and what to do, computed live from the trained models (ONNX) and the digital twin.")

pos = ctx.positions
at_risk = (pos.stockout_probability_14d >= 0.5).sum()
ui.kpis([
    ("Fill rate today", f"{k['fill_rate_pct']:.1f}%", f"28-day average {k['fill_rate_28d_avg']:.1f}%", "down" if k["fill_rate_pct"] < k["fill_rate_28d_avg"] else "up"),
    ("On-time in-full", f"{k['otif_pct']:.1f}%", None),
    ("Inventory value", ui.money(k["inventory_value_usd"]), None),
    ("Active disruptions", f"{k['active_disruptions']}", "recorded events", "down" if k["active_disruptions"] else "flat"),
    ("Series at stockout risk", f"{at_risk} / 400", "≥50% chance within 14 days", "down" if at_risk else "up"),
])
st.write("")
c1, c2 = st.columns([1.5, 1])
with c1:
    d = data.daily_network().loc["2025-07-01":]
    fig = go.Figure()
    fig.add_scatter(x=d.index, y=d.fill_rate_pct.rolling(7, min_periods=1).mean(), name="Fill rate", line=dict(color=ui.NAVY, width=2.5))
    fig.add_scatter(x=d.index, y=d.otif_pct.rolling(7, min_periods=1).mean(), name="OTIF", line=dict(color=ui.TEAL, width=2))
    fig.add_bar(x=d.index, y=d.active_disruptions, name="Active disruptions", marker_color=ui.CORAL, opacity=0.3, yaxis="y2")
    fig.update_layout(title="Service KPIs, last six months (7-day averages)", height=360,
                      yaxis=dict(title="%"), yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Disruptions"))
    st.plotly_chart(fig, width="stretch")
with c2:
    st.markdown("#### Executive summary")
    for line in res.executive_summary:
        st.markdown(f"- {line}")
    try:
        st.page_link("views/decision_room.py", label="Open the AI decision room →", icon=":material/gavel:")
    except Exception:  # page links only resolve inside the full app's navigation
        pass

t1, t2, t3 = st.tabs(["Stockout risk (next 14 days)", "Supplier watch-list", "Anomalies to investigate"])
with t1:
    top = pos.sort_values("expected_lost_revenue_14d", ascending=False).head(15)
    st.dataframe(top[["sku_id", "warehouse_id", "category", "on_hand", "receipts_14d", "demand_p50_14d", "days_of_supply",
                      "stockout_probability_14d", "expected_lost_revenue_14d"]],
                 column_config={"stockout_probability_14d": st.column_config.ProgressColumn("Stockout probability", format="percent", min_value=0, max_value=1),
                                "expected_lost_revenue_14d": st.column_config.NumberColumn("Expected lost revenue", format="$%.0f"),
                                "demand_p50_14d": st.column_config.NumberColumn("Forecast 14d (P50)", format="%.0f"),
                                "days_of_supply": st.column_config.NumberColumn("Days of supply", format="%.1f")},
                 hide_index=True, width="stretch")
    st.caption(f"Monte Carlo over the TFT forecast range ({meta['mc_samples']:,} samples) and scheduled receipts.")
with t2:
    rs = ctx.risk_scores.sort_values("probability", ascending=False).assign(active_disruption=lambda d: d.own_active_event > 0)
    cols = ["supplier_id", "name", "country", "category", "probability", "active_disruption"]
    st.dataframe(rs[cols].head(15), column_config={"probability": st.column_config.ProgressColumn("14-day disruption probability", format="%.3f", min_value=0, max_value=0.3)},
                 hide_index=True, width="stretch")
    st.caption(f"Model: {meta['risk_model']} (alert threshold {meta['risk_threshold']:.3f}). Suppliers with an active recorded disruption are handled regardless of the score.")
with t3:
    al = ctx.anomaly_alerts
    recent = al[al.date >= al.date.max() - __import__("pandas").Timedelta(days=13)] if len(al) else al
    st.dataframe(recent.sort_values("score", ascending=False).head(15), hide_index=True, width="stretch")
    st.caption("Autoencoder alerts from the last 14 days. Alerts are leads to investigate, never treated as confirmed events.")

ui.footer()
