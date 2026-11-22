import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from lib import data, ui

ui.header("Supply network", "Who supplies what, and where the risk sits",
          "50 suppliers in 12 countries feed 100 products stocked across 12 warehouses.")

sup = data.table("suppliers")
wh = data.table("warehouses")
prod = data.table("products")
sw = data.table("sku_warehouse")
rs = data.ctx().risk_scores.set_index("supplier_id")
sup = sup.join(rs[["probability"]], on="supplier_id")
links = prod[["sku_id", "primary_supplier_id"]].merge(sw[["sku_id", "warehouse_id"]], on="sku_id").groupby(["primary_supplier_id", "warehouse_id"]).size().reset_index(name="skus")

t1, t2 = st.tabs(["Network map", "Supplier dependency"])
with t1:
    show_links = st.toggle("Show supply links", value=True)
    fig = go.Figure()
    if show_links:
        s = sup.set_index("supplier_id")
        w = wh.set_index("warehouse_id")
        for _, lk in links.iterrows():
            a, b = s.loc[lk.primary_supplier_id], w.loc[lk.warehouse_id]
            fig.add_trace(go.Scattergeo(lon=[a.lng, b.lng], lat=[a.lat, b.lat], mode="lines", line=dict(width=0.6, color="rgba(31,42,68,.18)"),
                                        hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scattergeo(lon=sup.lng, lat=sup.lat, mode="markers", name="Suppliers",
                                marker=dict(size=(sup.annual_spend_usd / sup.annual_spend_usd.max() * 22 + 6), color=sup.probability, colorscale=[[0, ui.TEAL], [0.5, ui.GOLD], [1, ui.CORAL]],
                                            cmin=0, cmax=0.12, colorbar=dict(title="14-day risk", x=1.0), line=dict(width=0.5, color="white")),
                                text=sup.supplier_id + " · " + sup.name + "<br>" + sup.country + " · risk " + sup.probability.round(3).astype(str), hoverinfo="text"))
    fig.add_trace(go.Scattergeo(lon=wh.lng, lat=wh.lat, mode="markers", name="Warehouses", marker=dict(size=13, symbol="square", color=ui.NAVY, line=dict(width=1, color="white")),
                                text=wh.warehouse_id + " · " + wh.city, hoverinfo="text"))
    fig.update_geos(projection_type="natural earth", showcountries=True, countrycolor="#D9D3C7", landcolor="#F6F1E8", oceancolor="#FFFFFF", showocean=True, coastlinecolor="#D9D3C7")
    fig.update_layout(height=560, title="Suppliers (size = annual spend, colour = predicted risk) and warehouses", legend=dict(y=-0.05))
    st.plotly_chart(fig, width="stretch")

with t2:
    rev = data.table("demand_daily")
    rev = rev[rev.date >= "2025-10-01"].groupby("sku_id").demand_units.sum().rename("q4_units").reset_index().merge(prod, on="sku_id")
    rev["q4_revenue"] = rev.q4_units * rev.list_price
    dep = rev.groupby("primary_supplier_id").agg(products=("sku_id", "count"), q4_revenue=("q4_revenue", "sum")).join(sup.set_index("supplier_id")[["name", "country", "probability"]])
    dep["revenue_at_risk_14d"] = dep.q4_revenue / 92 * 14 * dep.probability
    dep = dep.sort_values("q4_revenue", ascending=False)
    fig = go.Figure(go.Scatter(x=dep.probability, y=dep.q4_revenue, mode="markers+text", text=dep.index, textposition="top center",
                               marker=dict(size=dep.products * 4 + 6, color=ui.NAVY, opacity=0.7)))
    fig.update_layout(title="Exposure: supplier risk vs revenue that depends on it (bubble = number of products)", height=460,
                      xaxis_title="14-day disruption probability", yaxis_title="Q4 2025 revenue of dependent products ($)")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(dep.reset_index().rename(columns={"primary_supplier_id": "supplier_id"}),
                 column_config={"q4_revenue": st.column_config.NumberColumn("Q4 revenue depending on it", format="$%.0f"),
                                "revenue_at_risk_14d": st.column_config.NumberColumn("Expected revenue at risk, 14 days", format="$%.0f"),
                                "probability": st.column_config.NumberColumn("Risk", format="%.3f")}, hide_index=True, width="stretch")
    st.caption("Revenue at risk = average daily revenue of dependent products × 14 days × disruption probability (a simple exposure estimate).")
    _ = pd

ui.footer()
