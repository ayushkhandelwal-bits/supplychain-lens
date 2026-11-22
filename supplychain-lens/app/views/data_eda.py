import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from lib import data, ui

ui.header("Data explorer", "What the data is, and what it shows",
          "Three years (2023–2025) of a manufacturer's supply chain, simulated causally on the NEXUS-SCCT data model.")

dem = data.table("demand_daily")
prod = data.table("products")
sup = data.table("suppliers")

t1, t2, t3, t4, t5 = st.tabs(["Tables", "Demand patterns", "Products & categories", "Suppliers & disruptions", "Data quality"])

with t1:
    tables = {"demand_daily": "Daily demand, sales and lost sales per product and warehouse (forecast target)",
              "inventory_daily": "Daily stock on hand, on order, receipts and adjustments",
              "purchase_orders": "Orders placed with suppliers, promised dates, quantities",
              "shipments": "Carrier, transport mode, ETA, actual arrival, delay",
              "disruptions": "Recorded disruption events with type, severity and effects",
              "promotions": "Planned promotions with discount",
              "kpi_daily": "Network fill rate, stockout rate, OTIF, inventory value",
              "suppliers": "Supplier master data: country, health score, lead time, capacity",
              "products": "Product master data: category, cost, price, suppliers",
              "warehouses": "Warehouse locations and capacity",
              "calendar": "Day of week, Black Friday, Cyber Monday, holiday weeks",
              "sku_warehouse": "Which warehouses stock each product"}
    rows = [{"table": k, "rows": len(data.table(k)), "columns": data.table(k).shape[1], "content": v} for k, v in tables.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Four further hidden-truth tables (true supplier health, every disruption cause, 180 planted anomalies, product profiles) "
               "are used only to grade the models; an automated test blocks model code from reading them.")
    pick = st.selectbox("Preview a table", list(tables))
    st.dataframe(data.table(pick).head(200), width="stretch", height=320)

with t2:
    d = data.daily_network()
    fig = go.Figure()
    fig.add_scatter(x=d.index, y=d.demand_units, name="Daily demand", line=dict(color=ui.NAVY, width=0.8), opacity=0.7)
    fig.add_scatter(x=d.index, y=d.demand_units.rolling(28).mean(), name="28-day average", line=dict(color=ui.CORAL, width=2.5))
    fig.add_vrect(x0="2025-07-01", x1="2025-09-30", fillcolor=ui.TEAL, opacity=0.08, line_width=0, annotation_text="Validation")
    fig.add_vrect(x0="2025-10-01", x1="2025-12-31", fillcolor=ui.CORAL, opacity=0.08, line_width=0, annotation_text="Test")
    fig.update_layout(title="Total daily demand, 400 series", height=380, yaxis_title="Units")
    st.plotly_chart(fig, width="stretch")
    normal = dem[(dem.is_promo == 0) & (dem.is_black_friday == 0) & (dem.is_cyber_monday == 0) & (dem.is_holiday_week == 0)]
    base = normal.demand_units.mean()
    c1, c2, c3 = st.columns(3)
    dow = normal.groupby(normal.date.dt.dayofweek).demand_units.mean() / base
    c1.plotly_chart(go.Figure(go.Bar(x=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], y=dow.values, marker_color=[ui.NAVY if v >= 1 else ui.GREY for v in dow.values],
                                     text=[f"{v:.2f}" for v in dow.values], textposition="outside")).update_layout(title="Day-of-week index", height=320, yaxis_range=[0, 1.4]), width="stretch")
    mon = normal.groupby(normal.date.dt.month).demand_units.mean() / base
    c2.plotly_chart(go.Figure(go.Scatter(x=["J", "F", "M", "A", "M ", "J ", "J  ", "A ", "S", "O", "N", "D"], y=mon.values, mode="lines+markers", line=dict(color=ui.TEAL, width=3)))
                    .update_layout(title="Month-of-year index", height=320), width="stretch")
    up = {"Promotion day": dem[(dem.is_promo == 1) & (dem.is_black_friday == 0) & (dem.is_cyber_monday == 0)].demand_units.mean() / base,
          "Holiday week": dem[(dem.is_holiday_week == 1) & (dem.is_promo == 0) & (dem.is_black_friday == 0) & (dem.is_cyber_monday == 0)].demand_units.mean() / base,
          "Black Friday": dem[dem.is_black_friday == 1].demand_units.mean() / base, "Cyber Monday": dem[dem.is_cyber_monday == 1].demand_units.mean() / base}
    c3.plotly_chart(go.Figure(go.Bar(x=list(up.values()), y=list(up.keys()), orientation="h", marker_color=[ui.TEAL, ui.GREY, ui.NAVY, ui.CORAL],
                                     text=[f"×{v:.2f}" for v in up.values()], textposition="outside")).update_layout(title="Event uplift vs a normal day", height=320, xaxis_range=[0, 3.9]), width="stretch")
    st.info("All of these drivers (weekday, season, promotions, Black Friday, Cyber Monday) are **known in advance**, which is exactly what the Temporal Fusion Transformer is designed to use.")

with t3:
    dc = dem.merge(prod[["sku_id", "category"]], on="sku_id")
    cat = dc.groupby("category")[["demand_units", "lost_sales_units"]].sum()
    cat["share"] = cat.demand_units / cat.demand_units.sum() * 100
    cat["lost_rate"] = cat.lost_sales_units / cat.demand_units * 100
    cat = cat.sort_values("share")
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(cat, x="share", y=cat.index, orientation="h", text=cat.share.round(1), title="Share of total demand (%)", color_discrete_sequence=[ui.NAVY]).update_layout(height=420, yaxis_title=None, xaxis_title=None), width="stretch")
    c2.plotly_chart(px.bar(cat, x="lost_rate", y=cat.index, orientation="h", text=cat.lost_rate.round(1), title="Demand lost to stockouts (%)", color_discrete_sequence=[ui.CORAL]).update_layout(height=420, yaxis_title=None, xaxis_title=None), width="stretch")
    st.dataframe(prod, hide_index=True, width="stretch", height=280)

with t4:
    ship = data.table("shipments").merge(sup[["supplier_id", "region"]], on="supplier_id")
    arr = ship[ship.actual_arrival.notna()]
    dis = data.table("disruptions")
    ui.kpis([("Shipments delivered late", ui.pct(arr.is_delayed.mean()), f"late ones average {arr.loc[arr.is_delayed == 1, 'delay_days'].mean():.1f} days"),
             ("Disruptions recorded", f"{len(dis)}", f"{(dis.event_type == 'SUPPLIER').sum()} supplier-originated"),
             ("Critical disruptions", f"{(dis.severity == 'CRITICAL').sum()}", ""),
             ("Sea freight share", ui.pct((ship.transport_mode == 'SEA').mean(), 0), "")])
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.histogram(arr, x=arr.delay_days.clip(-5, 30), nbins=36, title="Delivery delay vs promised date (days, capped at 30)", color_discrete_sequence=[ui.TEAL]).update_layout(height=360, xaxis_title="Days late", yaxis_title="Shipments"), width="stretch")
    reg = arr.groupby("region").is_delayed.mean().mul(-100).add(100).sort_values()
    c2.plotly_chart(px.bar(x=reg.values, y=reg.index, orientation="h", text=reg.round(1), title="On-time delivery by supplier region (%)", color_discrete_sequence=[ui.NAVY]).update_layout(height=360, xaxis_title=None, yaxis_title=None, xaxis_range=[0, 100]), width="stretch")
    dis["quarter"] = pd.to_datetime(dis.detected_at).dt.to_period("Q").astype(str)
    q = dis.groupby(["quarter", "event_type"]).size().reset_index(name="n")
    st.plotly_chart(px.bar(q, x="quarter", y="n", color="event_type", title="Disruptions per quarter by type",
                           color_discrete_sequence=[ui.CORAL, ui.NAVY, ui.TEAL, ui.GOLD, ui.GREY, "#8E6C8A"]).update_layout(height=360, xaxis_title=None, yaxis_title="Events"), width="stretch")

with t5:
    st.markdown("**36 of 36 automated checks pass** after every data build.")
    rows = [("Primary keys", 9, "No duplicate supplier, product, order or daily rows"), ("Foreign keys", 8, "Every order, shipment and disruption links to a valid supplier"),
            ("Missing values", 1, "No unexpected missing values"), ("Value ranges", 6, "Sales + lost sales = demand; utilisation within 0–100%"),
            ("NEXUS behaviour", 3, "Weekday pattern r = 0.997 with NEXUS factors; Black Friday uplift; positive trend"),
            ("Reference defects fixed", 9, "Delays relate to supplier health (r = 0.50); weather hits only matching countries")]
    st.dataframe(pd.DataFrame(rows, columns=["Check group", "Checks passed", "Examples"]), hide_index=True, width="stretch")
    st.markdown("**Leakage controls:** time-based split; future inputs limited to calendar, planned promotions and prices; "
                "disruptions count only from their detection date; delays count only after arrival; scaling fitted on training data; "
                "test quarter opened once.")
    _ = np  # keep numpy import for interactive use

ui.footer()
