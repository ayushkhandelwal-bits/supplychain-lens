import plotly.graph_objects as go
import streamlit as st
from lib import data, ui

h = data.headline()
cmp, imp = h["cmp"], h["imp"]
ov, M = cmp["overall"], imp["metrics"]

ui.hero("Deep Learning Project · BITS Pilani", "Supply Chain Lens",
        "Deep learning that forecasts demand, flags supplier risk and spots anomalies, then turns those predictions "
        "into decisions a manager can approve. Built on the open-source NEXUS-SCCT supply chain model.")

ui.kpis([
    ("TFT forecast error (WAPE)", ui.pct(ov["TFT (PyTorch)"]["test"]["wape"]), f"vs {ui.pct(ov['Seasonal naive']['test']['wape'])} for the old-style rule", "up"),
    ("Fill rate, Q4 2025 replay", ui.pct(M["fill_rate"]["ai_mean"]), f"from {ui.pct(M['fill_rate']['baseline_mean'])} (+{M['fill_rate']['diff'] * 100:.1f} pts)", "up"),
    ("Total supply chain cost", f"{M['total_cost']['relative_change'] * 100:+.1f}%", "simulated, p < 0.001", "up"),
    ("Lost revenue", f"{M['lost_revenue']['relative_change'] * 100:+.0f}%", "simulated", "up"),
    ("Budget", "₹0", "open-source tools, laptop CPU", "flat"),
])
st.write("")

t1, t2, t3, t4, t5 = st.tabs(["① Business problem", "② Objectives", "③ The data", "④ What we built", "⑤ Outcomes"])

with t1:
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("### Too little stock loses sales; too much ties up cash")
        st.markdown(
            "- **Long lead times.** 50 suppliers in 12 countries take about **20 to 100 days** to deliver, so today's order must match demand weeks ahead.\n"
            "- **Spiky demand.** Black Friday and Cyber Monday roughly **triple** daily demand; promotions double it.\n"
            "- **Fragile suppliers.** Weather, port congestion and supplier failures delay or cut deliveries.\n"
            "- **A blind rule.** The company orders the last 28 days' average plus 15% safety stock. It cannot see a festival coming or a supplier weakening.")
    with c2:
        d = data.daily_network()
        q4 = d.loc["2025-10-01":]
        fig = go.Figure()
        fig.add_scatter(x=q4.index, y=q4.fill_rate_pct.rolling(7, min_periods=1).mean(), name="Fill rate (7-day avg)", line=dict(color=ui.NAVY, width=2.5))
        fig.add_vrect(x0="2025-11-28", x1="2025-12-01", fillcolor=ui.CORAL, opacity=0.15, line_width=0, annotation_text="BF–CM")
        fig.update_layout(title="Recorded fill rate under the old rule, Q4 2025", height=330, yaxis_title="%", showlegend=False)
        st.plotly_chart(fig, width="stretch")
    ui.card("The question we asked",
            "Can deep learning forecast demand and supplier risk better than established methods, and does that accuracy "
            "turn into measurably better service and lower cost when it drives ordering decisions?", ui.CORAL)

with t2:
    cols = st.columns(4)
    objs = [("Forecast demand", "Next 28 days for all 400 product–warehouse series, as low (P10), likely (P50) and high (P90) values.", ui.CORAL),
            ("Flag supplier risk", "Probability that each supplier is disrupted in the next 14 days.", ui.GOLD),
            ("Catch anomalies", "Unusual demand, stock or delivery patterns, before they escalate.", ui.TEAL),
            ("Prove the value", "Beat simpler methods on unseen data, then measure fill rate, stockouts and cost.", ui.NAVY)]
    for c, (t, b, col) in zip(cols, objs, strict=False):
        with c:
            ui.card(t, b, col)
    ui.card("Success test", "A model is adopted only if it beats simpler methods on data it has never seen: October to December 2025, "
            "the quarter that contains Black Friday and Cyber Monday.", ui.GOLD)

with t3:
    c1, c2 = st.columns([1, 1.15])
    with c1:
        st.markdown("### One causal, day-by-day simulation")
        st.markdown(
            "Real supplier and inventory data is confidential, so we generated a realistic dataset on the data model of "
            "**NEXUS-SCCT** (open source, by Syed Zain Ahmed): same countries, categories, disruption types and seasonal factors.\n\n"
            "Each supplier has a **hidden health** score. A weak supplier is more likely to be disrupted, deliver late and ship short; "
            "late deliveries reduce stock; low stock turns demand into lost sales. We also fixed **9 defects** in the reference generator.")
        st.markdown("**Split by time:** train Jan 2023 – Jun 2025 · validate Q3 2025 · test Q4 2025 (opened once).")
    with c2:
        ui.kpis([("Suppliers", "50", "12 countries"), ("Products", "100", "10 categories"), ("Warehouses", "12", "")])
        st.write("")
        ui.kpis([("Demand series", "400", "product × warehouse"), ("Days", "1,096", "2023–2025"), ("Daily records", "4.4 lakh", "")])
        st.write("")
        ui.kpis([("Purchase orders", "53,133", ""), ("Disruptions", "171", "113 supplier-originated"), ("Data checks", "36 / 36", "all pass", "up")])
    st.caption("Open **Data explorer** in the sidebar for the tables and charts.")

with t4:
    st.markdown("### From data to a decision a manager approves")
    steps = [("Data", "Generate and validate 18 tables; 36 automated checks."),
             ("Deep learning", "Temporal Fusion Transformer and LSTM (forecasts), autoencoder (anomalies), neural network (risk), all in PyTorch."),
             ("Honest comparison", "Every model faces simpler rivals (LightGBM, Holt-Winters, XGBoost, z-score…) on the unseen test quarter, with significance tests."),
             ("Serving", "Models exported to ONNX: all 400 forecasts in about 0.26 s, no PyTorch needed."),
             ("Decisions", "7 tools, 5 specialist agents and a coded approval policy; big or irreversible actions always go to a manager."),
             ("Business impact", "A digital twin replays Q4 2025 under the old rule and the AI policy, 30 times each.")]
    c1, c2 = st.columns(2)
    for i, (t, b) in enumerate(steps):
        (c1 if i < 3 else c2).markdown(f'<div class="step"><div class="n">{i + 1}</div><div><b>{t}.</b> {b}</div></div>', unsafe_allow_html=True)

with t5:
    c1, c2 = st.columns([1.2, 1])
    with c1:
        names = ["TFT (PyTorch)", "LSTM (PyTorch)", "LightGBM (quantile)", "Holt-Winters (ETS)", "Seasonal naive"]
        w = [ov[n]["test"]["wape"] * 100 for n in names]
        fig = go.Figure(go.Bar(x=w[::-1], y=[ui.SHORT[n] for n in names][::-1], orientation="h",
                               marker_color=[ui.MODEL_COLORS[ui.SHORT[n]] for n in names][::-1], text=[f"{v:.1f}%" for v in w[::-1]], textposition="outside"))
        fig.update_layout(title="Forecast error on the unseen test quarter (lower is better)", height=360, xaxis_title="WAPE (%)", xaxis_range=[0, 33])
        st.plotly_chart(fig, width="stretch")
    with c2:
        ui.card("Deep learning wins where data is rich",
                f"The TFT is the best forecaster ({ui.pct(ov['TFT (PyTorch)']['test']['wape'])}), with a significant lead that widens in Black Friday week "
                f"({ui.pct(cmp['peak_week_test']['TFT (PyTorch)']['wape'])} vs {ui.pct(cmp['peak_week_test']['LightGBM (quantile)']['wape'])} for LightGBM).", ui.NAVY)
        ui.card("…and not where it is scarce",
                "For rare supplier failures, XGBoost and logistic regression beat the neural network; the autoencoder only matches a tuned z-score.", ui.GOLD)
        ui.card("Business result " + ui.simulated(),
                f"Fill rate {ui.pct(M['fill_rate']['baseline_mean'])} → {ui.pct(M['fill_rate']['ai_mean'])}, lost revenue "
                f"{M['lost_revenue']['relative_change'] * 100:+.0f}%, total cost {M['total_cost']['relative_change'] * 100:+.1f}%, "
                f"inventory {M['avg_inventory_value']['relative_change'] * 100:+.0f}%.", ui.TEAL)

ui.footer()
