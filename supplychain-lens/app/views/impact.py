import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from lib import data, ui

imp = data.art_json("business_impact", "results.json")
M, AB, cal = imp["metrics"], imp["ablation_no_predicted_risk_switching"], imp["twin_calibration"]

ui.header("Business impact", "What better forecasts are worth",
          "The old rule and the TFT-driven policy each replayed the real Oct–Dec 2025 demand through the digital twin, 30 times with identical supplier delays.")
st.markdown(ui.simulated("SIMULATED · synthetic data · digital-twin replay"), unsafe_allow_html=True)

ui.kpis([("Fill rate", f"{ui.pct(M['fill_rate']['baseline_mean'])} → {ui.pct(M['fill_rate']['ai_mean'])}", f"+{M['fill_rate']['diff'] * 100:.1f} points", "up"),
         ("Stockout rate", f"{ui.pct(M['stockout_rate']['baseline_mean'])} → {ui.pct(M['stockout_rate']['ai_mean'])}", f"{M['stockout_rate']['relative_change'] * 100:+.0f}%", "up"),
         ("Lost revenue", f"{M['lost_revenue']['relative_change'] * 100:+.0f}%", f"{ui.money(M['lost_revenue']['diff'])} over the quarter", "up"),
         ("Total cost", f"{M['total_cost']['relative_change'] * 100:+.1f}%", f"{ui.money(M['total_cost']['diff'])}", "up"),
         ("Inventory value", f"{M['avg_inventory_value']['relative_change'] * 100:+.0f}%", "the price of higher service", "down")])
st.write("")
t1, t2, t3, t4, t5 = st.tabs(["Results", "Day by day", "Where the money goes", "Ablation", "How we measure it"])

with t1:
    names = [("fill_rate", "Fill rate", "pct"), ("stockout_rate", "Stockout rate", "pct"), ("lost_revenue", "Lost revenue", "m"), ("lost_margin", "Lost margin", "m"),
             ("holding_cost", "Holding cost", "m"), ("ordering_cost", "Ordering cost", "m"), ("backup_premium", "Backup premium", "m"),
             ("total_cost", "Total cost", "m"), ("avg_inventory_value", "Average inventory value", "m")]
    rows = []
    for k, lab, kind in names:
        x = M[k]
        f = ui.pct if kind == "pct" else ui.money
        ci = f"{x['ci_low'] * 100:+.2f} to {x['ci_high'] * 100:+.2f} pts" if kind == "pct" else f"{ui.money(x['ci_low'])} to {ui.money(x['ci_high'])}"
        rows.append({"Measure": lab, "Old rule": f(x["baseline_mean"]), "TFT-driven policy": f(x["ai_mean"]),
                     "Change": (f"{x['diff'] * 100:+.1f} pts" if kind == "pct" else (f"{x['relative_change'] * 100:+.1f}%" if x["baseline_mean"] else "new")),
                     "95% interval of the difference": ci, "p-value": f"{x['p_value']:.1e}" if x["p_value"] == x["p_value"] else "–"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Twin calibration: the old rule replayed in the twin fills {ui.pct(cal['twin_baseline_fill_rate'])} against "
               f"{ui.pct(cal['history_fill_rate'])} recorded in the data. {imp['uncertainty_note']}")

with t2:
    dl = data.art_csv("business_impact", "daily.csv")
    dl["date"] = pd.to_datetime(dl.date)
    fig = go.Figure()
    fig.add_scatter(x=dl.date, y=dl.baseline_fill_rate.rolling(7, min_periods=1).mean() * 100, name="Old rule", line=dict(color=ui.GREY, width=2.5))
    fig.add_scatter(x=dl.date, y=dl.ai_fill_rate.rolling(7, min_periods=1).mean() * 100, name="TFT-driven policy", line=dict(color=ui.NAVY, width=2.5))
    fig.add_vrect(x0="2025-11-28", x1="2025-12-01", fillcolor=ui.CORAL, opacity=0.15, line_width=0, annotation_text="Black Friday – Cyber Monday")
    fig.update_layout(title="Fill rate through the quarter (7-day average)", height=380, yaxis_title="%")
    st.plotly_chart(fig, width="stretch")
    fig = go.Figure()
    fig.add_scatter(x=dl.date, y=dl.baseline_inventory_value, name="Old rule", line=dict(color=ui.GREY, width=2))
    fig.add_scatter(x=dl.date, y=dl.ai_inventory_value, name="TFT-driven policy", line=dict(color=ui.TEAL, width=2))
    fig.update_layout(title="Inventory value held", height=320, yaxis_title="$")
    st.plotly_chart(fig, width="stretch")
    st.markdown("The two policies are similar in October. From mid-November the old rule collapses through the peak, while the TFT policy, "
                "having built stock ahead of Black Friday, recovers within days.")

with t3:
    comp = [("lost_margin", "Lost margin", ui.CORAL), ("holding_cost", "Holding cost", ui.TEAL), ("ordering_cost", "Ordering cost", ui.GREY),
            ("backup_premium", "Backup premium", ui.GOLD), ("overflow_cost", "Overflow cost", "#8E6C8A")]
    lbl = ["Old rule", "TFT + risk switching", "TFT forecasts only"]
    fig = go.Figure()
    for k, n, c in comp:
        fig.add_bar(y=lbl, x=[M[k]["baseline_mean"], M[k]["ai_mean"], AB[k]["ai_mean"]], name=n, orientation="h", marker_color=c)
    fig.update_layout(barmode="stack", title="Total cost over the quarter by component", height=360, xaxis_title="$", yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")
    st.markdown("Lost margin, the biggest cost, falls by about a third. Holding cost rises because more stock is held, and a new cost "
                "appears: the premium paid to backup suppliers.")

with t4:
    st.markdown("**Question:** how much of the gain comes from the forecasts, and how much from switching suppliers on predicted risk?")
    tbl = pd.DataFrame([
        {"Measure": "Fill rate", "Old rule": ui.pct(M["fill_rate"]["baseline_mean"]), "TFT + risk switching": ui.pct(M["fill_rate"]["ai_mean"]), "TFT forecasts only": ui.pct(AB["fill_rate"]["ai_mean"])},
        {"Measure": "Backup premium", "Old rule": ui.money(M["backup_premium"]["baseline_mean"]), "TFT + risk switching": ui.money(M["backup_premium"]["ai_mean"]), "TFT forecasts only": ui.money(AB["backup_premium"]["ai_mean"])},
        {"Measure": "Total cost", "Old rule": ui.money(M["total_cost"]["baseline_mean"]), "TFT + risk switching": ui.money(M["total_cost"]["ai_mean"]), "TFT forecasts only": ui.money(AB["total_cost"]["ai_mean"])},
        {"Measure": "Change in total cost", "Old rule": "–", "TFT + risk switching": f"{M['total_cost']['relative_change'] * 100:+.1f}%", "TFT forecasts only": f"{AB['total_cost']['relative_change'] * 100:+.1f}%"}])
    st.dataframe(tbl, hide_index=True, width="stretch")
    ui.card("Finding", "Removing switching on predicted risk keeps the fill rate and cuts cost further (−16.7% instead of −14.5%). "
            "The forecasts deliver essentially all of the value; supplier switching should stay a manual decision until the risk model improves.", ui.TEAL)

with t5:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### How we measured it")
        for i, (t, b) in enumerate([("Same world for both policies", "actual Q4 2025 demand, actual stock on 30 Sep, supplier delays drawn from each supplier's history, recorded disruptions"),
                                    ("Old rule", "order up to the 28-day average × (lead time + 7 days) + 15%"),
                                    ("TFT-driven policy", "order up to the sum of P50 forecasts over lead time + review, plus safety stock from the P10–P90 spread and lead-time risk (z = 1.65, ≈95% service)"),
                                    ("30 paired replications", "common random numbers; paired t-tests and 95% intervals"),
                                    ("Sanity check", "the twin reproduces the recorded fill rate within 1.5 points")]):
            st.markdown(f'<div class="step"><div class="n">{i + 1}</div><div><b>{t}:</b> {b}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown("#### How a company would measure it: 90-day pilot")
        st.dataframe(pd.DataFrame([("Days 0–30", "Pilot", "TFT alongside the current rule for the top 20 products; compare error weekly"),
                                   ("Days 31–60", "Integrate", "Planner dashboard with forecast ranges; approval workflow"),
                                   ("Days 61–90", "Scale", "All products and warehouses; monthly retraining; drift alerts")],
                                  columns=["When", "Phase", "Activities"]), hide_index=True, width="stretch")
        st.markdown("**KPIs every week:** forecast error (WAPE) · fill rate · stockout rate · inventory value · planner override rate")
        st.markdown("**Success:** TFT error below the current method for 4 straight weeks, then higher fill rate at lower total cost.")
    st.caption("Assumptions: holding cost 25% a year, order cost $500, backup premium 8%, lost sales are lost (no backorders).")

ui.footer()
