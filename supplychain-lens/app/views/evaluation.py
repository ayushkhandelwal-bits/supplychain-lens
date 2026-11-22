import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from lib import data, ui

cmp = data.art_json("forecast_comparison", "comparison.json")
ov = cmp["overall"]
ORDER = ["TFT (PyTorch)", "LSTM (PyTorch)", "LightGBM (quantile)", "Holt-Winters (ETS)", "Seasonal naive"]

ui.header("Model evaluation", "How every model was judged",
          "Strict time split, one look at the test quarter, significance tests, and honest reporting of losses.")

t1, t2, t3, t4, t5 = st.tabs(["Forecast accuracy", "Significance", "By horizon & category", "Peak week", "Method & limitations"])

with t1:
    split = st.radio("Period", ["test", "val"], format_func=lambda s: {"test": "Test (Oct–Dec 2025)", "val": "Validation (Jul–Sep 2025)"}[s], horizontal=True)
    rows = []
    for m in ORDER:
        x = ov[m][split]
        rows.append({"Model": ui.SHORT[m], "WAPE": x["wape"], "MAE": x["mae"], "RMSE": x["rmse"], "Bias": x["bias"],
                     "Pinball (scaled)": x.get("pinball_scaled", float("nan")), "P10–P90 coverage": x.get("coverage_p10_p90", float("nan"))})
    df = pd.DataFrame(rows)
    st.dataframe(df, column_config={"WAPE": st.column_config.NumberColumn(format="%.3f"), "MAE": st.column_config.NumberColumn(format="%.1f"),
                                    "RMSE": st.column_config.NumberColumn(format="%.1f"), "Bias": st.column_config.NumberColumn(format="%.3f"),
                                    "Pinball (scaled)": st.column_config.NumberColumn(format="%.4f"), "P10–P90 coverage": st.column_config.NumberColumn(format="%.3f")},
                 hide_index=True, width="stretch")
    st.caption("400 series × 3 forecast origins × 28 days = 33,600 points per period. Model selection used validation only. "
               "WAPE = total absolute error ÷ total demand (12.5% = off by 12.5 units per 100 sold). Coverage target 80%.")

with t2:
    bs = cmp["bootstrap_vs_selected"]
    rows = [{"TFT vs": ui.SHORT[m], "WAPE difference (pts)": bs[m]["diff"] * 100, "95% low": bs[m]["ci_low"] * 100, "95% high": bs[m]["ci_high"] * 100,
             "Significant": "Yes" if bs[m]["significant"] else "No"} for m in ORDER[1:]]
    df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_scatter(x=df["WAPE difference (pts)"], y=df["TFT vs"], mode="markers", marker=dict(size=12, color=ui.NAVY),
                    error_x=dict(type="data", symmetric=False, array=df["95% high"] - df["WAPE difference (pts)"], arrayminus=df["WAPE difference (pts)"] - df["95% low"]))
    fig.add_vline(x=0, line_dash="dot")
    fig.update_layout(title="TFT minus competitor, WAPE points (left of 0 = TFT better), 95% bootstrap interval", height=340)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(df, hide_index=True, width="stretch")
    st.caption("Paired bootstrap: the 400 series were resampled 1,000 times and the difference recomputed each time.")

with t3:
    bh = cmp["by_horizon_test"]
    labs = list(bh[ORDER[0]].keys())
    fig = go.Figure()
    for m in ORDER:
        fig.add_scatter(x=labs, y=[bh[m][lab] * 100 for lab in labs], name=ui.SHORT[m], mode="lines+markers", line=dict(color=ui.MODEL_COLORS[ui.SHORT[m]], width=3 if m == ORDER[0] else 1.8))
    st.plotly_chart(fig.update_layout(title="Test WAPE by horizon week", height=360, yaxis_title="%"), width="stretch")
    bc = cmp["by_category_test"]
    cats = sorted(bc[ORDER[0]], key=lambda c: bc[ORDER[0]][c])
    fig = go.Figure()
    for m in ORDER[:4]:
        fig.add_bar(x=cats, y=[bc[m][c] * 100 for c in cats], name=ui.SHORT[m], marker_color=ui.MODEL_COLORS[ui.SHORT[m]])
    st.plotly_chart(fig.update_layout(title="Test WAPE by product category", height=380, yaxis_title="%", barmode="group"), width="stretch")

with t4:
    pk = cmp["peak_week_test"]
    fig = go.Figure(go.Bar(x=[ui.SHORT[m] for m in ORDER], y=[pk[m]["wape"] * 100 for m in ORDER], marker_color=[ui.MODEL_COLORS[ui.SHORT[m]] for m in ORDER],
                           text=[f"{pk[m]['wape'] * 100:.1f}%" for m in ORDER], textposition="outside"))
    st.plotly_chart(fig.update_layout(title="Forecast error in Black Friday / Cyber Monday week", height=380, yaxis_title="WAPE %", yaxis_range=[0, 52]), width="stretch")
    st.dataframe(pd.DataFrame([{"Model": ui.SHORT[m], "WAPE": pk[m]["wape"], "Bias": pk[m]["bias"], "Coverage": pk[m].get("coverage_p10_p90", float("nan"))} for m in ORDER]),
                 column_config={k: st.column_config.NumberColumn(format="%.3f") for k in ["WAPE", "Bias", "Coverage"]}, hide_index=True, width="stretch")
    st.markdown("The TFT's lead over LightGBM grows from 1.2 points overall to 3.1 points in peak week; statistical methods under-forecast by about a third.")

with t5:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Method")
        st.markdown("- Train Jan 2023 – Jun 2025, validate Q3 2025, **test Q4 2025 once**\n- Selection and thresholds on validation only\n"
                    "- Leakage controls: future inputs = calendar, planned promotions, prices only; hidden truth unreadable by model code (tested)\n"
                    "- Paired bootstrap for forecasts; paired t-tests for the business simulation\n- Seed 42; 32 automated tests; checksummed artifacts")
    with c2:
        st.markdown("#### Limitations")
        st.markdown("- Synthetic data on the NEXUS-SCCT model, not a real company\n- Business impact is simulated in a digital twin\n"
                    "- Impact intervals vary supplier delays only, not demand\n- Forecast ranges slightly narrow at peak (coverage 76–79%)\n"
                    "- Only 13 supplier failures in the test quarter: risk evaluation is noisy")
        st.markdown("#### Next steps")
        st.markdown("- Pilot on real company data\n- Graph neural network over supplier links\n- Reinforcement-learning ordering policy in the twin\n- Drift monitoring and monthly retraining")

ui.footer()
