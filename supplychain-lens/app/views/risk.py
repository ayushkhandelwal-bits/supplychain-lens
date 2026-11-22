import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from lib import data, ui
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

ui.header("Supplier risk", "Which suppliers may be disrupted in the next 14 days?",
          "Three models score every supplier each day from 19 trailing signals. Deep learning lost this one, and the page shows why.")

ctx = data.ctx()
rs = ctx.risk_scores.assign(active_recorded_event=lambda d: d.own_active_event > 0)
rm = data.art_json("risk_classifier", "metrics.json")
sel = rm.get("selected", "Neural network (PyTorch MLP)")
thr = float(rs.threshold.iloc[0])

t1, t2, t3, t4 = st.tabs(["Current risk scores", "Model comparison", "What drives risk", "Supplier drill-down"])

with t1:
    MODELS = {"Neural network (in use)": "p_Neural network (PyTorch MLP)", "Logistic regression": "p_Logistic regression", "XGBoost": "p_XGBoost"}
    mname = st.radio("Score suppliers with", list(MODELS), horizontal=True)
    rs = rs.assign(probability=rs[MODELS[mname]]).sort_values("probability", ascending=False)
    nn = "in use" in mname
    ui.kpis([("Suppliers above alert threshold", f"{int((rs.probability >= thr).sum())} / {len(rs)}" if nn else "–", f"threshold {thr:.3f}" if nn else "threshold set for the neural network only"),
             ("Highest score", f"{rs.probability.iloc[0]:.3f}", rs.supplier_id.iloc[0]),
             ("Active recorded disruptions", f"{int(rs.active_recorded_event.sum()) if 'active_recorded_event' in rs else 0}", "handled regardless of score"),
             ("Model shown", mname.split(" (")[0], "selected on validation" if "in use" in mname else "comparison model")])
    fig = go.Figure(go.Bar(x=rs.supplier_id, y=rs.probability, marker_color=[ui.CORAL if a else (ui.GOLD if nn and p >= thr else ui.NAVY)
                                                                              for p, a in zip(rs.probability, rs.get("active_recorded_event", [False] * len(rs)), strict=False)],
                           hovertext=rs.name + " · " + rs.country))
    if nn:
        fig.add_hline(y=thr, line_dash="dot", annotation_text="alert threshold")
    fig.update_layout(title=f"{mname.split(' (')[0]}: 14-day disruption probability by supplier, 31 Dec 2025 (coral = active recorded disruption)", height=380)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(rs[["supplier_id", "name", "country", "category", "p_Neural network (PyTorch MLP)", "p_Logistic regression", "p_XGBoost", "active_recorded_event",
                     "health_score", "late_share_28", "overdue_open"]].rename(columns={"p_Neural network (PyTorch MLP)": "Neural network", "p_Logistic regression": "Logistic regression", "p_XGBoost": "XGBoost"}),
                 hide_index=True, width="stretch", height=300)
    st.caption("The threshold line belongs to the neural network; the other models' scores are on their own scales and shown for comparison.")

with t2:
    tp = data.art_parquet("risk_classifier", "test_predictions.parquet")
    names = {"Neural network (PyTorch MLP)": ui.NAVY, "Logistic regression": ui.TEAL, "XGBoost": ui.CORAL}
    rows = []
    c1, c2 = st.columns(2)
    roc, pr = go.Figure(), go.Figure()
    for n, col in names.items():
        p = tp[f"p_{n}"]
        fpr, tpr, _ = roc_curve(tp.y, p)
        pp, rr, _ = precision_recall_curve(tp.y, p)
        roc.add_scatter(x=fpr, y=tpr, name=f"{n} ({roc_auc_score(tp.y, p):.3f})", line=dict(color=col))
        pr.add_scatter(x=rr, y=pp, name=f"{n} ({average_precision_score(tp.y, p):.3f})", line=dict(color=col))
        v, t = rm["models"][n]["val"], rm["models"][n]["test"]
        rows.append({"Model": n, "Validation PR-AUC": v["pr_auc"], "Test PR-AUC": t["pr_auc"], "Test ROC-AUC": t["roc_auc"], "Test Brier": t["brier"]})
    roc.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dot", color=ui.GREY), showlegend=False)
    pr.add_hline(y=tp.y.mean(), line_dash="dot", annotation_text=f"random = {tp.y.mean():.3f}")
    c1.plotly_chart(roc.update_layout(title="ROC curve, test quarter", height=400, xaxis_title="False positive rate", yaxis_title="True positive rate"), width="stretch")
    c2.plotly_chart(pr.update_layout(title="Precision–recall curve, test quarter", height=400, xaxis_title="Recall", yaxis_title="Precision", yaxis_range=[0, 0.4]), width="stretch")
    st.dataframe(pd.DataFrame(rows), column_config={k: st.column_config.NumberColumn(format="%.3f") for k in ["Validation PR-AUC", "Test PR-AUC", "Test ROC-AUC", "Test Brier"]},
                 hide_index=True, width="stretch")
    ui.card("Why deep learning lost here", "Disruptions are rare: 3.6% of supplier-days and only 13 supplier failures in the test quarter. "
            "The neural network ranked best on validation (PR-AUC 0.179) but worst on test (0.058): it overfitted. "
            "Recommendation: use XGBoost or logistic regression for supplier risk in production.", ui.GOLD)

with t3:
    pi = data.art_csv("risk_classifier", "permutation_importance.csv").sort_values("pr_auc_drop")
    fig = go.Figure(go.Bar(x=pi.pr_auc_drop * 1000, y=pi.feature.str.replace("_", " "), orientation="h", error_x=dict(array=pi["std"] * 1000), marker_color=ui.TEAL))
    fig.update_layout(title="Permutation importance (drop in PR-AUC × 1000 when the feature is shuffled)", height=520)
    st.plotly_chart(fig, width="stretch")
    cal = pd.DataFrame(rm["models"][sel]["calibration_test"])
    fig = go.Figure()
    fig.add_scatter(x=[0, 0.15], y=[0, 0.15], line=dict(dash="dot", color=ui.GREY), name="perfect")
    fig.add_scatter(x=cal.mean_predicted, y=cal.observed_rate, mode="lines+markers", line=dict(color=ui.NAVY), name="neural network")
    st.plotly_chart(fig.update_layout(title="Calibration on the test quarter", height=340, xaxis_title="Mean predicted probability", yaxis_title="Observed rate"), width="stretch")

with t4:
    sid = st.selectbox("Supplier", rs.supplier_id, format_func=lambda s: f"{s} · {rs.set_index('supplier_id').loc[s, 'name']}")
    sup = data.table("suppliers").set_index("supplier_id").loc[sid]
    ship = data.table("shipments")
    sh = ship[(ship.supplier_id == sid) & ship.actual_arrival.notna()].copy()
    sh["month"] = pd.to_datetime(sh.actual_arrival).dt.to_period("M").dt.start_time
    mm = sh.groupby("month").agg(late=("is_delayed", "mean"), delay=("delay_days", "mean"))
    ui.kpis([("Country", sup.country, sup.region), ("Health score", f"{sup.health_score}", sup.risk_level), ("Avg lead time", f"{sup.avg_lead_time_days} days", None),
             ("Annual spend", ui.money(sup.annual_spend_usd), None)])
    fig = go.Figure()
    fig.add_bar(x=mm.index, y=mm.late * 100, name="Late shipments (%)", marker_color=ui.GOLD)
    dis = data.table("disruption_suppliers")
    ev = data.table("disruptions").merge(dis[dis.supplier_id == sid], on="event_id")
    for _, e in ev.iterrows():
        x1 = e.resolved_at if pd.notna(e.resolved_at) else pd.Timestamp("2025-12-31")
        fig.add_vrect(x0=e.detected_at, x1=x1, fillcolor=ui.CORAL, opacity=0.15, line_width=0)
    fig.update_layout(title=f"{sid}: monthly share of late shipments (shaded = recorded disruptions)", height=360, yaxis_title="%")
    st.plotly_chart(fig, width="stretch")
    if len(ev):
        st.dataframe(ev[["event_id", "event_type", "severity", "title", "detected_at", "resolved_at"]], hide_index=True, width="stretch")
    dep = data.table("products")
    st.markdown(f"**Products depending on {sid}:** {', '.join(dep[dep.primary_supplier_id == sid].sku_id) or 'none as primary'}"
                f"; backup for {', '.join(dep[dep.backup_supplier_id == sid].sku_id) or 'none'}.")

ui.footer()
