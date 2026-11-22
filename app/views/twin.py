import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from lib import data, ui

from supplychain_lens.simulation.snapshot import run_scenario
from supplychain_lens.simulation.twin import Scenario

ui.header("Digital twin & scenarios", "What happens if… ?",
          "A simulation of the whole network for the next 12 weeks from 31 Dec 2025. Every scenario runs both ordering policies on identical random draws.")
st.markdown(ui.simulated("SIMULATED · synthetic data"), unsafe_allow_html=True)

ctx = data.ctx()
state, q = ctx.twin_state, ctx.twin_quantiles
sc = ctx.risk_scores.set_index("supplier_id").loc[state.supplier_ids]
sup = data.table("suppliers").set_index("supplier_id")


@st.cache_data(show_spinner="Running the digital twin…", max_entries=64)
def simulate(sc_dict: dict, use_risk: bool, reps: int) -> dict:
    s = Scenario(**sc_dict)
    thr = float(sc.threshold.iloc[0]) if use_risk else 1.0
    return run_scenario(state, q, s, sc.probability.to_numpy(), thr, reps, 42)


def show(res: dict, ref: dict | None = None):
    b, a = res["baseline"], res["ai"]
    ui.kpis([("Fill rate · old rule", ui.pct(b["fill_rate"]["mean"]), f"90% range {ui.pct(b['fill_rate']['p05'])}–{ui.pct(b['fill_rate']['p95'])}"),
             ("Fill rate · AI policy", ui.pct(a["fill_rate"]["mean"]), f"{(a['fill_rate']['mean'] - b['fill_rate']['mean']) * 100:+.1f} pts vs old rule",
              "up" if a["fill_rate"]["mean"] >= b["fill_rate"]["mean"] else "down"),
             ("Lost revenue · old → AI", f"{ui.money(b['lost_revenue']['mean'])} → {ui.money(a['lost_revenue']['mean'])}", None),
             ("Total cost · old → AI", f"{ui.money(b['total_cost']['mean'])} → {ui.money(a['total_cost']['mean'])}",
              f"{(a['total_cost']['mean'] / b['total_cost']['mean'] - 1) * 100:+.1f}%", "up" if a["total_cost"]["mean"] <= b["total_cost"]["mean"] else "down")])
    if ref:
        st.caption(f"Under normal conditions the old rule fills {ui.pct(ref['baseline']['fill_rate']['mean'])}; "
                   f"this scenario changes that by {(b['fill_rate']['mean'] - ref['baseline']['fill_rate']['mean']) * 100:+.1f} points.")
    dts = pd.date_range("2026-01-01", periods=len(res["daily"]["baseline_fill"]))
    fig = go.Figure()
    fig.add_scatter(x=dts, y=np.array(res["daily"]["baseline_fill"]) * 100, name="Old rule", line=dict(color=ui.GREY, width=2.5))
    fig.add_scatter(x=dts, y=np.array(res["daily"]["ai_fill"]) * 100, name="AI policy (TFT forecasts)", line=dict(color=ui.NAVY, width=2.5))
    fig.update_layout(title="Daily fill rate over the next 12 weeks (mean of replications)", height=360, yaxis_title="%")
    st.plotly_chart(fig, width="stretch")
    keys = [("fill_rate", "Fill rate", True), ("stockout_rate", "Stockout rate", True), ("lost_revenue", "Lost revenue", False), ("holding_cost", "Holding cost", False),
            ("backup_premium", "Backup premium", False), ("total_cost", "Total cost", False), ("avg_inventory_value", "Average inventory value", False), ("backup_orders", "Backup orders", None)]
    rows = []
    for k, lab, is_pct in keys:
        if k not in b:
            continue
        f = (lambda v: ui.pct(v)) if is_pct else ((lambda v: f"{v:,.0f}") if is_pct is None else ui.money)
        rows.append({"Measure": lab, "Old rule": f(b[k]["mean"]), "AI policy": f(a[k]["mean"])})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if a["total_cost"]["mean"] > b["total_cost"]["mean"]:
        st.info(f"Why does the AI policy cost more here? Mostly the 8% premium for routing orders to backup suppliers "
                f"({ui.money(a['backup_premium']['mean'] - b['backup_premium']['mean'])}) when a primary supplier has an active disruption, "
                f"against {ui.money(b['lost_margin']['mean'] - a['lost_margin']['mean'])} of margin protected. January to March is a quiet season; "
                "the forecasts pay off most around peaks (see **Business impact** for the Black Friday quarter).")


t1, t2, t3 = st.tabs(["Scenario builder", "Preset scenarios", "Stock transfer optimiser"])

with t1:
    st.markdown("Set up a disruption, then run both policies through the twin.")
    c1, c2, c3 = st.columns(3)
    demand = c1.slider("Demand change (%)", -30, 50, 0, 5)
    lt = c1.slider("Extra lead time, all suppliers (days)", 0, 20, 0)
    off = c2.selectbox("Take a supplier offline", ["None"] + list(sc.sort_values("probability", ascending=False).index),
                       format_func=lambda s: s if s == "None" else f"{s} · {sup.loc[s, 'name']}")
    cap = c2.slider("Supplier capacity (%)", 40, 100, 100, 5)
    whc = c3.slider("Warehouse capacity (%)", 60, 100, 100, 5)
    tr = c3.slider("Transport delay (days)", 0, 14, 0)
    o1, o2 = st.columns([1, 1])
    use_risk = o1.toggle("AI policy also switches suppliers on predicted risk", value=False,
                         help="Off = switch to backup only on recorded disruptions (recommended; see Business impact → ablation).")
    reps = o2.select_slider("Replications", [10, 20, 30, 40], value=20)
    s = dict(name="Custom scenario", demand_change_pct=float(demand), lead_time_add_days=int(lt), supplier_capacity_pct=float(cap),
             warehouse_capacity_pct=float(whc), transport_delay_days=int(tr), offline_supplier=None if off == "None" else off)
    if st.button("Run simulation", type="primary"):
        st.session_state.twin_custom = (s, use_risk, reps)
    if "twin_custom" in st.session_state:
        s0, ur, rp = st.session_state.twin_custom
        res = simulate(s0, ur, rp)
        ref = simulate(dict(name="Baseline conditions"), ur, rp)
        show(res, ref)
        if s0 != s or ur != use_risk or rp != reps:
            st.info("Settings changed: press **Run simulation** to update.")

with t2:
    pres = ctx.scenarios["results"]
    names = [r["scenario"]["name"] for r in pres]
    pick = st.selectbox("Scenario", names, index=min(3, len(names) - 1))
    r = pres[names.index(pick)]
    show(r, pres[0])
    tbl = pd.DataFrame([{"Scenario": x["scenario"]["name"], "Fill · old rule": x["baseline"]["fill_rate"]["mean"], "Fill · AI": x["ai"]["fill_rate"]["mean"],
                         "Lost revenue · old": x["baseline"]["lost_revenue"]["mean"], "Lost revenue · AI": x["ai"]["lost_revenue"]["mean"]} for x in pres])
    st.dataframe(tbl, column_config={"Fill · old rule": st.column_config.NumberColumn(format="%.3f"), "Fill · AI": st.column_config.NumberColumn(format="%.3f"),
                                     "Lost revenue · old": st.column_config.NumberColumn(format="$%.0f"), "Lost revenue · AI": st.column_config.NumberColumn(format="$%.0f")},
                 hide_index=True, width="stretch")
    st.caption(f"Precomputed with {ctx.scenarios['reps']} replications; the AI policy here also switches on predicted risk, as stored in the snapshot. {ctx.scenarios.get('assumption', '')}")

with t3:
    from supplychain_lens.tools.core import transfer_options
    pos = ctx.positions
    need = pos.groupby("sku_id").expected_lost_revenue_14d.sum().sort_values(ascending=False)
    opts = list(need.index[:30])
    sku = st.selectbox("Product to rebalance", opts, index=opts.index("SKU-0001") if "SKU-0001" in opts else 0,
                       format_func=lambda s: f"{s} · expected lost revenue {ui.money(need[s])} (14 days)")
    if st.button("Optimise transfers (OR-Tools)", type="primary"):
        with st.spinner("Solving a min-cost flow…"):
            out = transfer_options(ctx, sku)
        sm = out["summary"]
        ui.kpis([("Solver status", sm["status"]), ("Units moved", f"{sm['units_moved']:,.0f}"), ("Transfer cost", ui.money(sm["transfer_cost_usd"])),
                 ("Margin protected", ui.money(sm["margin_protected_usd"]), f"net benefit {ui.money(sm['net_benefit_usd'])}", "up")])
        if sm["status"] == "NOTHING_TO_MOVE":
            st.warning("No other warehouse holds spare stock of this product, so a transfer cannot help: it needs an emergency order instead.")
        if out["rows"]:
            st.dataframe(pd.DataFrame(out["rows"]), hide_index=True, width="stretch")
        st.caption(f"Unmet demand before: {sm['unmet_before']:,.0f} units → after: {sm['unmet_after']:,.0f} units.")

ui.footer()
