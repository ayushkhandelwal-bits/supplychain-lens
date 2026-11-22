import pandas as pd
import streamlit as st
from lib import ui
from lib.agents import results

res = results()
ui.header("AI decision room", "Agents recommend; rules check; a manager decides",
          "Five specialist agents call deterministic tools and must cite every number. A coded policy decides what may be automated.")

if "verdicts" not in st.session_state:
    st.session_state.verdicts = {}

t1, t2, t3 = st.tabs(["Recommendations", "How the agents worked", "Approval policy"])

with t1:
    decisions = res.decisions
    types = sorted({d.decision_type for d in decisions})
    f1, f2 = st.columns([2, 1])
    pick = f1.multiselect("Show decision types", types, default=types)
    only_pending = f2.toggle("Only those needing approval", value=False)
    shown = [d for d in decisions if d.decision_type in pick and (not only_pending or d.requires_human_approval)]
    approved = sum(v == "APPROVED" for v in st.session_state.verdicts.values())
    rejected = sum(v == "REJECTED" for v in st.session_state.verdicts.values())
    ui.kpis([("Recommendations", f"{len(decisions)}"), ("Need approval", f"{sum(d.requires_human_approval for d in decisions)}"),
             ("Approved in this session", f"{approved}", None, "up"), ("Rejected in this session", f"{rejected}", None, "down")])
    st.write("")
    for d in shown:
        verdict = st.session_state.verdicts.get(d.decision_id)
        status = verdict or d.status
        with st.expander(f"{d.title}", expanded=False):
            st.markdown(f"{ui.tag(d.risk_level)} {ui.tag(status)} <span class='small'>{d.agent} agent · {d.decision_type.replace('_', ' ').title()} · {d.decision_id}</span>",
                        unsafe_allow_html=True)
            c1, c2 = st.columns([1.6, 1])
            with c1:
                st.markdown(f"**Action:** {d.action}")
                st.markdown("**Evidence (from tools):**")
                for e in d.evidence:
                    st.markdown(f"- {e}")
                if d.alternatives:
                    st.markdown("**Alternatives considered:**")
                    st.dataframe(pd.DataFrame([a.__dict__ for a in d.alternatives]), hide_index=True, width="stretch")
                if d.uncertainty:
                    st.caption(f"Uncertainty: {d.uncertainty}")
            with c2:
                st.metric("Confidence", f"{d.confidence * 100:.0f}%", help=d.confidence_basis)
                st.caption(d.confidence_basis)
                if d.estimated_value_usd:
                    st.metric("Value at stake", ui.money(d.estimated_value_usd))
                st.caption(f"Reversible: {'yes' if d.reversible else 'no'}")
                if d.approval_reason:
                    st.caption(f"Policy: {d.approval_reason}")
                if d.requires_human_approval and d.decision_type not in ("INFORMATION",):
                    b1, b2 = st.columns(2)
                    if b1.button("Approve", key="a" + d.decision_id, type="primary", width="stretch"):
                        st.session_state.verdicts[d.decision_id] = "APPROVED"
                        st.rerun()
                    if b2.button("Reject", key="r" + d.decision_id, width="stretch"):
                        st.session_state.verdicts[d.decision_id] = "REJECTED"
                        st.rerun()

with t2:
    tr = pd.DataFrame(res.trace)
    tr["tools"] = tr.tools.apply(lambda x: ", ".join(x))
    st.dataframe(tr.rename(columns={"agent": "Agent", "tools": "Tools called", "decisions": "Decisions", "seconds": "Seconds"}), hide_index=True, width="stretch")
    for name, rep in res.reports.items():
        with st.expander(f"{name} agent: findings"):
            for f in rep.findings:
                st.markdown(f"- {f}")
    st.caption(f"Full run: {res.seconds:.1f} s. Language layer: templates by default; an optional local LLM may rephrase, "
               "but any answer containing a number absent from the tool output is discarded.")

with t3:
    st.markdown("An action is **auto-approved only if all four hold**; otherwise it goes to a manager with the reasons listed.")
    st.dataframe(pd.DataFrame([("Confidence", "≥ 85%", "computed from the models (e.g. Monte Carlo stockout probability)"),
                               ("Risk level", "not CRITICAL", "critical actions always reviewed"),
                               ("Value at stake", "< USD 5 million", "large commitments reviewed"),
                               ("Reversibility", "easily reversible", "our addition to the NEXUS rule")],
                              columns=["Condition", "Required", "Why"]), hide_index=True, width="stretch")
    st.markdown("Anomaly alerts are **never** auto-approved: they always go to a person for investigation.")

ui.footer()
