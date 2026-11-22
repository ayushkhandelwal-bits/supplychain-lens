import streamlit as st
from lib import data, ui

from supplychain_lens.llm import copilot

ui.header("Ask the copilot", "Questions in plain English, answered from the tools",
          "The copilot routes each question to one of the deterministic tools and answers only with numbers those tools return.")

ctx = data.ctx()
if "chat" not in st.session_state:
    st.session_state.chat = []

st.markdown("**Try one:**")
cols = st.columns(3)
clicked = None
for i, ex in enumerate(copilot.EXAMPLES):
    if cols[i % 3].button(ex, key=f"ex{i}", width="stretch"):
        clicked = ex

for role, text, extra in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(text)
        if extra:
            st.caption(extra)

q = st.chat_input("Ask about forecasts, stockouts, suppliers, anomalies or scenarios…") or clicked
if q:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Calling the tools…"):
            a = copilot.ask(ctx, q, allow_llm=False)
        lines = [x for x in a.text.split("\n") if x.strip()]
        st.markdown(lines[0] + ("\n" + "\n".join(f"- {x}" for x in lines[1:]) if len(lines) > 1 else ""))
        extra = f"Answered with the `{a.tool}` tool"
        st.caption(extra)
        if a.rows:
            st.dataframe(a.rows[:10], width="stretch")
    lines = [x for x in a.text.split("\n") if x.strip()]
    st.session_state.chat += [("user", q, None), ("assistant", lines[0] + ("\n" + "\n".join(f"- {x}" for x in lines[1:]) if len(lines) > 1 else ""), extra)]

ui.footer()
