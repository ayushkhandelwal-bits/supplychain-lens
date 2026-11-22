"""Cached agent run (about 1 second) shared by the control tower and the decision room."""

import streamlit as st

from lib import data


@st.cache_resource(show_spinner="Agents are calling the tools…")
def results():
    from supplychain_lens.agents.orchestrator import run_all
    return run_all(data.ctx())
