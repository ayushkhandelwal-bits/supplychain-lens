"""Supply Chain Lens: the project's interactive application.

Run:  streamlit run app/Home.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "app"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

st.set_page_config(page_title="Supply Chain Lens", page_icon="🔎", layout="wide", initial_sidebar_state="expanded")

from lib import ui  # noqa: E402

ui.setup()

V = "views/"
pages = {
    "Project": [
        st.Page(V + "overview.py", title="Project overview", icon=":material/home:", default=True),
        st.Page(V + "data_eda.py", title="Data explorer", icon=":material/database:"),
    ],
    "Control tower": [
        st.Page(V + "control_tower.py", title="Executive control tower", icon=":material/monitoring:"),
        st.Page(V + "decision_room.py", title="AI decision room", icon=":material/gavel:"),
        st.Page(V + "copilot.py", title="Ask the copilot", icon=":material/chat:"),
    ],
    "Deep learning models": [
        st.Page(V + "dl_lab.py", title="Deep learning lab", icon=":material/neurology:"),
        st.Page(V + "demand.py", title="Demand intelligence", icon=":material/trending_up:"),
        st.Page(V + "anomaly.py", title="Anomaly detection", icon=":material/troubleshoot:"),
        st.Page(V + "risk.py", title="Supplier risk", icon=":material/warning:"),
    ],
    "Simulation & impact": [
        st.Page(V + "twin.py", title="Digital twin & scenarios", icon=":material/science:"),
        st.Page(V + "network.py", title="Supply network", icon=":material/hub:"),
        st.Page(V + "impact.py", title="Business impact", icon=":material/payments:"),
        st.Page(V + "evaluation.py", title="Model evaluation", icon=":material/fact_check:"),
    ],
}

with st.sidebar:
    st.markdown("<div style='font-family:Poppins;font-size:1.45rem;font-weight:600;color:#fff'>Supply Chain Lens</div>"
                "<div style='font-size:.8rem;opacity:.8;margin-bottom:.6rem'>Deep learning for supply chain decisions</div>",
                unsafe_allow_html=True)

nav = st.navigation(pages, expanded=True)
nav.run()

with st.sidebar:
    st.markdown("<div style='font-size:.75rem;opacity:.75;margin-top:1rem'>Deep Learning Project · BITS Pilani<br>"
                "Prof. Abhishek Verma<br><br>Ayush Khandelwal · Bharadwaj Joshi · Divya Jain · Navya Badaya<br><br>"
                "All data synthetic · business results simulated</div>", unsafe_allow_html=True)
