"""Shared look and small UI components for Supply Chain Lens."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

NAVY, CORAL, TEAL, GOLD, GREY, SAND, CREAM = "#1F2A44", "#E8603C", "#2A9D8F", "#E3A21A", "#9AA5B1", "#F3EEE6", "#FBF8F3"
MODEL_COLORS = {"TFT": NAVY, "LSTM": TEAL, "LightGBM": CORAL, "Holt-Winters": GREY, "Seasonal naive": "#C9CED6"}
SHORT = {"TFT (PyTorch)": "TFT", "LSTM (PyTorch)": "LSTM", "LightGBM (quantile)": "LightGBM", "Holt-Winters (ETS)": "Holt-Winters",
         "Seasonal naive": "Seasonal naive"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, p, li, label, .stMetric { font-family: 'Inter', sans-serif; }
h1, h2, h3, h4 { font-family: 'Poppins', 'Inter', sans-serif !important; color: #1F2A44; letter-spacing: -0.01em; }
.block-container { padding-top: 3.6rem; padding-bottom: 3rem; max-width: 1400px; }
[data-testid="stSidebar"] { background: #1F2A44; }
[data-testid="stSidebar"] * { color: #E8ECF4 !important; }
[data-testid="stSidebarNav"] a[aria-current="page"] { background: rgba(232,96,60,.25) !important; border-radius: 10px; }
.kick { font-size: .78rem; font-weight: 700; letter-spacing: .18em; text-transform: uppercase; color: #E8603C; margin-bottom: .15rem; }
.hero { background: #1F2A44; color: #fff; border-radius: 24px; padding: 2.2rem 2.4rem; position: relative; overflow: hidden; margin-bottom: 1.2rem; }
.hero h1 { color: #fff !important; font-size: 2.6rem; margin: .2rem 0 .4rem 0; }
.hero p { color: #DCE6F0; font-size: 1.08rem; max-width: 640px; margin: 0; }
.hero > div:not(.b1):not(.b2), .hero h1, .hero p { position: relative; z-index: 1; }
.hero .b1 { position: absolute; right: -90px; top: -110px; width: 280px; height: 280px; border-radius: 50%; background: #E8603C; }
.hero .b2 { position: absolute; right: 60px; bottom: -130px; width: 190px; height: 190px; border-radius: 50%; background: #E3A21A; }
.kpi { background: #fff; border-radius: 18px; padding: 1.05rem 1.2rem; box-shadow: 0 4px 18px rgba(31,42,68,.06); height: 100%; border: 1px solid #EFE8DC; }
.kpi .l { font-size: .82rem; color: #6B7280; font-weight: 600; }
.kpi .v { font-family: 'Poppins', sans-serif; font-size: 1.75rem; font-weight: 600; color: #1F2A44; line-height: 1.2; margin-top: .2rem; }
.kpi .d { font-size: .82rem; margin-top: .25rem; }
.up { color: #2A9D8F; } .down { color: #E8603C; } .flat { color: #6B7280; }
.card { background: #fff; border-radius: 18px; padding: 1.1rem 1.3rem; box-shadow: 0 4px 18px rgba(31,42,68,.06); border: 1px solid #EFE8DC; margin-bottom: .8rem; }
.card h4 { margin: 0 0 .35rem 0; font-size: 1.05rem; }
.card p { margin: 0; color: #374151; font-size: .95rem; }
.tag { display: inline-block; padding: .15rem .6rem; border-radius: 999px; font-size: .75rem; font-weight: 700; margin-right: .3rem; }
.t-crit { background: #FDE2DA; color: #B4381A; } .t-high { background: #FCEBD0; color: #9A6510; }
.t-med { background: #E3F2EF; color: #1E7A6E; } .t-low { background: #EEF1F5; color: #4B5563; }
.t-auto { background: #E3F2EF; color: #1E7A6E; } .t-pend { background: #FCEBD0; color: #9A6510; } .t-info { background: #EEF1F5; color: #4B5563; }
.sim { display:inline-block; background:#FCEBD0; color:#9A6510; border-radius:999px; padding:.1rem .55rem; font-size:.72rem; font-weight:700; letter-spacing:.04em; }
.step { display:flex; gap:.9rem; align-items:flex-start; margin-bottom:.7rem; }
.step .n { flex:none; width:34px; height:34px; border-radius:50%; background:#E8603C; color:#fff; font-weight:700; display:flex; align-items:center; justify-content:center; }
.small { font-size: .85rem; color: #6B7280; }
div[data-testid="stTabs"] button p { font-weight: 600; }
</style>
"""


def setup():
    st.markdown(CSS, unsafe_allow_html=True)
    _plotly_template()


def _plotly_template():
    if "scl" in pio.templates:
        pio.templates.default = "scl"
        return
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family="Inter, sans-serif", size=13, color=NAVY),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
        colorway=[NAVY, CORAL, TEAL, GOLD, GREY, "#8E6C8A"],
        xaxis=dict(gridcolor="#EEEAE2", zeroline=False, linecolor="#D9D3C7"),
        yaxis=dict(gridcolor="#EEEAE2", zeroline=False, linecolor="#D9D3C7"),
        margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.18, x=0),
        title=dict(font=dict(family="Poppins, Inter, sans-serif", size=16, color=NAVY), x=0, xanchor="left"),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    pio.templates["scl"] = t
    pio.templates.default = "scl"


def hero(kick: str, title: str, text: str):
    st.markdown(f'<div class="hero"><div class="b1"></div><div class="b2"></div><div class="kick" style="color:#E3A21A">{kick}</div>'
                f'<h1>{title}</h1><p>{text}</p></div>', unsafe_allow_html=True)


def header(kick: str, title: str, text: str | None = None):
    st.markdown(f'<div class="kick">{kick}</div>', unsafe_allow_html=True)
    st.markdown(f"## {title}")
    if text:
        st.caption(text)


def kpis(items: list[tuple]):
    """items: (label, value, delta_text or None, direction 'up'|'down'|'flat')"""
    cols = st.columns(len(items))
    for c, it in zip(cols, items, strict=False):
        label, value = it[0], it[1]
        delta = it[2] if len(it) > 2 else None
        d = it[3] if len(it) > 3 else "flat"
        dh = f'<div class="d {d}">{delta}</div>' if delta else ""
        c.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>{dh}</div>', unsafe_allow_html=True)


def card(title: str, body: str, color: str | None = None):
    style = f' style="border-left:5px solid {color}"' if color else ""
    st.markdown(f'<div class="card"{style}><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)


def simulated(text: str = "SIMULATED"):
    return f'<span class="sim">{text}</span>'


def tag(level: str) -> str:
    cls = {"CRITICAL": "t-crit", "HIGH": "t-high", "MEDIUM": "t-med", "LOW": "t-low", "AUTO_APPROVED": "t-auto",
           "PENDING_APPROVAL": "t-pend", "INVESTIGATION_REQUIRED": "t-pend", "INFORMATION": "t-info",
           "APPROVED": "t-auto", "REJECTED": "t-crit"}.get(level, "t-low")
    return f'<span class="tag {cls}">{level.replace("_", " ").title()}</span>'


def money(x: float) -> str:
    a = abs(x)
    s = f"${a / 1e9:.2f}B" if a >= 1e9 else f"${a / 1e6:.1f}M" if a >= 1e6 else f"${a / 1e3:.0f}K" if a >= 1e3 else f"${a:,.0f}"
    return ("−" if x < 0 else "") + s


def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


def footer():
    st.markdown('<div class="small" style="margin-top:2rem;border-top:1px solid #E6DFD2;padding-top:.8rem">'
                'Supply Chain Lens · Deep Learning Project · BITS Pilani · All data synthetic (NEXUS-SCCT data model); '
                'business results are simulated. Reference project: '
                '<a href="https://github.com/ahmedzainsyed/nexus-scct">NEXUS-SCCT</a> by Syed Zain Ahmed.</div>',
                unsafe_allow_html=True)
