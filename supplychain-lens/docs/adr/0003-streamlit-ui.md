# ADR-0003 — Streamlit + Plotly for the dashboard

- **Status:** Accepted
- **Date:** 2026-10-03

## Context

The NEXUS UI is a single 1,200-line React file with inline styles, static data and `Math.random()` KPIs. The Next.js app described in its README is not in the repository, and its `package.json` can't install (audit check A6). Hosting a Next.js frontend *and* a Python backend for free means two hosts and a cross-origin API.

## Decision

Build the UI in **Streamlit** (multipage) with **Plotly** charts, a custom theme (`.streamlit/config.toml`) and a small CSS layer for cards and typography. The analytics, models and agents are Python, so the UI calls them directly — no separate API is needed for the demo.

Design direction: a calm, minimal "fintech" look — lots of whitespace, clear typography, one accent colour of our own, KPI cards with deltas. Inspired by the *philosophy* of modern consumer finance apps; no logos, colours, layouts or components copied from any brand.

## Consequences

- One deployable unit on a free host.
- Less pixel-level control than React; we'll compensate with reusable components (`kpi_card`, `decision_card`) and a consistent Plotly template.
- We're replacing the NEXUS UI rather than porting it, so all UI work is ours.
