"""Cached access to the project's data, models and results (no PyTorch needed)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from supplychain_lens.serving import registry  # noqa: E402
from supplychain_lens.tools.context import Context  # noqa: E402

DATA = ROOT / "data" / "nexus_v2"


@st.cache_resource(show_spinner="Loading the network data and models…")
def ctx() -> Context:
    return Context()


@st.cache_data(show_spinner=False)
def table(name: str) -> pd.DataFrame:
    return pd.read_parquet(DATA / f"{name}.parquet")


def _run(model: str):
    r = registry.load_run(model)
    if r is None:
        raise FileNotFoundError(f"Trained artifact '{model}' not found. Run `python -m pipelines.build_all` first.")
    return r


@st.cache_data(show_spinner=False)
def art_json(model: str, name: str) -> dict:
    return _run(model).json(name)


@st.cache_data(show_spinner=False)
def art_csv(model: str, name: str) -> pd.DataFrame:
    return pd.read_csv(_run(model).file(name))


@st.cache_data(show_spinner=False)
def art_parquet(model: str, name: str) -> pd.DataFrame:
    return pd.read_parquet(_run(model).file(name))


@st.cache_data(show_spinner=False)
def art_version(model: str) -> str:
    return _run(model).version


@st.cache_resource(show_spinner="Loading the TFT (ONNX)…")
def forecaster(model_artifact: str | None = None):
    from supplychain_lens.serving.forecaster import Forecaster
    return Forecaster.load(model_artifact, panel=panel())


@st.cache_resource(show_spinner=False)
def panel():
    from supplychain_lens.features.forecasting import load_panel
    return load_panel()


@st.cache_data(show_spinner=False)
def daily_network() -> pd.DataFrame:
    d = table("demand_daily").groupby("date")[["demand_units", "sales_units", "lost_sales_units"]].sum()
    return d.join(table("kpi_daily").set_index("date"))


@st.cache_data(show_spinner=False)
def headline() -> dict:
    cmp = art_json("forecast_comparison", "comparison.json")
    imp = art_json("business_impact", "results.json")
    risk = art_json("risk_classifier", "metrics.json")
    return {"cmp": cmp, "imp": imp, "risk": risk}


def snapshot_meta() -> dict:
    return art_json("snapshot", "meta.json")


def readme_json(path: Path) -> dict:
    return json.loads(path.read_text())
