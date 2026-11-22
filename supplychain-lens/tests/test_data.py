"""Data generator: reproducibility, quality, NEXUS fidelity and leakage guards."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from supplychain_lens.data import generator as gen
from supplychain_lens.data import quality

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def v2():
    return gen.generate_v2(gen.GeneratorConfig())


def test_same_seed_same_data(v2):
    again = gen.generate_v2(gen.GeneratorConfig())
    for name in ("demand_daily", "shipments", "disruptions", "inventory_daily"):
        pd.testing.assert_frame_equal(v2["tables"][name], again["tables"][name])


def test_different_seed_different_data(v2):
    other = gen.generate_v2(gen.GeneratorConfig(seed=7))
    assert not v2["tables"]["demand_daily"].demand_units.equals(other["tables"]["demand_daily"].demand_units)


def test_all_quality_checks_pass(v2):
    failed = [f"{c.area}: {c.name} ({c.detail})" for c in quality.run_checks(v2["tables"]) if not c.passed]
    assert not failed, failed


def test_expected_shapes(v2):
    t = v2["tables"]
    assert len(t["suppliers"]) == 50 and len(t["warehouses"]) == 12 and len(t["products"]) == 100
    assert len(t["sku_warehouse"]) == 400
    assert len(t["demand_daily"]) == 400 * 1096
    assert t["demand_daily"].date.max() == pd.Timestamp("2025-12-31")


def test_disruption_flows_into_operations(v2):
    """A supplier disruption must show up in that supplier's shipments (causal consistency)."""
    t = v2["tables"]
    ev = t["disruptions"].merge(t["disruption_suppliers"], on="event_id")
    ev = ev[ev.severity.isin(["HIGH", "CRITICAL"]) & (ev.event_type != "DEMAND")]
    po = t["purchase_orders"].merge(t["shipments"][["po_id", "delay_days"]], on="po_id")
    po = po[po.status == "RECEIVED"]
    during, outside = [], []
    for e in ev.itertuples():
        mine = po[po.supplier_id == e.supplier_id]
        end = e.resolved_at if pd.notna(e.resolved_at) else pd.Timestamp("2025-12-31")
        inside = mine.order_date.between(e.detected_at, end)
        during += list(mine[inside].delay_days)
        outside += list(mine[~inside].delay_days)
    assert pd.Series(during).mean() > pd.Series(outside).mean() + 3


def test_v1_mode_keeps_original_shape_and_defects():
    v1 = gen.generate_v1(gen.GeneratorConfig())["tables"]
    assert len(v1["suppliers"]) == 50 and len(v1["shipments"]) == 300 and len(v1["disruptions"]) == 50
    d = v1["demand_history"]
    assert d.warehouse_id.nunique() == 4  # D4 reproduced on purpose
    assert d.groupby(["sku_id", "warehouse_id"]).price.nunique().median() > 300  # D1 reproduced on purpose


def test_black_friday_rule():
    assert gen.black_friday(2023).isoformat() == "2023-11-24"
    assert gen.black_friday(2024).isoformat() == "2024-11-29"
    assert gen.black_friday(2025).isoformat() == "2025-11-28"


ALLOWED_TRUTH_READERS = {
    "supplychain_lens/data/generator.py",
    "supplychain_lens/data/dictionary.py",
    "supplychain_lens/evaluation/labels.py",
}


def test_truth_is_only_read_by_label_and_data_code():
    """Hidden truth must never reach feature or model code (anti-leakage guard)."""
    offenders = []
    for path in (ROOT / "supplychain_lens").rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if rel in ALLOWED_TRUTH_READERS:
            continue
        if re.search(r"""["'/]truth["'/]|truth_dir|data/truth""", path.read_text(encoding="utf-8")):
            offenders.append(rel)
    assert not offenders, f"these modules reference hidden truth: {offenders}"
