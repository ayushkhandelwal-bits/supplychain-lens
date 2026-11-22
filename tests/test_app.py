"""The Streamlit app starts, and every page renders without an exception."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
VIEWS = sorted(p.stem for p in (ROOT / "app" / "views").glob("*.py") if p.stem != "__init__")


def _has_artifacts() -> bool:
    from supplychain_lens.serving import registry
    return all(registry.latest_version(m) for m in ("forecast_comparison", "business_impact", "snapshot", "risk_classifier"))


def test_home_page_renders():
    testing = pytest.importorskip("streamlit.testing.v1")
    if not _has_artifacts():
        pytest.skip("run `python -m pipelines.build_all` first")
    at = testing.AppTest.from_file(str(ROOT / "app" / "Home.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    assert any("Supply Chain Lens" in m.value for m in at.markdown)


@pytest.mark.parametrize("view", VIEWS)
def test_every_page_renders(view):
    testing = pytest.importorskip("streamlit.testing.v1")
    if not _has_artifacts():
        pytest.skip("run `python -m pipelines.build_all` first")
    harness = f"""
import sys, runpy
sys.path[:0] = [{str(ROOT)!r}, {str(ROOT / 'app')!r}]
from lib import ui
ui.setup()
runpy.run_path({str(ROOT / 'app' / 'views' / (view + '.py'))!r}, run_name='__main__')
"""
    at = testing.AppTest.from_string(harness, default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
