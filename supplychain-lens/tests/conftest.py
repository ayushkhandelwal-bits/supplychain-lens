"""Shared pytest setup.

Tests marked ``torch`` need the training environment. They are skipped automatically
when PyTorch isn't installed, so the same suite runs in the inference environment.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HAS_TORCH = importlib.util.find_spec("torch") is not None


def pytest_collection_modifyitems(config, items):
    if HAS_TORCH:
        return
    skip = pytest.mark.skip(reason="PyTorch not installed (inference environment)")
    for item in items:
        if "torch" in item.keywords:
            item.add_marker(skip)
