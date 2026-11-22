from datetime import date

import pytest

from supplychain_lens.config import PROJECT_ROOT, Settings, get_settings


def test_default_calendar_and_split_are_chronological():
    s = get_settings()
    assert s.calendar_start == date(2023, 1, 1)
    assert s.calendar_end == date(2025, 12, 31)
    assert s.calendar_start < s.train_end < s.val_end < s.test_start <= s.calendar_end
    assert s.test_start == date(2025, 10, 1)  # Q4 2025 = peak season held out
    assert s.n_days == 1096  # 2024 is a leap year


def test_default_seed_matches_nexus():
    assert get_settings().seed == 42


def test_paths_live_inside_the_repository():
    s = get_settings()
    for p in (s.data_dir, s.artifacts_dir, s.reports_dir):
        assert PROJECT_ROOT in p.parents


def test_environment_override(monkeypatch):
    monkeypatch.setenv("SCL_SEED", "7")
    assert get_settings().seed == 7


def test_rejects_overlapping_split():
    with pytest.raises(ValueError):
        Settings(train_end=date(2025, 10, 1), val_end=date(2025, 9, 30))


def test_ollama_is_optional(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert get_settings().ollama_host is None
