"""Project-wide settings.

Design adapted from the settings idea in NEXUS-SCCT core_modules.py (85 environment
variables); reimplemented here as ~15 plain settings with no secrets required.

Everything that makes results reproducible lives here: the random seed, the fixed
calendar and the chronological split. Any value can be overridden with an
environment variable of the same name prefixed with ``SCL_`` (e.g. ``SCL_SEED=7``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(f"SCL_{name}", default)


def _date(name: str, default: str) -> date:
    return date.fromisoformat(_env(name, default))


@dataclass(frozen=True)
class Settings:
    # Reproducibility ---------------------------------------------------------
    seed: int = field(default_factory=lambda: int(_env("SEED", "42")))  # same seed as NEXUS seed_data.py

    # Fixed calendar (ADR-0007): 3 years of daily history, fixed "today" ------
    calendar_start: date = field(default_factory=lambda: _date("CALENDAR_START", "2023-01-01"))
    calendar_end: date = field(default_factory=lambda: _date("CALENDAR_END", "2025-12-31"))
    snapshot_date: date = field(default_factory=lambda: _date("SNAPSHOT_DATE", "2025-12-31"))

    # Chronological split: train <= train_end < validation <= val_end < test ----
    train_end: date = field(default_factory=lambda: _date("TRAIN_END", "2025-06-30"))
    val_end: date = field(default_factory=lambda: _date("VAL_END", "2025-09-30"))

    # Forecasting shape (sized from the Phase 0b CPU benchmark) ------------------
    lookback_days: int = field(default_factory=lambda: int(_env("LOOKBACK_DAYS", "56")))
    horizon_days: int = field(default_factory=lambda: int(_env("HORIZON_DAYS", "28")))
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)

    # Paths --------------------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(PROJECT_ROOT / "data"))))
    artifacts_dir: Path = field(
        default_factory=lambda: Path(_env("ARTIFACTS_DIR", str(PROJECT_ROOT / "artifacts")))
    )
    reports_dir: Path = field(
        default_factory=lambda: Path(_env("REPORTS_DIR", str(PROJECT_ROOT / "reports")))
    )

    # Optional local LLM (ADR-0004). Unset = template narration only. -----------
    ollama_host: str | None = field(default_factory=lambda: os.environ.get("OLLAMA_HOST") or None)

    def __post_init__(self) -> None:
        if not (self.calendar_start < self.train_end < self.val_end < self.calendar_end):
            raise ValueError(
                "Dates must satisfy calendar_start < train_end < val_end < calendar_end; "
                f"got {self.calendar_start}, {self.train_end}, {self.val_end}, {self.calendar_end}"
            )
        if self.snapshot_date > self.calendar_end:
            raise ValueError("snapshot_date cannot be after calendar_end")
        if self.lookback_days <= 0 or self.horizon_days <= 0:
            raise ValueError("lookback_days and horizon_days must be positive")

    @property
    def test_start(self) -> date:
        """First day of the held-out test period (the day after val_end)."""
        return date.fromordinal(self.val_end.toordinal() + 1)

    @property
    def n_days(self) -> int:
        return (self.calendar_end - self.calendar_start).days + 1


def get_settings() -> Settings:
    """Return settings, reading environment overrides at call time."""
    return Settings()
