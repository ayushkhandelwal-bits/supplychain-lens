"""Everything the tools need, loaded once (the dashboard caches one Context per session)."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import pandas as pd

from supplychain_lens.config import Settings, get_settings
from supplychain_lens.serving import registry


@dataclass
class Context:
    settings: Settings = field(default_factory=get_settings)

    @property
    def data_dir(self) -> Path:
        return self.settings.data_dir / "nexus_v2"

    def table(self, name: str) -> pd.DataFrame:
        if name not in self._tables:
            self._tables[name] = pd.read_parquet(self.data_dir / f"{name}.parquet")
        return self._tables[name]

    @cached_property
    def _tables(self) -> dict:
        return {}

    def run(self, model: str) -> registry.Run | None:
        if model not in self._runs:
            self._runs[model] = registry.load_run(model)
        return self._runs[model]

    @cached_property
    def _runs(self) -> dict:
        return {}

    # Snapshot artefacts -------------------------------------------------------
    @cached_property
    def snapshot_meta(self) -> dict:
        return self.run("snapshot").json("meta.json")

    @cached_property
    def positions(self) -> pd.DataFrame:
        return pd.read_parquet(self.run("snapshot").file("positions.parquet"))

    @cached_property
    def forecast(self) -> pd.DataFrame:
        return pd.read_parquet(self.run("snapshot").file("forecast.parquet"))

    @cached_property
    def scenarios(self) -> dict:
        return self.run("snapshot").json("scenarios.json")

    @cached_property
    def risk_scores(self) -> pd.DataFrame:
        return pd.read_parquet(self.run("risk_classifier").file("current_scores.parquet"))

    @cached_property
    def risk_metrics(self) -> dict:
        return self.run("risk_classifier").json("metrics.json")

    @cached_property
    def anomaly_alerts(self) -> pd.DataFrame:
        return pd.read_parquet(self.run("anomaly_autoencoder").file("test_alerts.parquet"))

    @cached_property
    def comparison(self) -> dict:
        return self.run("forecast_comparison").json("comparison.json")

    @cached_property
    def impact(self) -> dict | None:
        r = self.run("business_impact")
        return r.json("results.json") if r else None

    @property
    def snapshot_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.snapshot_meta["snapshot_date"])

    @cached_property
    def twin_state(self):
        from supplychain_lens.simulation.snapshot import SCENARIO_DAYS
        from supplychain_lens.simulation.twin import build_state

        return build_state(self.snapshot_date + pd.Timedelta(days=1), SCENARIO_DAYS,
                           data_dir=self.data_dir, known_arrivals=False)

    @cached_property
    def twin_quantiles(self):
        """(S, H, 3) snapshot forecast aligned to the twin's series order."""
        import numpy as np

        k = self.twin_state.keys[["sku_id", "warehouse_id"]]
        f = self.forecast.merge(k.reset_index(names="i"), on=["sku_id", "warehouse_id"]).sort_values(["i", "h"])
        S, H = len(k), int(f.h.max())
        return np.stack([f[c].to_numpy().reshape(S, H) for c in ("p10", "p50", "p90")], axis=-1)
