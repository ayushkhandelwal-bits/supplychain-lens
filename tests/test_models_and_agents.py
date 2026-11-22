"""Model architectures, the trained-artifact serving path, the approval policy and the agents.

Tests that need trained artifacts are skipped until ``python -m pipelines.build_all`` has run.
"""

from __future__ import annotations

import numpy as np
import pytest

from supplychain_lens.agents.policy import apply_policy
from supplychain_lens.agents.schemas import AgentDecision
from supplychain_lens.serving import registry

HAS_ARTIFACTS = all(registry.latest_version(m) for m in ("forecast_comparison", "risk_classifier", "snapshot"))
needs_artifacts = pytest.mark.skipif(not HAS_ARTIFACTS, reason="run `python -m pipelines.build_all` first")


def _decision(**kw) -> AgentDecision:
    base = dict(agent="t", decision_type="ORDER_ADJUSTMENT", title="t", action="a", evidence=[],
                confidence=0.95, confidence_basis="test", risk_level="MEDIUM", estimated_value_usd=10_000)
    return AgentDecision(**{**base, **kw})


def test_policy_auto_approves_only_safe_reversible_actions():
    assert apply_policy(_decision()).status == "AUTO_APPROVED"
    assert apply_policy(_decision(confidence=0.6)).requires_human_approval
    assert apply_policy(_decision(risk_level="CRITICAL")).requires_human_approval
    assert apply_policy(_decision(estimated_value_usd=6_000_000)).requires_human_approval
    assert apply_policy(_decision(reversible=False)).requires_human_approval


def test_policy_never_auto_approves_model_alerts():
    d = apply_policy(_decision(decision_type="INVESTIGATION", confidence=0.99))
    assert d.status == "INVESTIGATION_REQUIRED" and d.requires_human_approval


@pytest.mark.torch
@pytest.mark.parametrize("kind", ["lstm", "tft"])
def test_forecasters_output_ordered_quantiles_and_backprop(kind):
    import torch

    from supplychain_lens.models.lstm import LSTMConfig, LSTMForecaster
    from supplychain_lens.models.tft import TemporalFusionTransformer, TFTConfig

    torch.manual_seed(0)
    model = LSTMForecaster(LSTMConfig()) if kind == "lstm" else TemporalFusionTransformer(TFTConfig())
    B = 4
    past, future = torch.randn(B, 56, 10), torch.randn(B, 28, 9)
    codes = torch.stack([torch.randint(0, 100, (B,)), torch.randint(0, 12, (B,)), torch.randint(0, 10, (B,))], 1)
    out = model(past, future, codes, torch.zeros(B, 1))
    assert out.shape == (B, 28, 3)
    assert (out[..., 0] <= out[..., 1]).all() and (out[..., 1] <= out[..., 2]).all()
    out.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


@pytest.mark.torch
def test_autoencoder_reconstructs_input_shape():
    import torch

    from supplychain_lens.models.autoencoder import Autoencoder

    x = torch.randn(8, 36)
    assert Autoencoder(36)(x).shape == x.shape


@needs_artifacts
def test_registry_checksums_verify():
    for m in ("forecast_tft", "forecast_comparison", "anomaly_autoencoder", "risk_classifier", "business_impact"):
        run = registry.load_run(m)
        assert run is not None, m
        registry.verify(run)  # raises on any checksum mismatch


@needs_artifacts
def test_onnx_forecaster_serves_sorted_quantiles():
    from supplychain_lens.serving.forecaster import Forecaster

    fc = Forecaster.load()
    origin = len(fc.panel.dates) - fc.panel.settings.horizon_days
    q = fc.predict([origin], series=np.arange(5))
    assert q.shape == (1, 5, fc.panel.settings.horizon_days, 3)
    assert np.isfinite(q).all() and (q >= 0).all()
    assert (np.diff(q, axis=-1) >= -1e-4).all(), "P10 <= P50 <= P90 must hold"


@needs_artifacts
def test_agents_ground_every_decision_and_apply_policy():
    from supplychain_lens.agents.orchestrator import run_all
    from supplychain_lens.tools.context import Context

    res = run_all(Context())
    assert res.decisions, "agents should propose something on the snapshot date"
    for d in res.decisions:
        assert d.status in ("AUTO_APPROVED", "PENDING_APPROVAL", "INVESTIGATION_REQUIRED", "INFORMATION")
        assert 0.0 <= d.confidence <= 1.0
        if d.decision_type not in ("INFORMATION",):
            assert d.evidence, f"{d.title} has no tool evidence"
    assert len(res.executive_summary) == 4


@needs_artifacts
def test_business_impact_is_labelled_simulated():
    run = registry.load_run("business_impact")
    r = run.json("results.json")
    assert r["label"].startswith("SIMULATED")
    assert r["metrics"]["fill_rate"]["ai_mean"] > r["metrics"]["fill_rate"]["baseline_mean"]
