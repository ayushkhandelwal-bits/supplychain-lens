"""Deep-learning plumbing: window shapes, a model that learns, and ONNX parity."""

from dataclasses import replace

import numpy as np
import pytest

from supplychain_lens.utils import smoke

CFG = replace(smoke.SmokeConfig(), n_series=8, n_days=120, train_windows=1_024, batch_size=128)


def test_window_shapes_and_no_overlap_between_past_and_future():
    past, fut, sid, tgt = smoke.make_windows(CFG, np.random.default_rng(0))
    n = CFG.n_series * (CFG.n_days - CFG.lookback - CFG.horizon + 1)
    assert past.shape == (n, CFG.lookback, 3)
    assert fut.shape == (n, CFG.horizon, 2)
    assert tgt.shape == (n, CFG.horizon)
    assert sid.shape == (n,) and sid.max() == CFG.n_series - 1
    # The target must be the values right AFTER the past window, never inside it.
    past2, _, _, tgt2 = smoke.make_windows(CFG, np.random.default_rng(0))
    assert past2[1, -1, 0] == tgt2[0, 0]  # window i+1 ends where window i's target starts


def test_sizing_recommendation_scales_with_speed():
    fast = smoke.sizing_recommendation(smoke.REFERENCE_SEC_PER_20K / 2)
    slow = smoke.sizing_recommendation(smoke.REFERENCE_SEC_PER_20K * 10)
    assert fast["est_tft_30_epochs_min"] < slow["est_tft_30_epochs_min"]
    assert fast["preset"] == "standard" and slow["preset"] == "light"


@pytest.mark.torch
def test_model_output_shape_and_learning():
    import torch

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    data = smoke.make_windows(CFG, rng)
    model = smoke.build_model(CFG)
    with torch.no_grad():
        out = model(*(torch.from_numpy(a[:5]) for a in data[:3]))
    assert out.shape == (5, CFG.horizon, 3)
    _, first, last = smoke.train(model, data, CFG, n_windows=CFG.train_windows, rng=rng)
    assert last < first


@pytest.mark.torch
def test_onnx_matches_pytorch_at_several_batch_sizes(tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    data = smoke.make_windows(CFG, np.random.default_rng(1))
    model = smoke.build_model(CFG)
    result = smoke.export_and_check_onnx(model, data, tmp_path / "m.onnx", batch_sizes=(1, 3, 32))
    assert result["max_abs_diff"] < 1e-4
    assert list(tmp_path.iterdir()) == [tmp_path / "m.onnx"]  # single self-contained file
