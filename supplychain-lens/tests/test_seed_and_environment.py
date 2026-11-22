import random

import numpy as np
import pytest

from supplychain_lens.utils.environment import environment_report
from supplychain_lens.utils.seed import set_global_seed


def test_same_seed_same_numbers():
    rng_a = set_global_seed(123)
    a = (random.random(), np.random.rand(), rng_a.normal(size=3))
    rng_b = set_global_seed(123)
    b = (random.random(), np.random.rand(), rng_b.normal(size=3))
    assert a[0] == b[0] and a[1] == b[1]
    np.testing.assert_array_equal(a[2], b[2])


def test_different_seed_different_numbers():
    assert set_global_seed(1).random() != set_global_seed(2).random()


@pytest.mark.torch
def test_seed_controls_torch():
    import torch

    set_global_seed(5)
    a = torch.randn(4)
    set_global_seed(5)
    assert torch.equal(a, torch.randn(4))


def test_environment_report_shape():
    env = environment_report()
    for key in ("python", "os", "logical_cpus", "ram_gb", "gpu", "packages"):
        assert key in env
    assert env["packages"]["numpy"] is not None
