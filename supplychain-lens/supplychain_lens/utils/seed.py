"""One call to make a run reproducible.

Seeds Python's ``random``, NumPy and (if installed) PyTorch. PyTorch is optional so
this module also works in the inference environment, which has no PyTorch (ADR-0009).
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int, deterministic_torch: bool = False) -> np.random.Generator:
    """Seed every random number generator the project uses.

    Args:
        seed: The seed. The project default is 42, the same seed NEXUS-SCCT used.
        deterministic_torch: Also ask PyTorch to use deterministic algorithms. Slower,
            and some operations raise an error if no deterministic version exists.

    Returns:
        A NumPy ``Generator`` seeded with ``seed``. New code should draw from this
        generator rather than from the global ``np.random`` state.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)  # affects subprocesses started after this
    random.seed(seed)
    np.random.seed(seed)  # legacy global state, for libraries that still use it

    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)  # also seeds CUDA devices if any
        if deterministic_torch:
            torch.use_deterministic_algorithms(True)

    return np.random.default_rng(seed)
