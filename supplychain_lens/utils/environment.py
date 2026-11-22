"""Describe the machine and installed packages.

Used by the smoke test (to size models for your hardware) and by the dashboard's
model-health panel (to show which runtime is serving the models).
"""

from __future__ import annotations

import os
import platform
import sys
from importlib import metadata

PACKAGES = (
    "torch",
    "onnxruntime",
    "onnx",
    "numpy",
    "pandas",
    "streamlit",
    "plotly",
    "scikit-learn",
    "statsmodels",
)


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def total_ram_gb() -> float | None:
    """Total physical memory in GB, or None if it can't be determined."""
    try:
        import psutil
    except ImportError:
        pass
    else:
        return round(psutil.virtual_memory().total / 1024**3, 1)
    if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names:  # Linux / macOS fallback
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
    return None


def gpu_summary() -> str:
    """Name of a usable GPU, or 'none'. The project never requires one."""
    try:
        import torch
    except ImportError:
        return "unknown (PyTorch not installed)"
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "Apple MPS"
    return "none"


def environment_report() -> dict:
    """A JSON-serialisable snapshot of the environment."""
    return {
        "python": sys.version.split()[0],
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "logical_cpus": os.cpu_count(),
        "ram_gb": total_ram_gb(),
        "gpu": gpu_summary(),
        "packages": {name: package_version(name) for name in PACKAGES},
    }
