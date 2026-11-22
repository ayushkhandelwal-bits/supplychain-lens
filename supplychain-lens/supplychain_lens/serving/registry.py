"""Artifact registry: versioned folders that hold everything a trained model produced.

Replaces NEXUS's MLflow registry (ADR-0009). Layout::

    artifacts/<model>/<version>/
        model.onnx  model.pt  config.json  scaler.json  metrics.json
        loss_curve.csv  test_predictions.parquet  model_card.md  SHA256SUMS
    artifacts/<model>/LATEST          # text file with the current version

The dashboard reads ONLY these files, so every number it shows traces back to a run.
`load_run()` verifies the SHA-256 checksums before anything is used.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from supplychain_lens.config import get_settings


class ArtifactError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Run:
    model: str
    version: str
    path: Path

    def file(self, name: str) -> Path:
        p = self.path / name
        if not p.exists():
            raise ArtifactError(f"{self.model}/{self.version} has no {name}")
        return p

    def has(self, name: str) -> bool:
        return (self.path / name).exists()

    def json(self, name: str) -> dict:
        return json.loads(self.file(name).read_text(encoding="utf-8"))


def new_version() -> str:
    return datetime.now(timezone.utc).strftime("v%Y%m%d-%H%M%S")


def start_run(model: str, root: Path | None = None, version: str | None = None) -> Run:
    root = root or get_settings().artifacts_dir
    version = version or new_version()
    path = root / model / version
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return Run(model, version, path)


def write_json(run: Run, name: str, payload: dict) -> Path:
    p = run.path / name
    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return p


def finalize(run: Run, make_latest: bool = True) -> Run:
    """Write SHA256SUMS for every file and (optionally) point LATEST at this version."""
    files = sorted(p for p in run.path.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    (run.path / "SHA256SUMS").write_text("".join(f"{_sha256(p)}  {p.name}\n" for p in files), encoding="utf-8")
    if make_latest:
        (run.path.parent / "LATEST").write_text(run.version + "\n", encoding="utf-8")
    return run


def verify(run: Run) -> None:
    sums = run.file("SHA256SUMS").read_text(encoding="utf-8").splitlines()
    for line in sums:
        digest, name = line.split("  ", 1)
        p = run.path / name
        if not p.exists() or _sha256(p) != digest:
            raise ArtifactError(f"checksum mismatch for {run.model}/{run.version}/{name}")


def latest_version(model: str, root: Path | None = None) -> str | None:
    root = root or get_settings().artifacts_dir
    pointer = root / model / "LATEST"
    return pointer.read_text(encoding="utf-8").strip() if pointer.exists() else None


def load_run(model: str, version: str | None = None, root: Path | None = None, check: bool = True) -> Run | None:
    """Return the run (verified) or None if the model hasn't been trained yet."""
    root = root or get_settings().artifacts_dir
    version = version or latest_version(model, root)
    if version is None or not (root / model / version).exists():
        return None
    run = Run(model, version, root / model / version)
    if check:
        verify(run)
    return run


def prune(model: str, keep: int = 1, root: Path | None = None) -> None:
    """Delete old versions, keeping LATEST plus the newest ``keep - 1`` others."""
    root = root or get_settings().artifacts_dir
    latest = latest_version(model, root)
    versions = sorted((p for p in (root / model).iterdir() if p.is_dir()), reverse=True)
    kept = [v for v in versions if v.name == latest] + [v for v in versions if v.name != latest][: max(0, keep - 1)]
    for v in versions:
        if v not in kept:
            shutil.rmtree(v)
