"""SupplyChain Lens — 30-second hardware smoke test.

Run from the repository root (training environment activated):

    python scripts/smoke_test.py            # full check, about 30 s
    python scripts/smoke_test.py --quick    # CI / very slow machines, a few seconds

Writes reports/hardware_profile.json, which Phases 5–6 read to pick model sizes.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # allow running without `pip install -e .`

from supplychain_lens import PROJECT_NAME, __version__  # noqa: E402
from supplychain_lens.config import get_settings  # noqa: E402
from supplychain_lens.utils.environment import environment_report  # noqa: E402
from supplychain_lens.utils.seed import set_global_seed  # noqa: E402


def line(label: str, value: object = "") -> None:
    print(f"  {label:<34}{value}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles: print symbols safely
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="SupplyChain Lens hardware smoke test")
    parser.add_argument("--quick", action="store_true", help="tiny run for CI (timing not representative)")
    parser.add_argument("--out", default=None, help="output JSON (default: reports/hardware_profile.json)")
    args = parser.parse_args()

    settings = get_settings()
    out_path = Path(args.out) if args.out else settings.reports_dir / "hardware_profile.json"

    print(f"\n{PROJECT_NAME} {__version__} — hardware smoke test\n")
    env = environment_report()
    print("1. Environment")
    for key in ("python", "os", "processor", "logical_cpus", "ram_gb", "gpu"):
        line(key, env[key])
    for name in ("torch", "onnxruntime", "onnx"):
        line(name, env["packages"].get(name) or "NOT INSTALLED")

    missing = [n for n in ("torch", "onnxruntime", "onnx") if not env["packages"].get(n)]
    if missing:
        print(f"\n✗ Missing packages: {', '.join(missing)}. See docs/setup_windows.md, step 4.")
        return 1

    import torch

    from supplychain_lens.utils import smoke

    cfg = smoke.SmokeConfig()
    if args.quick:
        cfg = replace(cfg, n_series=16, n_days=150, train_windows=2_000)

    rng = set_global_seed(cfg.seed)
    torch.set_num_threads(max(1, env["logical_cpus"] or 1))

    print("\n2. Build synthetic seasonal windows (not project data)")
    t0 = time.perf_counter()
    data = smoke.make_windows(cfg, rng)
    line("windows", f"{len(data[0]):,}  (lookback {cfg.lookback}, horizon {cfg.horizon})")
    line("seconds", f"{time.perf_counter() - t0:.2f}")

    print("\n3. Train a seq2seq LSTM with quantile loss")
    model = smoke.build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    line("parameters", f"{n_params:,}")
    smoke.train(model, data, cfg, n_windows=cfg.batch_size * 4, rng=rng)  # warm-up, not timed
    seconds, first, last = smoke.train(model, data, cfg, n_windows=cfg.train_windows, rng=rng)
    sec_per_20k = seconds * 20_000 / cfg.train_windows
    learned = last < first
    line("training seconds", f"{seconds:.1f}  ({cfg.train_windows:,} windows)")
    line("seconds per 20k windows", f"{sec_per_20k:.1f}  (reference: {smoke.REFERENCE_SEC_PER_20K})")
    line(
        "pinball loss first → last",
        f"{first:.3f} → {last:.3f}  {'✓ learning' if learned else '✗ not decreasing'}",
    )

    print("\n4. Export to ONNX and compare with PyTorch")
    with tempfile.TemporaryDirectory() as tmp:
        onnx = smoke.export_and_check_onnx(model, data, Path(tmp) / "smoke.onnx")
    parity_ok = onnx["max_abs_diff"] < 1e-4
    line("exporter", onnx["exporter"])
    line("file size", f"{onnx['file_kb']} KB")
    line("batch sizes checked", onnx["batch_sizes_checked"])
    line("max |ONNX − PyTorch|", f"{onnx['max_abs_diff']:.2e}  {'✓' if parity_ok else '✗'}")
    line("ONNX Runtime latency (batch 64)", f"{onnx['ort_latency_ms_batch64']} ms")

    rec = smoke.sizing_recommendation(sec_per_20k)
    print("\n5. Sizing for Phases 5–6")
    if args.quick:
        print("  (quick mode: timing not representative — run without --quick on your laptop)")
    line("speed vs reference machine", f"{rec['relative_to_reference']}× the time  (lower is faster)")
    line("LSTM, 30 epochs (estimate)", f"~{rec['est_lstm_30_epochs_min']} min")
    line("TFT, 30 epochs (estimate)", f"~{rec['est_tft_30_epochs_min']} min")
    line("preset", f"{rec['preset']} — {rec['note']}")

    ok = learned and parity_ok
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project": PROJECT_NAME,
        "version": __version__,
        "quick_mode": args.quick,
        "environment": env,
        "config": asdict(cfg),
        "parameters": n_params,
        "training": {
            "seconds": round(seconds, 2),
            "sec_per_20k_windows": round(sec_per_20k, 2),
            "loss_first": round(first, 4),
            "loss_last": round(last, 4),
            "learned": learned,
        },
        "onnx": onnx,
        "sizing": rec,
        "reference_machine": smoke.REFERENCE_MACHINE,
        "passed": ok,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'✓ PASSED' if ok else '✗ FAILED'} — report written to {out_path}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
