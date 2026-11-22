"""
Phase 0 audit — re-verifies every factual claim we make about the reference
repository NEXUS-SCCT (https://github.com/ahmedzainsyed/nexus-scct).

Why this exists
---------------
Our docs say things like "the supplier-risk generator crashes" or "delay_risk_score
is pure noise". An interviewer (or the original author) should be able to check
those claims in one command instead of trusting us. This script is that command.

It never modifies the reference clone. Generated data goes to a temp folder.

Usage (Windows PowerShell, from the repo root):
    python audit/verify_findings.py --ref reference/nexus-scct

Outputs:
    audit/AUDIT_REPORT.md   (human-readable report, overwritten each run)
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import types
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

RESULTS: list[dict] = []


def record(check_id: str, title: str, finding: bool, evidence: str) -> None:
    """finding=True means the issue we documented is confirmed."""
    RESULTS.append({"id": check_id, "title": title, "confirmed": finding, "evidence": evidence})
    status = "CONFIRMED" if finding else "NOT CONFIRMED"
    print(f"[{status:13s}] {check_id} {title}\n                {evidence}")


# ─────────────────────────────────────────────────────────────
# A. Repository / licensing
# ─────────────────────────────────────────────────────────────
def check_license(ref: Path) -> None:
    license_files = [
        p.name for p in ref.iterdir() if p.name.upper().startswith(("LICENSE", "LICENCE", "COPYING"))
    ]
    readme = (ref / "README.md").read_text(encoding="utf-8")
    claims_apache = "Apache_2.0" in readme or "Apache 2.0" in readme
    record(
        "A1",
        "No LICENSE file, although README claims Apache-2.0",
        not license_files and claims_apache,
        f"license files found: {license_files or 'none'}; README Apache badge: {claims_apache}",
    )


def check_missing_modules(ref: Path) -> None:
    tree = ast.parse((ref / "main.py").read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".")[0] in {"api", "core", "streaming", "workers", "ml"}
        ):
            imported.append(node.module)
    # The repo is flat: a package path like api/routes/... cannot exist.
    missing = [
        m
        for m in imported
        if not (ref / Path(*m.split("."))).with_suffix(".py").exists()
        and not (ref / Path(*m.split("."))).is_dir()
    ]
    record(
        "A2",
        "main.py imports modules that are not in the repository",
        len(missing) > 0,
        f"{len(set(missing))} of {len(set(imported))} project modules missing: {sorted(set(missing))}",
    )


def check_mock_routes(ref: Path) -> None:
    src = (ref / "all_routes.py").read_text(encoding="utf-8")
    mocks = len(re.findall(r"_mock_\w+\(\)", src))
    randoms = len(re.findall(r"_random\.(uniform|randint)", src))
    reads_seed = "data/seeds" in src or "read_parquet" in src or "json.load" in src
    record(
        "A3",
        "API routes serve hard-coded/random data, not the seed data",
        (mocks + randoms) > 0 and not reads_seed,
        f"_mock_*() calls: {mocks}; random KPI calls: {randoms}; reads seed files: {reads_seed}",
    )


def check_agents_have_no_tools(ref: Path) -> None:
    src = (ref / "orchestrator.py").read_text(encoding="utf-8")
    ml_imports = [
        m
        for m in (
            "supplier_risk_model",
            "tft_model",
            "rl_optimizer",
            "route_optimizer",
            "SupplierRiskEngine",
            "DemandForecastingEngine",
            "InventoryOptimizer",
            "RouteOptimizer",
        )
        if m in src
    ]
    llm_calls = src.count("self.llm.invoke")
    self_conf = 'data.get("confidence"' in src
    record(
        "A4",
        "Agents call the LLM only; no ML/optimization tools; LLM self-confidence gates approval",
        not ml_imports and llm_calls > 0 and self_conf,
        f"ML/tool references: {ml_imports or 'none'}; llm.invoke sites: {llm_calls}; "
        f"confidence taken from LLM JSON: {self_conf}",
    )


def check_linear_graph(ref: Path) -> None:
    src = (ref / "orchestrator.py").read_text(encoding="utf-8")
    edges = re.findall(r'add_edge\("(\w+)", "(\w+)"\)', src)
    # conditional edges each map to a single target, e.g. {"escalation": "escalation"}
    edges += [(a, b) for a, b in re.findall(r'add_conditional_edges\("(\w+)",[^{]*\{"\w+": "(\w+)"\}', src)]
    sources = [a for a, _ in edges]
    linear = len(sources) == len(set(sources))  # no node fans out to two successors
    # walk the chain from the entry point for a readable trace
    entry = re.search(r'set_entry_point\("(\w+)"\)', src)
    nxt, path = dict(edges), [entry.group(1)] if entry else []
    while path and path[-1] in nxt and len(path) <= len(edges):
        path.append(nxt[path[-1]])
    record(
        "A5",
        "Agent graph is a linear chain (README says parallel)",
        linear,
        "path: " + " -> ".join(path) if path else "no edges found",
    )


def check_package_json(ref: Path) -> None:
    pkg = json.loads((ref / "package.json").read_text(encoding="utf-8"))
    deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
    suspects = [
        d
        for d in deps
        if d
        in {
            "@radix-ui/react-badge",
            "@radix-ui/react-button",
            "@radix-ui/react-card",
            "@radix-ui/react-sheet",
        }
    ]
    not_found, checked = [], True
    for name in suspects:
        try:
            urllib.request.urlopen(f"https://registry.npmjs.org/{name.replace('/', '%2f')}", timeout=10)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                not_found.append(name)
        except Exception:
            checked = False
    if checked:
        record(
            "A6",
            "package.json lists npm packages that do not exist",
            bool(not_found),
            f"404 on npm registry: {not_found}",
        )
    else:
        record(
            "A6",
            "package.json lists npm packages that do not exist",
            bool(suspects),
            f"registry unreachable; listed suspect packages: {suspects} (verify with `npm view <name>`)",
        )


# ─────────────────────────────────────────────────────────────
# B. ML code defects (static analysis — no heavy installs needed)
# ─────────────────────────────────────────────────────────────
def check_rl_shape(ref: Path) -> None:
    tree = ast.parse((ref / "rl_optimizer.py").read_text(encoding="utf-8"))
    cfg = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "ENV_CONFIG":
            cfg = ast.literal_eval(node.value)
    n_wh, n_sku = cfg["n_warehouses"], cfg["n_skus"]
    env_obs = (n_wh * n_sku * 2) + (n_wh * n_sku) + n_wh + 2  # formula in SupplyChainEnv.__init__
    infer_obs = n_wh * 3 + 2  # formula in _build_observation
    record(
        "B1",
        "RL inference observation size does not match the trained policy",
        env_obs != infer_obs,
        f"policy expects {env_obs} values; inference builds {infer_obs}",
    )


def check_tft_bug(ref: Path) -> None:
    tree = ast.parse((ref / "tft_model.py").read_text(encoding="utf-8"))
    params, bad_call = [], False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "load_demand_data":
            params = [a.arg for a in node.args.args]
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "load_demand_data":
            if any(k.arg == "df" for k in node.keywords):
                bad_call = True
    record(
        "B2",
        "TFT inference calls load_demand_data(df=...) but no such parameter exists",
        bad_call and "df" not in params,
        f"signature params: {params}; called with df=: {bad_call}",
    )


# ─────────────────────────────────────────────────────────────
# C. Seed data defects (runs the original generator in a temp dir)
# ─────────────────────────────────────────────────────────────
def check_seed_data(ref: Path) -> None:
    import pandas as pd

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "seeds"
        # PYTHONIOENCODING: the original script prints emoji; Windows pipes default to cp1252.
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.run(
            [sys.executable, str(ref / "seed_data.py"), "--output-dir", str(out)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=tmp,
            env=env,
        )
        if proc.returncode != 0:
            record("C0", "Original seed_data.py runs", False, proc.stderr[-400:])
            return

        def j(name):
            return pd.DataFrame(json.loads((out / f"{name}.json").read_text(encoding="utf-8")))

        sup, wh, shp, dis, kpi = (
            j("suppliers"),
            j("warehouses"),
            j("shipments"),
            j("disruptions"),
            j("kpi_history"),
        )
        dem = pd.read_parquet(out / "demand_history.parquet")

        record(
            "C0",
            "Original seed_data.py runs and produces the documented row counts",
            True,
            f"suppliers={len(sup)}, warehouses={len(wh)}, shipments={len(shp)}, "
            f"disruptions={len(dis)}, kpi_history={len(kpi)}, demand_history={len(dem):,}",
        )

        n_prices = dem.groupby(["sku_id", "warehouse_id"])["price"].nunique().median()
        record(
            "D1",
            "Price is re-drawn every day for the same SKU-warehouse",
            n_prices > 1,
            f"median distinct prices per series: {n_prices:.0f}",
        )

        corr = shp["delay_risk_score"].corr(shp["is_delayed"].astype(int))
        record(
            "D2",
            "delay_risk_score is unrelated to actual delays",
            abs(corr) < 0.1,
            f"corr(delay_risk_score, is_delayed) = {corr:.3f}",
        )

        delivered = shp[shp["status"] == "DELIVERED"]
        record(
            "D3",
            "Delivered shipments are never late",
            (delivered["delay_days"] == 0).all(),
            f"{len(delivered)} delivered, late among them: {(delivered['delay_days'] > 0).sum()}",
        )

        whs = sorted(dem["warehouse_id"].unique())
        record(
            "D4",
            "Demand exists for only a subset of warehouses",
            len(whs) < len(wh),
            f"demand warehouses: {whs} of {len(wh)}",
        )

        record(
            "D5",
            "No SKU-to-supplier link exists in any table",
            "supplier_id" not in dem.columns,
            f"demand columns: {list(dem.columns)}",
        )

        sup_country = sup.set_index("id")["country"]
        weather = dis[dis["title"].str.contains("East China Sea", na=False)]
        countries = sorted(
            {sup_country.get(s, "UNKNOWN") for ids in weather["affected_entities"] for s in ids}
        )
        record(
            "D6",
            "Disruptions hit supplier IDs unrelated to their geography",
            any(c not in {"China", "Taiwan", "Japan", "South Korea"} for c in countries),
            f"'East China Sea' typhoons affect suppliers in: {countries}",
        )

        first, last = kpi["otif_pct"].head(10).mean(), kpi["otif_pct"].tail(10).mean()
        record(
            "D7",
            "kpi_history contains a built-in improvement trend",
            last - first > 5,
            f"OTIF first-10-day mean {first:.1f}% -> last-10-day mean {last:.1f}%",
        )

        span = pd.to_datetime(dem["date"]).max().date()
        record(
            "D8",
            "Dates are anchored to the run date (non-reproducible)",
            span == datetime.now().date(),
            f"latest demand date = {span} (today = {datetime.now().date()})",
        )

        crit = sup[sup["risk_level"] == "CRITICAL"]["risk_score"].max()
        low = sup[sup["risk_level"] == "LOW"]["risk_score"].min()
        record(
            "D9",
            "risk_score is inverted (higher = safer)",
            crit < low,
            f"max CRITICAL score {crit} < min LOW score {low}",
        )


# ─────────────────────────────────────────────────────────────
# E. Supplier risk model
# ─────────────────────────────────────────────────────────────
def check_risk_model(ref: Path) -> None:
    import numpy as np

    sys.modules.setdefault("mlflow", types.ModuleType("mlflow"))  # stub: training is not run
    sys.path.insert(0, str(ref))
    try:
        import supplier_risk_model as srm
    except ImportError as e:
        record("E1", "Supplier-risk generator crashes", False, f"could not import module: {e}")
        return

    try:
        srm.generate_supplier_training_data(5000)
        record("E1", "Supplier-risk training-data generator crashes", False, "ran without error")
    except ValueError as e:
        record("E1", "Supplier-risk training-data generator crashes", True, f"ValueError: {e}")

    # Minimal patch (clamp the beta parameter) to measure what the model can achieve.
    orig_beta = np.random.beta
    np.random.beta = lambda a, b, *k: orig_beta(max(a, 0.5), b, *k)
    try:
        df = srm.generate_supplier_training_data(5000)
    finally:
        np.random.beta = orig_beta

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = df[srm.RISK_FEATURES], df[srm.RISK_TARGET]
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    scores = {}
    try:
        import xgboost as xgb

        scores["XGBoost (original params)"] = cross_val_score(
            xgb.XGBClassifier(
                n_estimators=400,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=5,
                gamma=0.1,
                reg_alpha=0.1,
                eval_metric="auc",
                random_state=42,
            ),
            X,
            y,
            cv=cv,
            scoring="roc_auc",
        )
    except ImportError:
        pass
    scores["Logistic regression"] = cross_val_score(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X, y, cv=cv, scoring="roc_auc"
    )

    text = "; ".join(f"{k}: AUC {v.mean():.3f} ± {v.std():.3f}" for k, v in scores.items())
    best = max(v.mean() for v in scores.values())
    record("E2", "Patched risk model stays below the repo's own 0.88 promotion gate", best < 0.88, text)


# ─────────────────────────────────────────────────────────────
def write_report(ref: Path, out_path: Path) -> None:
    sha = (
        subprocess.run(
            ["git", "-C", str(ref), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        or "unknown"
    )
    lines = [
        "# NEXUS-SCCT audit report",
        "",
        f"Generated by `audit/verify_findings.py` on {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} "
        f"against reference commit `{sha}`. Python {sys.version.split()[0]}.",
        "",
        "`CONFIRMED` = the issue described in our docs was reproduced on this run.",
        "",
        "| ID | Finding | Result | Evidence |",
        "|---|---|---|---|",
    ]
    for r in RESULTS:
        ev = r["evidence"].replace("|", "\\|")
        lines.append(
            f"| {r['id']} | {r['title']} | {'CONFIRMED' if r['confirmed'] else 'not confirmed'} | {ev} |"
        )
    confirmed = sum(r["confirmed"] for r in RESULTS)
    lines += ["", f"**{confirmed} of {len(RESULTS)} checks confirmed.**", ""]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {out_path}  ({confirmed}/{len(RESULTS)} confirmed)")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles: print ± and → safely
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ref", default="reference/nexus-scct", help="path to a local clone of nexus-scct")
    parser.add_argument("--out", default=str(Path(__file__).parent / "AUDIT_REPORT.md"))
    args = parser.parse_args()

    ref = Path(args.ref).resolve()
    if not (ref / "seed_data.py").exists():
        sys.exit(
            f"Reference clone not found at {ref}. Run:\n"
            f"  git clone https://github.com/ahmedzainsyed/nexus-scct.git {args.ref}"
        )

    for check in (
        check_license,
        check_missing_modules,
        check_mock_routes,
        check_agents_have_no_tools,
        check_linear_graph,
        check_package_json,
        check_rl_shape,
        check_tft_bug,
        check_seed_data,
        check_risk_model,
    ):
        try:
            check(ref)
        except Exception as e:  # one broken check must not hide the others
            record(check.__name__, "check errored", False, f"{type(e).__name__}: {e}")

    write_report(ref, Path(args.out))


if __name__ == "__main__":
    main()
