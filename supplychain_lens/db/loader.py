# Design adapted from NEXUS-SCCT (core_modules.py ORM tables), reimplemented for SQLite.
"""Build the SQLite database from the generated Parquet files.

SQLite replaces NEXUS's PostgreSQL (ADR-0002): one file, no server, plain SQL. The
database is a *derived* artefact — it can always be rebuilt from Parquet in a few
seconds, which is what the hosted app does on first start.

    python -m supplychain_lens.db.loader
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pandas as pd

from supplychain_lens.config import get_settings

# Indexes for the queries the app and the agent tools run.
INDEXES = {
    "demand_daily": [("sku_id", "warehouse_id", "date"), ("date",)],
    "inventory_daily": [("sku_id", "warehouse_id", "date"), ("date",)],
    "warehouse_daily": [("warehouse_id", "date")],
    "purchase_orders": [("supplier_id", "order_date"), ("sku_id", "warehouse_id")],
    "shipments": [("supplier_id",), ("status",), ("warehouse_id",)],
    "disruption_suppliers": [("supplier_id",), ("event_id",)],
    "disruptions": [("status",)],
}

# Tables the app needs; truth tables are deliberately NOT loaded into the app database.
TABLES = ["suppliers", "warehouses", "products", "sku_warehouse", "promotions", "calendar", "demand_daily",
          "inventory_daily", "warehouse_daily", "purchase_orders", "shipments", "disruptions",
          "disruption_suppliers", "kpi_daily"]


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")  # ISO dates sort and compare correctly as text
        elif pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].astype(int)
    return out


def build_sqlite(data_dir: Path, db_path: Path) -> dict:
    """(Re)build the database. Returns row counts and build time."""
    t0 = time.perf_counter()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_suffix(".tmp")
    if tmp.exists():
        tmp.unlink()
    counts = {}
    with sqlite3.connect(tmp) as con:
        for name in TABLES:
            df = _prepare(pd.read_parquet(data_dir / f"{name}.parquet"))
            df.to_sql(name, con, index=False, chunksize=50_000)
            counts[name] = len(df)
            for cols in INDEXES.get(name, []):
                con.execute(f"CREATE INDEX idx_{name}_{'_'.join(cols)} ON {name} ({', '.join(cols)})")
        con.execute("ANALYZE")
    tmp.replace(db_path)  # atomic swap: a half-built database is never visible
    return {"rows": counts, "seconds": round(time.perf_counter() - t0, 2), "size_mb": round(db_path.stat().st_size / 1e6, 1)}


def ensure_database(data_dir: Path | None = None, db_path: Path | None = None) -> Path:
    """Return the database path, building it first if missing or older than the data."""
    s = get_settings()
    data_dir = data_dir or s.data_dir / "nexus_v2"
    db_path = db_path or s.data_dir / "supplychain_lens.db"
    manifest = data_dir / "manifest.json"
    if not db_path.exists() or (manifest.exists() and manifest.stat().st_mtime > db_path.stat().st_mtime):
        build_sqlite(data_dir, db_path)
    return db_path


if __name__ == "__main__":
    s = get_settings()
    info = build_sqlite(s.data_dir / "nexus_v2", s.data_dir / "supplychain_lens.db")
    print(f"Built {s.data_dir / 'supplychain_lens.db'}: {info['size_mb']} MB in {info['seconds']} s")
