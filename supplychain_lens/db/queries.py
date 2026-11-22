"""Parameterised, read-only queries used by the dashboard and the agent tools.

All SQL lives here so pages and tools never build SQL strings themselves.
Every function takes an open ``sqlite3.Connection`` and returns a DataFrame.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


def connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    return con


def _q(con: sqlite3.Connection, sql: str, params: tuple | dict = (), dates: tuple[str, ...] = ()) -> pd.DataFrame:
    df = pd.read_sql_query(sql, con, params=params)
    for col in dates:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def kpi_daily(con, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    return _q(con, "SELECT * FROM kpi_daily WHERE date >= ? AND date <= ? ORDER BY date",
              (start or "0000", end or "9999"), dates=("date",))


def suppliers(con) -> pd.DataFrame:
    return _q(con, "SELECT * FROM suppliers ORDER BY supplier_id")


def warehouses(con) -> pd.DataFrame:
    return _q(con, "SELECT * FROM warehouses ORDER BY warehouse_id")


def products(con) -> pd.DataFrame:
    return _q(con, "SELECT * FROM products ORDER BY sku_id")


def series_list(con) -> pd.DataFrame:
    return _q(con, """SELECT sw.sku_id, sw.warehouse_id, p.category, p.primary_supplier_id, p.unit_cost, p.list_price
                      FROM sku_warehouse sw JOIN products p USING (sku_id) ORDER BY sw.sku_id, sw.warehouse_id""")


def demand_series(con, sku_id: str, warehouse_id: str, start: str = "0000", end: str = "9999") -> pd.DataFrame:
    return _q(con, """SELECT d.date, d.demand_units, d.sales_units, d.lost_sales_units, d.price, d.is_promo,
                             i.on_hand_units, i.on_order_units, i.receipts_units
                      FROM demand_daily d JOIN inventory_daily i USING (sku_id, warehouse_id, date)
                      WHERE d.sku_id = ? AND d.warehouse_id = ? AND d.date BETWEEN ? AND ?
                      ORDER BY d.date""", (sku_id, warehouse_id, start, end), dates=("date",))


def inventory_on(con, day: str) -> pd.DataFrame:
    return _q(con, """SELECT i.sku_id, i.warehouse_id, i.on_hand_units, i.on_order_units, p.unit_cost, p.category,
                             p.primary_supplier_id, p.backup_supplier_id
                      FROM inventory_daily i JOIN products p USING (sku_id) WHERE i.date = ?""", (day,))


def recent_demand(con, end: str, days: int) -> pd.DataFrame:
    return _q(con, """SELECT sku_id, warehouse_id, date, demand_units, sales_units, lost_sales_units
                      FROM demand_daily WHERE date > date(?, ?) AND date <= ?""",
              (end, f"-{days} days", end), dates=("date",))


def shipments(con, status: str | None = None, supplier_id: str | None = None) -> pd.DataFrame:
    sql, params = "SELECT * FROM shipments WHERE 1=1", []
    if status:
        sql += " AND status = ?"
        params.append(status)
    if supplier_id:
        sql += " AND supplier_id = ?"
        params.append(supplier_id)
    return _q(con, sql, tuple(params), dates=("ship_date", "eta", "actual_arrival"))


def open_shipments(con) -> pd.DataFrame:
    return _q(con, "SELECT * FROM shipments WHERE status IN ('IN_TRANSIT','AT_PORT','DELAYED')",
              dates=("ship_date", "eta"))


def disruptions(con, status: str | None = None) -> pd.DataFrame:
    sql = "SELECT * FROM disruptions" + (" WHERE status = ?" if status else "") + " ORDER BY detected_at DESC"
    return _q(con, sql, (status,) if status else (), dates=("detected_at", "resolved_at"))


def disruption_suppliers(con) -> pd.DataFrame:
    return _q(con, "SELECT * FROM disruption_suppliers")


def warehouse_daily(con, start: str = "0000", end: str = "9999") -> pd.DataFrame:
    return _q(con, "SELECT * FROM warehouse_daily WHERE date BETWEEN ? AND ?", (start, end), dates=("date",))


def supplier_skus(con, supplier_id: str) -> pd.DataFrame:
    """SKUs and warehouses that depend on a supplier (primary or backup)."""
    return _q(con, """SELECT p.sku_id, sw.warehouse_id, p.category, p.unit_cost, p.list_price,
                             CASE WHEN p.primary_supplier_id = ? THEN 'primary' ELSE 'backup' END AS role
                      FROM products p JOIN sku_warehouse sw USING (sku_id)
                      WHERE p.primary_supplier_id = ? OR p.backup_supplier_id = ?""",
              (supplier_id, supplier_id, supplier_id))
