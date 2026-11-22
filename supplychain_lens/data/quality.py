"""Data-quality report for the generated data.

Checks keys, foreign keys, missing values and value ranges; confirms that the NEXUS
demand formula is still visible in the data; and re-runs the Phase 0 audit checks
D1–D9 to show each defect is fixed in nexus_v2.

    python -m supplychain_lens.data.quality      # writes reports/data_quality.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from supplychain_lens.data import reference as ref


@dataclass
class Check:
    area: str
    name: str
    passed: bool
    detail: str


def load(data_dir: Path) -> dict[str, pd.DataFrame]:
    return {p.stem: pd.read_parquet(p) for p in sorted(data_dir.glob("*.parquet"))}


def run_checks(t: dict[str, pd.DataFrame]) -> list[Check]:
    c: list[Check] = []
    add = lambda area, name, ok, detail: c.append(Check(area, name, bool(ok), detail))  # noqa: E731

    # Keys and foreign keys
    for table, key in [("suppliers", ["supplier_id"]), ("warehouses", ["warehouse_id"]), ("products", ["sku_id"]),
                       ("sku_warehouse", ["sku_id", "warehouse_id"]), ("demand_daily", ["sku_id", "warehouse_id", "date"]),
                       ("inventory_daily", ["sku_id", "warehouse_id", "date"]), ("purchase_orders", ["po_id"]),
                       ("shipments", ["shipment_id"]), ("disruptions", ["event_id"])]:
        dup = t[table].duplicated(key).sum()
        add("Keys", f"{table} unique on {', '.join(key)}", dup == 0, f"{dup} duplicates")
    fks = [("products", "primary_supplier_id", "suppliers", "supplier_id"),
           ("products", "backup_supplier_id", "suppliers", "supplier_id"),
           ("sku_warehouse", "warehouse_id", "warehouses", "warehouse_id"),
           ("purchase_orders", "supplier_id", "suppliers", "supplier_id"),
           ("shipments", "po_id", "purchase_orders", "po_id"),
           ("disruption_suppliers", "supplier_id", "suppliers", "supplier_id"),
           ("disruption_suppliers", "event_id", "disruptions", "event_id")]
    for child, col, parent, pcol in fks:
        missing = (~t[child][col].isin(t[parent][pcol])).sum()
        add("Foreign keys", f"{child}.{col} → {parent}.{pcol}", missing == 0, f"{missing} orphans")
    same_cat = (t["products"].merge(t["suppliers"][["supplier_id", "category"]], left_on="primary_supplier_id",
                                    right_on="supplier_id", suffixes=("", "_s")).eval("category == category_s")).all()
    add("Foreign keys", "primary supplier sells the SKU's category", same_cat, "")

    # Missing values (only allowed where documented)
    allowed = {("purchase_orders", "received_date"), ("shipments", "actual_arrival"), ("disruptions", "resolved_at"),
               ("disruptions", "affected_category"), ("kpi_daily", "otif_pct"),
               ("suppliers", "on_time_delivery_rate")}  # backup-only suppliers receive no orders
    for name, df in t.items():
        for col in df.columns:
            n = df[col].isna().sum()
            if n and (name, col) not in allowed:
                add("Missing values", f"{name}.{col}", False, f"{n} missing")
    add("Missing values", "no unexpected missing values", all(x.passed for x in c if x.area == "Missing values"), "")

    # Ranges and accounting identities
    d, inv = t["demand_daily"], t["inventory_daily"]
    add("Ranges", "demand, sales, lost sales ≥ 0", (d[["demand_units", "sales_units", "lost_sales_units"]] >= 0).all().all(), "")
    add("Ranges", "sales + lost sales = demand", (d.sales_units + d.lost_sales_units == d.demand_units).all(), "")
    add("Ranges", "on hand ≥ 0", (inv.on_hand_units >= 0).all(), "")
    u = t["warehouse_daily"].utilization_pct
    add("Ranges", "utilization within 0–100%", u.between(0, 100).all(), f"min {u.min():.1f}, max {u.max():.1f}")
    po = t["purchase_orders"]
    add("Ranges", "qty shipped ≤ qty ordered", (po.qty_shipped <= po.qty_ordered).all(), "")
    rec = po.dropna(subset=["received_date"])
    add("Ranges", "received after ordered", (rec.received_date > rec.order_date).all(), "")

    # NEXUS demand formula still visible
    g = d.assign(dow=d.date.dt.dayofweek).groupby("dow").demand_units.mean()
    corr = np.corrcoef(g.to_numpy(), ref.DOW_FACTORS)[0, 1]
    add("NEXUS formula", "day-of-week pattern matches NEXUS factors", corr > 0.95, f"correlation {corr:.3f}")
    bf = d[d.is_black_friday].demand_units.mean() / d[d.date.dt.month == 11].demand_units.mean()
    add("NEXUS formula", "Black Friday uplift visible", bf > 1.8, f"BF / November mean = {bf:.2f}×")
    yearly = d.groupby(d.date.dt.year).demand_units.mean()
    add("NEXUS formula", "positive average trend", yearly.iloc[-1] > yearly.iloc[0], f"{yearly.round(0).to_dict()}")

    # Phase 0 defects D1–D9 re-checked on v2
    n_prices = d[~d.is_promo].groupby(["sku_id", "warehouse_id"]).price.nunique().max()
    add("Defects fixed", "D1 price constant outside promotions", n_prices == 1, f"max distinct non-promo prices = {n_prices}")
    sh = t["shipments"]
    sup = t["suppliers"]
    corr_h = sup.health_score.corr(sup.on_time_delivery_rate)
    add("Defects fixed", "D2 delays relate to supplier health", corr_h > 0.3, f"corr(health_score, on-time rate) = {corr_h:.2f}")
    late_delivered = (sh[sh.status == "DELIVERED"].delay_days > 0).mean()
    add("Defects fixed", "D3 delivered shipments can be late", late_delivered > 0.05, f"{late_delivered:.1%} of delivered were late")
    n_wh = d.warehouse_id.nunique()
    add("Defects fixed", "D4 demand in all 12 warehouses", n_wh == 12, f"{n_wh} warehouses")
    add("Defects fixed", "D5 SKU → supplier link exists", t["products"].primary_supplier_id.notna().all(), "")
    ds = t["disruption_suppliers"].merge(t["disruptions"][["event_id", "title"]], on="event_id").merge(
        sup[["supplier_id", "country"]], on="supplier_id")
    bad = sum((~g_.country.isin(ref.WEATHER_FOOTPRINT[title])).sum()
              for title, g_ in ds[ds.title.isin(list(ref.WEATHER_FOOTPRINT))].groupby("title"))
    add("Defects fixed", "D6 weather hits only matching countries", bad == 0, f"{bad} mismatches")
    k = t["kpi_daily"]
    slope = np.polyfit(np.arange(len(k)), k.fill_rate_pct.to_numpy(), 1)[0] * 365
    add("Defects fixed", "D7 no built-in KPI trend", abs(slope) < 3, f"fill-rate trend {slope:+.2f} pts/yr")
    add("Defects fixed", "D8 fixed snapshot date", d.date.max() == pd.Timestamp("2025-12-31"), f"last date {d.date.max().date()}")
    hl = sup.groupby("risk_level").health_score.mean()
    add("Defects fixed", "D9 higher health_score = lower risk", hl["LOW"] > hl["CRITICAL"] if "CRITICAL" in hl else True,
        f"mean health by level {hl.round(0).to_dict()}")
    return c


def summary_stats(t: dict[str, pd.DataFrame]) -> list[str]:
    d, sh, dis, sup = t["demand_daily"], t["shipments"], t["disruptions"], t["suppliers"]
    dl = sh[sh.status == "DELIVERED"]
    return [
        f"- Fill rate (sales / demand): **{d.sales_units.sum() / d.demand_units.sum():.1%}**; "
        f"series-days with a stockout: **{(d.lost_sales_units > 0).mean():.1%}**",
        f"- Delivered shipments late: **{(dl.delay_days > 0).mean():.1%}**; mean delay {dl.delay_days.mean():+.1f} days; "
        f"P90 {dl.delay_days.quantile(0.9):.0f} days",
        f"- Supplier on-time rate range: {sup.on_time_delivery_rate.min():.0%} – {sup.on_time_delivery_rate.max():.0%}",
        f"- Disruptions: {len(dis)} events ({', '.join(f'{k} {v}' for k, v in dis.event_type.value_counts().items())}); "
        f"{(dis.status == 'ACTIVE').sum()} active at the snapshot",
        f"- Promotions: {len(t['promotions']):,}; promo days {d.is_promo.mean():.1%} of series-days",
    ]


def render(checks: list[Check], t: dict[str, pd.DataFrame]) -> str:
    passed = sum(x.passed for x in checks)
    lines = ["# Data-quality report — nexus_v2", "",
             "Generated by `python -m supplychain_lens.data.quality`. **All data is synthetic.**", "",
             f"**{passed} of {len(checks)} checks passed.**", "", "## Summary", "", *summary_stats(t), "",
             "## Row counts", "", "| Table | Rows |", "|---|---|"]
    lines += [f"| `{n}` | {len(df):,} |" for n, df in t.items()]
    for area in dict.fromkeys(x.area for x in checks):
        lines += ["", f"## {area}", "", "| Check | Result | Detail |", "|---|---|---|"]
        lines += [f"| {x.name} | {'✅' if x.passed else '❌'} | {x.detail} |" for x in checks if x.area == area]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    from supplychain_lens.config import get_settings

    s = get_settings()
    tables = load(s.data_dir / "nexus_v2")
    checks = run_checks(tables)
    s.reports_dir.mkdir(parents=True, exist_ok=True)
    (s.reports_dir / "data_quality.md").write_text(render(checks, tables), encoding="utf-8")
    print(f"{sum(x.passed for x in checks)}/{len(checks)} checks passed -> {s.reports_dir / 'data_quality.md'}")
    for x in checks:
        if not x.passed:
            print("  FAILED:", x.area, x.name, x.detail)
