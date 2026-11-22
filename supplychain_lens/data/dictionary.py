"""Data dictionary: what every column means and where it comes from.

Origin labels
  Original   same field and meaning as NEXUS-SCCT seed_data.py
  Fixed      NEXUS field, regenerated to remove a documented defect (D1–D10)
  Extension  new in SupplyChain Lens
  Derived    computed from other tables by the simulation (not drawn at random)
  Truth      hidden ground truth, used only for evaluation; never a model input

`render_markdown()` builds docs/data_dictionary.md from this spec plus the actual
parquet files, so dtypes and row counts in the doc are always real.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TABLES: dict[str, dict] = {
    "suppliers": {
        "grain": "one row per supplier (50)",
        "columns": {
            "supplier_id": ("Original", "Supplier key, SUP-0001…"),
            "name": ("Original", "Display name"),
            "country": ("Original", "One of the 12 NEXUS countries"),
            "region": ("Original", "APAC / EMEA / LATAM / Americas"),
            "currency": ("Original", "Local currency code"),
            "category": ("Original", "Product category supplied; every category has ≥ 2 suppliers (Fixed for D5)"),
            "health_score": ("Fixed", "NEXUS `risk_score` renamed: 5–99, higher = healthier (D9)"),
            "risk_level": ("Original", "LOW / MEDIUM / HIGH / CRITICAL from health_score, NEXUS cut-offs"),
            "avg_lead_time_days": ("Original", "Nominal lead time = category lead time + U(-5, 15)"),
            "tier": ("Original", "Supplier tier 1–3"),
            "certifications": ("Original", "Comma-separated certifications"),
            "primary_port": ("Original", "Export port"),
            "established_year": ("Original", "Year founded"),
            "employee_count": ("Original", "Headcount"),
            "lat": ("Extension", "Approximate location for maps (country centroid + jitter)"),
            "lng": ("Extension", "Approximate location for maps"),
            "on_time_delivery_rate": ("Derived", "Share of delivered shipments with delay ≤ 0 over the full history; empty for backup-only suppliers that received no orders. Descriptive only; never a model feature (it would include the test period)"),
            "annual_spend_usd": ("Derived", "Value of POs placed in 2025"),
            "capacity_units_per_day": ("Extension", "Nominal capacity = average daily demand it serves × U(1.3, 2.0)"),
        },
    },
    "warehouses": {
        "grain": "one row per warehouse (12, NEXUS locations)",
        "columns": {
            "warehouse_id": ("Original", "WH-001…WH-012"),
            "name": ("Original", "City + facility type"),
            "facility_type": ("Extension", "Hub / DC / Gateway, taken from the NEXUS names"),
            "city": ("Original", ""), "state": ("Original", ""), "country": ("Original", ""),
            "region": ("Original", ""), "lat": ("Original", ""), "lng": ("Original", ""),
            "capacity_units": ("Fixed", "Peak simulated stock × U(1.02, 1.25); NEXUS 30k–100k units was far below v2 volumes"),
        },
    },
    "products": {
        "grain": "one row per SKU (100)",
        "columns": {
            "sku_id": ("Original", "SKU-0000…SKU-0099"),
            "name": ("Extension", "Display name"),
            "category": ("Original", "Random category per SKU, as in NEXUS"),
            "unit_cost": ("Fixed", "Category average value × U(0.6, 1.4); constant (D1)"),
            "list_price": ("Fixed", "unit_cost × U(1.25, 1.8); constant (D1)"),
            "fragile": ("Original", "From category"),
            "primary_supplier_id": ("Extension", "Main source, same category (D5)"),
            "backup_supplier_id": ("Extension", "Alternative source, same category (D5)"),
        },
    },
    "sku_warehouse": {
        "grain": "one row per stocked SKU × warehouse (400)",
        "columns": {
            "sku_id": ("Extension", ""), "warehouse_id": ("Extension", "4 warehouses per SKU, across all 12 (D4)"),
            "demand_share": ("Extension", "Location demand multiplier U(0.6, 1.4)"),
            "review_weekday": ("Extension", "Weekday the historical policy reviews this series (0 = Monday)"),
        },
    },
    "promotions": {
        "grain": "one row per planned promotion",
        "columns": {
            "promo_id": ("Extension", ""), "sku_id": ("Extension", ""),
            "start_date": ("Extension", "Planned in advance, so usable as a known-future input"),
            "end_date": ("Extension", ""), "discount_pct": ("Extension", "10–30% price reduction"),
        },
    },
    "calendar": {
        "grain": "one row per day, 2023-01-01 … 2025-12-31",
        "columns": {
            "date": ("Extension", ""), "dow": ("Extension", "0 = Monday"), "month": ("Extension", ""),
            "day_of_year": ("Extension", ""),
            "is_black_friday": ("Fixed", "Friday after the 4th Thursday of November (NEXUS rule could mark two Fridays)"),
            "is_cyber_monday": ("Fixed", "Monday after Black Friday"),
            "is_holiday_week": ("Extension", "Thanksgiving Thursday → Cyber Monday"),
        },
    },
    "demand_daily": {
        "grain": "one row per SKU × warehouse × day up to the snapshot (438,400)",
        "columns": {
            "date": ("Original", ""), "sku_id": ("Original", ""), "warehouse_id": ("Original", ""),
            "demand_units": ("Fixed", "Customer demand (orders). NEXUS formula — trend, day-of-week, monthly sine, Black Friday ×3.4, Cyber Monday ×2.8, lognormal noise — scaled by product profiles, promotions, price and demand events. Forecasting target"),
            "sales_units": ("Derived", "Fulfilled = min(demand, on hand)"),
            "lost_sales_units": ("Derived", "Unfulfilled demand (stockout)"),
            "price": ("Fixed", "List price, minus discount on promo days (D1)"),
            "is_promo": ("Extension", "Promotion active"),
            "is_black_friday": ("Fixed", ""), "is_cyber_monday": ("Fixed", ""), "is_holiday_week": ("Extension", ""),
        },
    },
    "inventory_daily": {
        "grain": "one row per SKU × warehouse × day (end of day)",
        "columns": {
            "date": ("Extension", ""), "sku_id": ("Extension", ""), "warehouse_id": ("Extension", ""),
            "on_hand_units": ("Derived", "Stock after receipts, sales and adjustments"),
            "on_order_units": ("Derived", "Shipped quantity not yet received"),
            "receipts_units": ("Derived", "Units received that day"),
            "adjustment_units": ("Derived", "Recorded cycle-count adjustment (small noise)"),
        },
    },
    "warehouse_daily": {
        "grain": "one row per warehouse × day",
        "columns": {
            "warehouse_id": ("Extension", ""), "date": ("Extension", ""),
            "on_hand_units": ("Derived", "Sum over stocked SKUs"),
            "utilization_pct": ("Fixed", "on_hand / capacity × 100 (NEXUS drew a random static value)"),
        },
    },
    "purchase_orders": {
        "grain": "one row per replenishment order placed by the historical policy",
        "columns": {
            "po_id": ("Extension", ""), "sku_id": ("Extension", ""), "warehouse_id": ("Extension", ""),
            "supplier_id": ("Extension", "Primary supplier of the SKU"),
            "order_date": ("Extension", "Weekly review day"),
            "promised_date": ("Extension", "order_date + nominal lead time + supplier's quoted buffer"),
            "received_date": ("Extension", "Empty if not yet received at the snapshot"),
            "qty_ordered": ("Extension", "Order-up-to quantity of the NEXUS-style static rule"),
            "qty_shipped": ("Extension", "Reduced by supplier capacity during disruptions"),
            "unit_cost": ("Extension", ""), "status": ("Extension", "RECEIVED / OPEN"),
        },
    },
    "shipments": {
        "grain": "one row per PO (NEXUS shipment schema, full history)",
        "columns": {
            "shipment_id": ("Original", ""), "po_id": ("Extension", ""), "supplier_id": ("Original", ""),
            "warehouse_id": ("Original", "NEXUS: destination_warehouse_id"), "sku_id": ("Extension", ""),
            "origin_country": ("Original", ""), "origin_port": ("Original", ""), "carrier": ("Original", ""),
            "transport_mode": ("Original", "ROAD/RAIL within a region, else SEA (75%) / AIR"),
            "ship_date": ("Original", "NEXUS: origin_date"), "eta": ("Original", "Promised date"),
            "actual_arrival": ("Fixed", "Known only after arrival (D3: delivered shipments can be late)"),
            "status": ("Fixed", "DELIVERED / DELAYED / AT_PORT / IN_TRANSIT / CANCELLED at the snapshot"),
            "quantity_units": ("Original", ""), "value_usd": ("Original", "quantity × unit cost"),
            "delay_days": ("Fixed", "Arrival − promised for delivered; days overdue for open late shipments (D2, D3)"),
            "is_delayed": ("Original", "delay_days > 0"),
        },
    },
    "disruptions": {
        "grain": "one row per disruption event started on or before the snapshot",
        "columns": {
            "event_id": ("Original", ""), "event_type": ("Original", "NEXUS types"), "severity": ("Original", "NEXUS weights"),
            "title": ("Original", "NEXUS titles; weather titles hit only matching countries (D6)"),
            "scope": ("Extension", "Supplier, country, region or category affected"),
            "affected_category": ("Extension", "DEMAND events only"),
            "detected_at": ("Original", "Event start"), "resolved_at": ("Original", "Empty while active"),
            "status": ("Original", "ACTIVE / RESOLVED"),
            "capacity_factor": ("Extension", "Share of ordered quantity suppliers can ship during the event"),
            "lead_time_add_days": ("Extension", "Extra lead time during the event"),
            "demand_multiplier": ("Extension", "DEMAND events: demand scaling"),
            "estimated_impact_usd": ("Fixed", "Value of POs placed with affected suppliers during the event (NEXUS: random)"),
            "affected_entities": ("Original", "JSON list of supplier ids"),
        },
    },
    "disruption_suppliers": {
        "grain": "one row per event × affected supplier",
        "columns": {"event_id": ("Extension", ""), "supplier_id": ("Extension", "")},
    },
    "kpi_daily": {
        "grain": "one row per day",
        "columns": {
            "date": ("Original", ""),
            "fill_rate_pct": ("Fixed", "Network sales / demand (D7: computed, no built-in trend)"),
            "stockout_rate_pct": ("Fixed", "Share of series with lost sales that day"),
            "inventory_value_usd": ("Derived", "Σ on hand × unit cost"),
            "warehouse_utilization_pct": ("Fixed", "Average across warehouses"),
            "otif_pct": ("Fixed", "7-day rolling share of deliveries on time and ≥ 98% in full"),
            "active_disruptions": ("Fixed", "Events active that day"),
        },
    },
}

TRUTH: dict[str, str] = {
    "supplier_latent_daily": "Latent supplier health per day (drives delays and disruption hazard)",
    "injected_anomalies": "Unrecorded anomalies: demand_spike, inventory_shrink, supplier_delay_burst",
    "product_profiles": "Hidden demand-generating parameters per SKU",
    "all_disruption_events": "Every supplier-affecting event with start/end, used to build risk labels",
}


def render_markdown(data_dir: Path, truth_dir: Path) -> str:
    lines = [
        "# Data dictionary — SupplyChain Lens (nexus_v2)",
        "",
        "Generated by `python -m supplychain_lens.data.dictionary` from the actual Parquet files.",
        "**All data is synthetic**, produced by `supplychain_lens/data/generator.py` from the NEXUS-SCCT data model.",
        "",
        "**Origin:** *Original* = same as NEXUS-SCCT · *Fixed* = NEXUS field regenerated to remove a defect (D1–D10, "
        "see `docs/feature_audit.md`) · *Extension* = new · *Derived* = computed by the simulation · "
        "*Truth* = hidden, evaluation only.",
        "",
        "Calendar 2023-01-01 → 2025-12-31 · snapshot 2025-12-31 · split: train ≤ 2025-06-30, "
        "validation ≤ 2025-09-30, test = Q4 2025.",
        "",
    ]
    for table, spec in TABLES.items():
        df = pd.read_parquet(data_dir / f"{table}.parquet")
        lines += [f"## `{table}`", "", f"{spec['grain']} — **{len(df):,} rows**", "",
                  "| Column | Type | Origin | Description |", "|---|---|---|---|"]
        for col in df.columns:
            origin, desc = spec["columns"].get(col, ("?", "UNDOCUMENTED"))
            lines.append(f"| `{col}` | {df[col].dtype} | {origin} | {desc} |")
        lines.append("")
    lines += ["## Hidden truth (`data/truth/`)", "",
              "Never read by feature or model code (enforced by `tests/test_data.py`). Used only to build labels "
              "and evaluate models.", "", "| Table | Rows | Content |", "|---|---|---|"]
    for name, desc in TRUTH.items():
        n = len(pd.read_parquet(truth_dir / f"{name}.parquet"))
        lines.append(f"| `{name}` | {n:,} | {desc} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    from supplychain_lens.config import PROJECT_ROOT, get_settings

    s = get_settings()
    out = PROJECT_ROOT / "docs" / "data_dictionary.md"
    out.write_text(render_markdown(s.data_dir / "nexus_v2", s.data_dir / "truth"), encoding="utf-8")
    print(f"Written {out}")
