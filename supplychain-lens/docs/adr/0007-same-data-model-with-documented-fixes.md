# ADR-0007 — Same NEXUS data model, with documented fixes

- **Status:** Accepted (extended 2026-10-27 for deep learning; details finalized in Phase 2)
- **Date:** 2026-10-03

## Context

The project must use the same data as NEXUS-SCCT. NEXUS uses only synthetic data produced by `seed_data.py` — no external datasets. That generator defines the entities (suppliers, warehouses, shipments, disruptions, KPI history, demand history), embedded reference tables (12 countries with risk baselines and ports, 10 product categories with values and lead times, 6 disruption types, 8 carriers) and the distributions.

The audit found defects that block real analytics (checks D1–D9):

| ID | Defect | Blocks |
|---|---|---|
| D1 | Price re-drawn daily per row | Revenue, value at risk |
| D2 | `delay_risk_score` is uniform noise | Any delay model |
| D3 | Delivered shipments are never late | OTIF, lead-time variance |
| D4 | Demand for 4 of 12 warehouses | Network inventory planning |
| D5 | No SKU → supplier link | Disruption → SKU → warehouse tracing |
| D6 | Disruptions hit random suppliers regardless of geography | Coherent risk analytics |
| D7 | KPI history has a built-in improvement trend | Honest impact claims |
| D8 | Dates anchored to the run date | Reproducibility |
| D9 | `risk_score` higher = safer | User comprehension |
| D10 | No inventory, product master or lane tables | Inventory and logistics agents |

## Decision

Reimplement the generator with two modes:

- **`nexus_v1`** — reproduces the original schema, reference tables, distributions and seed, so the original behaviour can be compared.
- **`nexus_v2`** (default) — the same entities and distributions plus only the fixes above. Examples: constant price per SKU; shipment delays drawn from the supplier's on-time rate; demand for all warehouses; a SKU master linking SKU → category → supplier(s); inventory positions; disruptions matched to supplier geography and category; a fixed snapshot date.

### Extension for deep learning (2026-10-27)

The original data can't support the deep-learning models ([change impact analysis §C](../change_impact_deep_learning.md)): one year of history, identical series shapes, no promotions, no inventory or lead-time history, and 50 disruptions over 60 days. `nexus_v2` therefore also:

1. **Covers 3 years:** 1 Jan 2023 → 31 Dec 2025, with a fixed snapshot date of 31 Dec 2025. Split: train → 30 Jun 2025, validation → 30 Sep 2025, **test = Q4 2025** (peak season).
2. **Runs one causal daily simulation instead of independent tables:** latent supplier health → disruptions → capacity and lead time → receipts → inventory → sales and stockouts → KPIs. A disruption therefore shows up consistently in its supplier's shipments, its SKUs' inventory and their stockouts.
3. **Adds the tables deep learning needs:** SKU master, SKU-warehouse assortment (4 warehouses per SKU, spread across all 12), promotion calendar, purchase orders, full shipment history, daily inventory, and correlated, geography-matched disruptions.
4. **Stores hidden truth separately** (`truth.*`: latent supplier health, injected anomalies). Feature code can't read it, and a test enforces that. Labels are built only from future events, never from inputs.
5. **Uses product-specific demand profiles** (trend, weekly and annual amplitude, promotion lift, price sensitivity) on top of the NEXUS formula, so series differ in shape and not only in scale.
6. **Does not add a store layer.** NEXUS warehouses (Hub / DC / Gateway) remain the demand nodes.

The data dictionary marks every column as **Original**, **Fixed**, **Extension**, **Derived** or **Truth**. KPI history is kept but labelled "synthetic trend from the NEXUS generator" and is never used as evidence of impact.

## Consequences

- We honour "same data" while making the data usable — and we can show both versions side by side.
- No unrelated datasets (M5, Kaggle, etc.) are introduced.
