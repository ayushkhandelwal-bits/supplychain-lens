# Design adapted from NEXUS-SCCT by Syed Zain Ahmed
# (https://github.com/ahmedzainsyed/nexus-scct), seed_data.py.
# Reimplemented and extended in SupplyChain Lens.
"""Synthetic supply-chain data generator (NEXUS data model).

Two modes
---------
``nexus_v1``  The original NEXUS schema and distributions: 50 suppliers, 12 warehouses,
              300 shipments, 50 disruptions, 90 days of KPIs and 1 year of demand for
              100 SKUs in 4 warehouses — including the original defects (D1–D9), so the
              two versions can be compared. Dates are anchored to the fixed snapshot
              date instead of "now" (fix D8 applied to both modes).

``nexus_v2``  (default) The same entities, reference tables and demand formula, run as ONE
              causal daily simulation over 3 years:

                  latent supplier health ─┐
                  regional congestion ────┼─> disruptions ─> lead time & capacity
                                          │                       │
                  demand (NEXUS formula   │                       v
                  + product profiles      └──────────────> purchase orders / shipments
                  + promotions + events)                          │
                          │                                       v
                          └───────────────> inventory ─> sales, lost sales, KPIs

              so a disruption shows up consistently in its supplier's shipments, its SKUs'
              inventory and their stockouts. Hidden "truth" (latent health, injected
              anomalies, product profiles) is written to a separate folder that feature
              code must never read (enforced by tests).

Usage
-----
    python -m supplychain_lens.data.generator               # nexus_v2 -> data/nexus_v2, data/truth
    python -m supplychain_lens.data.generator --mode nexus_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from supplychain_lens import __version__
from supplychain_lens.config import get_settings
from supplychain_lens.data import reference as ref

SEVERITIES = list(ref.SEVERITY_WEIGHTS)
SEV_P = np.array(list(ref.SEVERITY_WEIGHTS.values()))
# v2 effect sizes per severity (Extension: NEXUS had durations only).
SEV_DURATION = {"LOW": (3, 7), "MEDIUM": (7, 14), "HIGH": (14, 30), "CRITICAL": (30, 60)}
SEV_CAPACITY = {"LOW": 0.85, "MEDIUM": 0.65, "HIGH": 0.40, "CRITICAL": 0.10}
SEV_LT_ADD = {"LOW": (2, 5), "MEDIUM": (5, 10), "HIGH": (10, 20), "CRITICAL": (20, 40)}


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int = 42
    start: date = date(2023, 1, 1)
    end: date = date(2025, 12, 31)
    snapshot: date = date(2025, 12, 31)
    n_suppliers: int = 50
    n_skus: int = 100
    warehouses_per_sku: int = 4
    review_period_days: int = 7
    promo_events_per_sku_year: float = 5.0
    supplier_deterioration_per_year: float = 0.55
    supplier_disruption_base_rate: float = 0.00012  # per supplier-day at full health
    supplier_disruption_health_slope: float = 8.0
    port_events_per_year: float = 6.0
    weather_events_per_year: float = 3.0
    geopolitical_events_per_year: float = 2.0
    logistics_events_per_year: float = 2.0
    demand_events_per_year: float = 4.0
    anomalies: dict = field(
        default_factory=lambda: {"demand_spike": 80, "inventory_shrink": 60, "supplier_delay_burst": 40}
    )

    @classmethod
    def from_settings(cls) -> GeneratorConfig:
        s = get_settings()
        return cls(seed=s.seed, start=s.calendar_start, end=s.calendar_end, snapshot=s.snapshot_date)

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), default=str, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
# Calendar helpers
# ─────────────────────────────────────────────────────────────────────────────
def black_friday(year: int) -> date:
    """Friday after the 4th Thursday of November.

    NEXUS used `month == 11 and day >= 22 and Friday`, which can mark two Fridays in some
    years; this is the corrected rule (documented as a small fix).
    """
    nov1 = date(year, 11, 1)
    first_thu = nov1 + timedelta(days=(3 - nov1.weekday()) % 7)
    return first_thu + timedelta(days=22)


def calendar_frame(start: date, end: date) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    bf = {pd.Timestamp(black_friday(y)) for y in range(start.year, end.year + 1)}
    cm = {d + pd.Timedelta(days=3) for d in bf}
    cal = pd.DataFrame({"date": dates})
    cal["dow"] = cal.date.dt.dayofweek
    cal["month"] = cal.date.dt.month
    cal["day_of_year"] = cal.date.dt.dayofyear
    cal["is_black_friday"] = cal.date.isin(bf)
    cal["is_cyber_monday"] = cal.date.isin(cm)
    cal["is_holiday_week"] = False
    for d in bf:  # Thanksgiving week through Cyber Monday
        cal.loc[(cal.date >= d - pd.Timedelta(days=1)) & (cal.date <= d + pd.Timedelta(days=3)), "is_holiday_week"] = True
    return cal


# ─────────────────────────────────────────────────────────────────────────────
# Master data
# ─────────────────────────────────────────────────────────────────────────────
def _risk_level(health: float) -> str:
    # Same cut-offs as NEXUS risk_score (renamed health_score: higher = healthier, fix D9).
    return "CRITICAL" if health < 35 else "HIGH" if health < 55 else "MEDIUM" if health < 75 else "LOW"


def make_suppliers(cfg: GeneratorConfig, rng: np.random.Generator) -> pd.DataFrame:
    countries = list(ref.COUNTRIES)
    cats = [c["name"] for c in ref.PRODUCT_CATEGORIES]
    # Every category gets at least two suppliers so each SKU has a backup source (needed for D5 fix).
    cat_assign = cats * 2 + list(rng.choice(cats, cfg.n_suppliers - 2 * len(cats)))
    rng.shuffle(cat_assign)
    rows = []
    for i in range(cfg.n_suppliers):
        country = countries[rng.integers(len(countries))]
        cdata = ref.COUNTRIES[country]
        category = cat_assign[i]
        risk_base = cdata["risk_base"] + rng.uniform(-0.1, 0.2)  # NEXUS formula
        health = int(np.clip((1 - risk_base) * 100 + rng.normal(0, 8), 5, 99))  # NEXUS risk_score
        lat, lng = ref.COUNTRY_COORDS[country]
        rows.append({
            "supplier_id": f"SUP-{i + 1:04d}",
            "name": f"{country} {category} {'Co.' if i % 3 == 0 else 'Ltd' if i % 3 == 1 else 'GmbH'}",
            "country": country,
            "region": cdata["region"],
            "currency": cdata["currency"],
            "category": category,
            "health_score": health,
            "risk_level": _risk_level(health),
            "avg_lead_time_days": ref.CATEGORY_BY_NAME[category]["lead_time_days"] + int(rng.integers(-5, 16)),
            "tier": int(rng.choice([1, 1, 1, 2, 2, 3])),
            "certifications": ",".join(sorted(rng.choice(ref.CERTIFICATIONS, int(rng.integers(1, 5)), replace=False))),
            "primary_port": str(rng.choice(cdata["ports"])),
            "established_year": int(rng.integers(1985, 2016)),
            "employee_count": int(rng.integers(50, 5001)),
            "lat": round(lat + rng.normal(0, 1.5), 3),
            "lng": round(lng + rng.normal(0, 1.5), 3),
            "_risk_base": float(np.clip(risk_base, 0.02, 0.9)),
        })
    return pd.DataFrame(rows)


def make_warehouses() -> pd.DataFrame:
    rows = []
    for i, loc in enumerate(ref.WAREHOUSE_LOCATIONS):
        ftype = "Hub" if i < 4 else "DC" if i < 8 else "Gateway"
        rows.append({"warehouse_id": f"WH-{i + 1:03d}", "name": f"{loc['city']} {ftype}", "facility_type": ftype,
                     **{k: loc[k] for k in ("city", "state", "country", "region", "lat", "lng")}})
    return pd.DataFrame(rows)


def make_products(cfg: GeneratorConfig, suppliers: pd.DataFrame, rng: np.random.Generator):
    cats = [c["name"] for c in ref.PRODUCT_CATEGORIES]
    products, profiles = [], []
    for k in range(cfg.n_skus):
        category = cats[rng.integers(len(cats))]  # NEXUS: random category per SKU
        cinfo = ref.CATEGORY_BY_NAME[category]
        pool = suppliers.loc[suppliers.category == category, "supplier_id"].to_numpy()
        primary, backup = rng.choice(pool, 2, replace=False)
        unit_cost = round(cinfo["avg_value"] * rng.uniform(0.6, 1.4), 2)
        sku = f"SKU-{k:04d}"
        products.append({
            "sku_id": sku, "name": f"{category} item {k:04d}", "category": category,
            "unit_cost": unit_cost, "list_price": round(unit_cost * rng.uniform(1.25, 1.8), 2),
            "fragile": cinfo["fragile"], "primary_supplier_id": primary, "backup_supplier_id": backup,
        })
        # Extension: volume scaled by inverse unit value so revenue mixes are plausible
        # (NEXUS drew U(100, 2000) units/day for every category, including $15k machinery).
        volume_factor = float(np.clip(300 / cinfo["avg_value"], 0.05, 1.5))
        profiles.append({
            "sku_id": sku,
            "base_demand": rng.uniform(100, 2000) * volume_factor,   # NEXUS range x volume factor
            "trend_annual": rng.normal(ref.ANNUAL_TREND, 0.05),       # NEXUS +1.5%/yr on average
            "weekly_strength": rng.uniform(0.4, 1.6),                 # scales NEXUS day-of-week factors
            "annual_amplitude": rng.uniform(0.0, 0.35),               # NEXUS 0.20 on average
            "annual_phase_months": float(rng.choice([0, 0, 0, 3, 6])),
            "holiday_sensitivity": rng.uniform(0.3, 1.2),             # scales NEXUS 3.4x / 2.8x uplift
            "promo_lift": rng.uniform(0.2, 0.8),
            "price_elasticity": rng.uniform(0.8, 2.5),
            "noise_sigma": rng.uniform(0.08, 0.20),                   # NEXUS 0.12 on average
        })
    return pd.DataFrame(products), pd.DataFrame(profiles)


def make_assortment(cfg: GeneratorConfig, products: pd.DataFrame, warehouses: pd.DataFrame, rng):
    """Each SKU is stocked in 4 warehouses, spread across all 12 (fix D4)."""
    wh_ids = warehouses.warehouse_id.to_numpy()
    rows = []
    for sku in products.sku_id:
        for wh in sorted(rng.choice(wh_ids, cfg.warehouses_per_sku, replace=False)):
            rows.append({"sku_id": sku, "warehouse_id": wh, "demand_share": rng.uniform(0.6, 1.4),
                         "review_weekday": int(rng.integers(0, 7))})
    return pd.DataFrame(rows)


def make_promotions(cfg: GeneratorConfig, products: pd.DataFrame, cal: pd.DataFrame, rng) -> pd.DataFrame:
    """Planned promotions per SKU (known in advance -> a legitimate future input)."""
    n_days, years = len(cal), len(cal) / 365.25
    rows, pid = [], 0
    for sku in products.sku_id:
        for _ in range(rng.poisson(cfg.promo_events_per_sku_year * years)):
            start = int(rng.integers(0, n_days - 7))
            length = int(rng.integers(3, 8))
            pid += 1
            rows.append({"promo_id": f"PRM-{pid:05d}", "sku_id": sku,
                         "start_date": cal.date.iloc[start],
                         "end_date": cal.date.iloc[min(n_days - 1, start + length - 1)],
                         "discount_pct": round(float(rng.uniform(0.10, 0.30)), 3)})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Latent supplier health, congestion and disruptions
# ─────────────────────────────────────────────────────────────────────────────
def simulate_supplier_health(cfg, suppliers: pd.DataFrame, n_days: int, rng) -> np.ndarray:
    """AR(1) health in [0.05, 0.99] per supplier-day with occasional deterioration episodes."""
    base = 1 - suppliers._risk_base.to_numpy()
    S = len(base)
    target = np.repeat(base[:, None], n_days, axis=1)
    for s in range(S):
        n_ep = rng.poisson(cfg.supplier_deterioration_per_year * n_days / 365 * (1.4 - base[s]))
        for _ in range(n_ep):
            t0, length = int(rng.integers(0, n_days)), int(rng.integers(45, 100))
            target[s, t0 : t0 + length] -= rng.uniform(0.25, 0.45)
    h = np.empty((S, n_days))
    h[:, 0] = base
    shocks = rng.normal(0, 0.012, (S, n_days))
    for t in range(1, n_days):
        h[:, t] = h[:, t - 1] + 0.06 * (target[:, t] - h[:, t - 1]) + shocks[:, t]
    return np.clip(h, 0.05, 0.99)


def generate_disruptions(cfg, cal, suppliers, products, health, rng):
    """Create disruption events. Returns (events DataFrame, bridge DataFrame, congestion array)."""
    n_days = len(cal)
    years = n_days / 365.25
    countries = list(ref.COUNTRIES)
    cidx = {c: i for i, c in enumerate(countries)}
    sup_country = suppliers.country.map(cidx).to_numpy()
    congestion = np.zeros((len(countries), n_days))
    events = []

    def new_event(etype, title, start, severity, supplier_idx, category=None, scope=""):
        lo, hi = SEV_DURATION[severity]
        dur = int(rng.integers(lo, hi + 1))
        lt_lo, lt_hi = SEV_LT_ADD[severity]
        events.append({
            "event_type": etype, "title": title, "severity": severity, "scope": scope,
            "start_idx": start, "end_idx": min(n_days - 1, start + dur - 1),
            "capacity_factor": SEV_CAPACITY[severity] if etype != "DEMAND" else 1.0,
            "lead_time_add_days": int(rng.integers(lt_lo, lt_hi + 1)) if etype != "DEMAND" else 0,
            "demand_multiplier": 1.0 if etype != "DEMAND" else
            (float(rng.uniform(1.5, 2.2)) if "drop" not in title else float(rng.uniform(0.5, 0.7))),
            "affected_category": category, "supplier_idx_list": list(map(int, supplier_idx)),
        })

    def severity():
        return SEVERITIES[rng.choice(4, p=SEV_P)]

    # 1) Supplier-specific events: hazard rises sharply as latent health falls.
    hazard = cfg.supplier_disruption_base_rate * np.exp(cfg.supplier_disruption_health_slope * (0.9 - health).clip(0))
    u = rng.random(health.shape)
    cooldown = np.zeros(len(suppliers), dtype=int)
    for t in range(n_days):
        hits = np.where((u[:, t] < hazard[:, t]) & (cooldown <= t))[0]
        for s in hits:
            sev = severity()
            new_event("SUPPLIER", str(rng.choice(ref.DISRUPTION_TYPES["SUPPLIER"])), t, sev, [s],
                      scope=suppliers.supplier_id.iloc[s])
            cooldown[s] = t + 60

    # 2) Port events, preceded by observable congestion build-up in that country (2–5 weeks).
    for _ in range(rng.poisson(cfg.port_events_per_year * years)):
        c = int(rng.integers(len(countries)))
        t = int(rng.integers(40, n_days))
        build = int(rng.integers(14, 36))
        congestion[c, t - build : t] += np.linspace(0, 1, build)
        congestion[c, t : t + 10] += np.linspace(1, 0, 10)[: max(0, min(10, n_days - t))]
        members = np.where(sup_country == c)[0]
        if len(members):
            new_event("PORT", str(rng.choice(ref.DISRUPTION_TYPES["PORT"])), t, severity(), members,
                      scope=countries[c])

    # 3) Weather: footprint-matched countries only (fix D6). Not predictable from supplier data.
    weather_titles = list(ref.WEATHER_FOOTPRINT)
    for _ in range(rng.poisson(cfg.weather_events_per_year * years)):
        title = weather_titles[rng.integers(len(weather_titles))]
        cs = [cidx[c] for c in ref.WEATHER_FOOTPRINT[title]]
        members = np.where(np.isin(sup_country, cs))[0]
        if len(members):
            new_event("WEATHER", title, int(rng.integers(0, n_days)), severity(), members,
                      scope=",".join(ref.WEATHER_FOOTPRINT[title]))

    # 4) Geopolitical: one country; 5) Logistics: one region.
    for _ in range(rng.poisson(cfg.geopolitical_events_per_year * years)):
        c = int(rng.integers(len(countries)))
        members = np.where(sup_country == c)[0]
        if len(members):
            new_event("GEOPOLITICAL", str(rng.choice(ref.DISRUPTION_TYPES["GEOPOLITICAL"])),
                      int(rng.integers(0, n_days)), severity(), members, scope=countries[c])
    regions = sorted(suppliers.region.unique())
    for _ in range(rng.poisson(cfg.logistics_events_per_year * years)):
        reg = regions[rng.integers(len(regions))]
        members = np.where(suppliers.region.to_numpy() == reg)[0]
        new_event("LOGISTICS", str(rng.choice(ref.DISRUPTION_TYPES["LOGISTICS"])),
                  int(rng.integers(0, n_days)), severity(), members, scope=reg)

    # 6) Demand events: category-level surges or drops (recorded, so not "anomalies").
    cats = sorted(products.category.unique())
    for _ in range(rng.poisson(cfg.demand_events_per_year * years)):
        cat = cats[rng.integers(len(cats))]
        title = str(rng.choice(["Unexpected demand surge", "Product launch demand spike",
                                "Demand drop due to recession signal"]))
        new_event("DEMAND", title, int(rng.integers(0, n_days)), str(rng.choice(["MEDIUM", "HIGH"])), [],
                  category=cat, scope=cat)

    ev = pd.DataFrame(events).sort_values(["start_idx", "event_type"]).reset_index(drop=True)
    ev.insert(0, "event_id", [f"DISR-{i + 1:04d}" for i in range(len(ev))])
    bridge = pd.DataFrame([(r.event_id, suppliers.supplier_id.iloc[s]) for r in ev.itertuples() for s in r.supplier_idx_list],
                          columns=["event_id", "supplier_id"])
    return ev, bridge, np.clip(congestion, 0, 1.5)


# ─────────────────────────────────────────────────────────────────────────────
# Demand
# ─────────────────────────────────────────────────────────────────────────────
def build_demand(cal, products, profiles, assortment, promotions, events, rng):
    """True demand per (series, day) using the NEXUS formula plus product-specific profiles."""
    n_days = len(cal)
    t = np.arange(n_days)
    prof = profiles.set_index("sku_id")
    prod = products.set_index("sku_id")
    series = assortment.reset_index(drop=True)
    S = len(series)

    dow_f = np.array(ref.DOW_FACTORS)[cal.dow.to_numpy()]
    month = cal.month.to_numpy()
    bf = cal.is_black_friday.to_numpy()
    cm = cal.is_cyber_monday.to_numpy()

    # promo discount per (sku, day)
    sku_index = {s: i for i, s in enumerate(products.sku_id)}
    discount = np.zeros((len(products), n_days))
    d0 = cal.date.iloc[0]
    for p in promotions.itertuples():
        a, b = (pd.Timestamp(p.start_date) - d0).days, (pd.Timestamp(p.end_date) - d0).days
        discount[sku_index[p.sku_id], a : b + 1] = p.discount_pct

    # category demand events
    cat_mult = {c: np.ones(n_days) for c in products.category.unique()}
    for e in events[events.event_type == "DEMAND"].itertuples():
        cat_mult[e.affected_category][e.start_idx : e.end_idx + 1] *= e.demand_multiplier

    demand = np.zeros((S, n_days))
    price = np.zeros((S, n_days))
    promo = np.zeros((S, n_days), dtype=bool)
    for i, r in enumerate(series.itertuples()):
        p, q = prof.loc[r.sku_id], prod.loc[r.sku_id]
        k = sku_index[r.sku_id]
        base = p.base_demand * r.demand_share
        trend = 1 + p.trend_annual * t / 365
        weekly = 1 + p.weekly_strength * (dow_f - 1)
        seasonal = 1 + p.annual_amplitude * np.sin(2 * np.pi * (month + p.annual_phase_months) / 12)
        holiday = np.where(bf, 1 + p.holiday_sensitivity * (ref.BLACK_FRIDAY_MULTIPLIER - 1), 1.0)
        holiday = holiday * np.where(cm, 1 + p.holiday_sensitivity * (ref.CYBER_MONDAY_MULTIPLIER - 1), 1.0)
        disc = discount[k]
        price_ratio = 1 - disc
        promo_effect = np.where(disc > 0, (1 + p.promo_lift) * price_ratio ** (-p.price_elasticity), 1.0)
        noise = rng.lognormal(0, p.noise_sigma, n_days)
        demand[i] = base * trend * weekly * seasonal * holiday * promo_effect * cat_mult[q.category] * noise
        price[i] = q.list_price * price_ratio
        promo[i] = disc > 0
    return series, demand.clip(0), price, promo


# ─────────────────────────────────────────────────────────────────────────────
# Anomalies (hidden truth)
# ─────────────────────────────────────────────────────────────────────────────
def plan_anomalies(cfg, n_series, n_suppliers, n_days, promo, rng) -> pd.DataFrame:
    rows = []
    for kind, n in cfg.anomalies.items():
        for _ in range(n):
            if kind == "supplier_delay_burst":
                rows.append({"type": kind, "entity_type": "supplier", "entity_idx": int(rng.integers(n_suppliers)),
                             "start_idx": int(rng.integers(30, n_days - 30)), "length": int(rng.integers(10, 25)),
                             "magnitude": float(rng.uniform(8, 20))})  # extra delay days
                continue
            for _attempt in range(20):  # avoid promo days so spikes are genuinely unexplained
                s, t0 = int(rng.integers(n_series)), int(rng.integers(30, n_days - 10))
                length = int(rng.integers(1, 5)) if kind == "demand_spike" else 1
                if not promo[s, t0 : t0 + length].any():
                    break
            mag = float(rng.uniform(2.5, 4.0)) if kind == "demand_spike" else float(rng.uniform(0.25, 0.5))
            rows.append({"type": kind, "entity_type": "series", "entity_idx": s, "start_idx": t0,
                         "length": length, "magnitude": mag})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Operations simulation (historical replenishment policy = NEXUS-style static rule)
# ─────────────────────────────────────────────────────────────────────────────
def simulate_operations(cfg, cal, series, demand, products, suppliers, warehouses, health, congestion,
                        events, bridge, anomalies, rng):
    n_days = len(cal)
    S = len(series)
    prod = products.set_index("sku_id")
    sup = suppliers.reset_index(drop=True)
    sup_idx = {s: i for i, s in enumerate(sup.supplier_id)}
    country_idx = {c: i for i, c in enumerate(ref.COUNTRIES)}
    sup_country = sup.country.map(country_idx).to_numpy()
    primary = np.array([sup_idx[prod.loc[s, "primary_supplier_id"]] for s in series.sku_id])
    unit_cost = np.array([prod.loc[s, "unit_cost"] for s in series.sku_id])
    lt_nominal = sup.avg_lead_time_days.to_numpy()[primary]
    lt_sigma = (1.0 + 4.0 * sup._risk_base.to_numpy())  # days, per supplier
    # Suppliers quote a promised date with a buffer of about one standard deviation of their noise.
    quote_buffer = np.ceil(1.2 * lt_sigma).astype(int)
    review_day = series.review_weekday.to_numpy()
    dows = cal.dow.to_numpy()

    # Active disruption effects per supplier-day.
    cap = np.ones((len(sup), n_days))
    lt_add = np.zeros((len(sup), n_days))
    for e in events[events.event_type != "DEMAND"].itertuples():
        for s in bridge.loc[bridge.event_id == e.event_id, "supplier_id"].map(sup_idx):
            cap[s, e.start_idx : e.end_idx + 1] = np.minimum(cap[s, e.start_idx : e.end_idx + 1], e.capacity_factor)
            lt_add[s, e.start_idx : e.end_idx + 1] = np.maximum(lt_add[s, e.start_idx : e.end_idx + 1],
                                                                 e.lead_time_add_days)
    burst = np.zeros((len(sup), n_days))
    for a in anomalies[anomalies.type == "supplier_delay_burst"].itertuples():
        burst[a.entity_idx, a.start_idx : a.start_idx + a.length] = a.magnitude

    # Demand anomalies modify the realised demand (not recorded anywhere observable).
    demand = demand.copy()
    for a in anomalies[anomalies.type == "demand_spike"].itertuples():
        demand[a.entity_idx, a.start_idx : a.start_idx + a.length] *= a.magnitude
    demand = np.round(demand).astype(np.int64)
    shrink_by_day: dict[int, list[tuple[int, float]]] = {}
    for a in anomalies[anomalies.type == "inventory_shrink"].itertuples():
        shrink_by_day.setdefault(a.start_idx, []).append((a.entity_idx, a.magnitude))

    # Initial state: steady state under the historical rule.
    mean0 = demand[:, :28].mean(axis=1)
    R = cfg.review_period_days
    on_hand = np.round(mean0 * (R + 3) * (1 + ref.NEXUS_SAFETY_STOCK_PCT)).astype(np.int64)
    pipeline: dict[int, list] = {}
    pos = []
    po_seq = 0

    def place_po(i, t, qty, initial=False):
        nonlocal po_seq
        s = primary[i]
        nominal = int(lt_nominal[i])
        if initial:
            actual = t  # t is the arrival day for warm-up stock
            order_day = actual - nominal
            shipped = qty
        else:
            order_day = t
            h = health[s, t]
            base_h = 1 - sup._risk_base.iloc[s]
            extra = (
                rng.normal(0, lt_sigma[s])
                + 30 * max(0.0, base_h - h - 0.05)    # deteriorating supplier ships late
                + 12 * congestion[sup_country[s], t]  # port congestion
                + lt_add[s, t]                        # active disruption
            )
            arrival_guess = min(n_days - 1, t + nominal + max(0, int(extra)))
            extra += burst[s, min(arrival_guess, n_days - 1)]
            actual = t + max(int(round(nominal * 0.7)), int(round(nominal + extra)))
            # A deteriorating supplier starts short-shipping before it fails outright (leading signal).
            short_ship = float(np.clip(1 - 1.5 * max(0.0, base_h - h - 0.08), 0.5, 1.0))
            shipped = int(round(qty * cap[s, t] * short_ship))
        po_seq += 1
        pos.append((po_seq, i, s, order_day, qty, shipped, order_day + nominal + quote_buffer[s], actual, initial))
        if shipped > 0 and actual < n_days:
            pipeline.setdefault(actual, []).append((i, shipped))
        return shipped

    # Warm-up pipeline: one review's worth of stock arriving every R days until the first real orders land.
    for i in range(S):
        for arr in range(int(review_day[i]) % R, int(lt_nominal[i]) + R, R):
            place_po(i, arr, int(round(mean0[i] * R)), initial=True)

    on_order = np.zeros(S, dtype=np.int64)
    for (_po_id, i, _s, _od, _q, shp, _prom, act, _init) in pos:
        if 0 <= act < n_days:
            on_order[i] += shp

    rec = {k: np.zeros((S, n_days), dtype=np.int64) for k in ("on_hand", "on_order", "receipts", "sales", "lost_sales", "adjustment")}
    for t in range(n_days):
        arrived = np.zeros(S, dtype=np.int64)
        for i, q in pipeline.pop(t, []):
            arrived[i] += q
        on_hand += arrived
        on_order -= arrived
        d = demand[:, t]
        sales = np.minimum(on_hand, d)
        on_hand -= sales
        # Recorded cycle-count adjustments (small, observable) and unrecorded shrink (anomaly).
        adj = np.round(rng.normal(0, 0.003, S) * on_hand).astype(np.int64)
        for i, mag in shrink_by_day.get(t, []):
            on_hand[i] -= int(on_hand[i] * mag)
        on_hand = np.maximum(0, on_hand + adj)
        # Historical policy (NEXUS-style static rule): weekly review, order up to
        # trailing-28-day mean x (lead time + review) x (1 + 15% safety stock). No forecast,
        # no promotion or holiday awareness, no lead-time variance.
        due = np.where(review_day == dows[t])[0]
        if len(due):
            lo = max(0, t - 27)
            rate = demand[due, lo : t + 1].mean(axis=1)
            target = rate * (lt_nominal[due] + R) * (1 + ref.NEXUS_SAFETY_STOCK_PCT)
            need = np.round(target - (on_hand[due] + on_order[due])).astype(np.int64)
            for i, q in zip(due, need, strict=True):
                if q > 0:
                    on_order[i] += place_po(i, t, int(q))
        for k, v in (("on_hand", on_hand), ("on_order", on_order), ("receipts", arrived), ("sales", sales),
                     ("lost_sales", d - sales), ("adjustment", adj)):
            rec[k][:, t] = v

    po = pd.DataFrame(pos, columns=["po_seq", "series_idx", "supplier_idx", "order_idx", "qty_ordered",
                                    "qty_shipped", "promised_idx", "arrival_idx", "warmup"])
    return demand, rec, po, unit_cost


# ─────────────────────────────────────────────────────────────────────────────
# Assemble tables
# ─────────────────────────────────────────────────────────────────────────────
def _idx_to_date(cal, idx):
    """Day index (0 = calendar start) -> datetime64 values."""
    return (cal.date.iloc[0] + pd.to_timedelta(np.asarray(idx, dtype="int64"), unit="D")).to_numpy()


def generate_v2(cfg: GeneratorConfig) -> dict[str, pd.DataFrame]:
    ss = np.random.SeedSequence(cfg.seed)
    r_master, r_health, r_events, r_demand, r_anom, r_ops = (np.random.default_rng(s) for s in ss.spawn(6))

    cal = calendar_frame(cfg.start, cfg.end)
    n_days = len(cal)
    snap_idx = (pd.Timestamp(cfg.snapshot) - cal.date.iloc[0]).days

    suppliers = make_suppliers(cfg, r_master)
    warehouses = make_warehouses()
    products, profiles = make_products(cfg, suppliers, r_master)
    assortment = make_assortment(cfg, products, warehouses, r_master)
    promotions = make_promotions(cfg, products, cal, r_master)

    health = simulate_supplier_health(cfg, suppliers, n_days, r_health)
    events, bridge, congestion = generate_disruptions(cfg, cal, suppliers, products, health, r_events)
    series, demand, price, promo = build_demand(cal, products, profiles, assortment, promotions, events, r_demand)
    anomalies = plan_anomalies(cfg, len(series), len(suppliers), n_days, promo, r_anom)
    demand_int, rec, po, unit_cost = simulate_operations(cfg, cal, series, demand, products, suppliers,
                                                         warehouses, health, congestion, events, bridge,
                                                         anomalies, r_ops)

    dates = cal.date.to_numpy()
    S = len(series)
    long = lambda arr: arr.reshape(-1)  # noqa: E731  (series-major order)
    base = pd.DataFrame({
        "date": np.tile(dates, S),
        "sku_id": np.repeat(series.sku_id.to_numpy(), n_days),
        "warehouse_id": np.repeat(series.warehouse_id.to_numpy(), n_days),
    })
    demand_daily = base.copy()
    demand_daily["demand_units"] = long(demand_int)
    demand_daily["sales_units"] = long(rec["sales"])
    demand_daily["lost_sales_units"] = long(rec["lost_sales"])
    demand_daily["price"] = np.round(long(price), 2)
    demand_daily["is_promo"] = long(promo)
    for c in ("is_black_friday", "is_cyber_monday", "is_holiday_week"):
        demand_daily[c] = np.tile(cal[c].to_numpy(), S)

    inventory_daily = base.copy()
    for k in ("on_hand", "on_order", "receipts", "adjustment"):
        inventory_daily[f"{k}_units"] = long(rec[k])

    # Warehouse capacity derived from simulated stock so typical utilisation is 55–95%
    # (Extension: NEXUS capacities of 30k–100k units are far below v2 volumes).
    inv_by_wh = inventory_daily.groupby(["warehouse_id", "date"]).on_hand_units.sum().unstack()
    peak = inv_by_wh.max(axis=1)
    rng_cap = np.random.default_rng(ss.spawn(1)[0])
    warehouses["capacity_units"] = [int(round(peak[w] * rng_cap.uniform(1.02, 1.25), -3)) for w in warehouses.warehouse_id]
    warehouse_daily = inv_by_wh.stack().rename("on_hand_units").reset_index()
    warehouse_daily = warehouse_daily.merge(warehouses[["warehouse_id", "capacity_units"]], on="warehouse_id")
    warehouse_daily["utilization_pct"] = (100 * warehouse_daily.on_hand_units / warehouse_daily.capacity_units).round(2)
    warehouse_daily = warehouse_daily.drop(columns="capacity_units")

    # Purchase orders and shipments (one shipment per PO). Only events up to the snapshot are known.
    sup_ids = suppliers.supplier_id.to_numpy()
    po = po[(~po.warmup) & (po.order_idx <= snap_idx)].reset_index(drop=True)
    po["po_id"] = [f"PO-{k:06d}" for k in range(1, len(po) + 1)]
    po["sku_id"] = series.sku_id.to_numpy()[po.series_idx]
    po["warehouse_id"] = series.warehouse_id.to_numpy()[po.series_idx]
    po["supplier_id"] = sup_ids[po.supplier_idx]
    po["order_date"] = _idx_to_date(cal, po.order_idx)
    po["promised_date"] = _idx_to_date(cal, po.promised_idx)
    arrived = po.arrival_idx <= snap_idx
    po["received_date"] = pd.Series(_idx_to_date(cal, po.arrival_idx)).where(arrived).to_numpy()
    po["unit_cost"] = unit_cost[po.series_idx]
    purchase_orders = po[["po_id", "sku_id", "warehouse_id", "supplier_id", "order_date", "promised_date",
                          "received_date", "qty_ordered", "qty_shipped", "unit_cost"]].copy()
    purchase_orders["status"] = np.where(arrived, "RECEIVED", "OPEN")

    srng = np.random.default_rng(ss.spawn(2)[1])
    wh_region = warehouses.set_index("warehouse_id").region
    same_region = suppliers.set_index("supplier_id").region.loc[po.supplier_id].to_numpy() == \
        wh_region.loc[po.warehouse_id].to_numpy()
    mode = np.where(same_region, srng.choice(["ROAD", "RAIL"], len(po)), srng.choice(["SEA", "SEA", "SEA", "AIR"], len(po)))
    delay = (po.arrival_idx - po.promised_idx).to_numpy()
    status = np.where(arrived, "DELIVERED",
                      np.where(snap_idx > po.promised_idx.to_numpy(), "DELAYED",
                               np.where(po.arrival_idx.to_numpy() - snap_idx <= 3, "AT_PORT", "IN_TRANSIT")))
    status = np.where(po.qty_shipped.to_numpy() == 0, "CANCELLED", status)
    shipments = pd.DataFrame({
        "shipment_id": [f"SHP-{k:06d}" for k in range(1, len(po) + 1)],
        "po_id": po.po_id.to_numpy(),
        "supplier_id": po.supplier_id.to_numpy(),
        "warehouse_id": po.warehouse_id.to_numpy(),
        "sku_id": po.sku_id.to_numpy(),
        "origin_country": suppliers.set_index("supplier_id").country.loc[po.supplier_id].to_numpy(),
        "origin_port": suppliers.set_index("supplier_id").primary_port.loc[po.supplier_id].to_numpy(),
        "carrier": srng.choice(ref.CARRIER_NAMES, len(po)),
        "transport_mode": mode,
        "ship_date": _idx_to_date(cal, po.order_idx + np.round(0.3 * (po.promised_idx - po.order_idx)).astype(int)),
        "eta": po.promised_date.to_numpy(),
        "actual_arrival": po.received_date.to_numpy(),
        "status": status,
        "quantity_units": po.qty_shipped.to_numpy(),
        "value_usd": np.round(po.qty_shipped.to_numpy() * po.unit_cost.to_numpy(), 2),
        # Delay is only known once the shipment has arrived (or once it's overdue at the snapshot).
        "delay_days": np.where(arrived, delay, np.where(status == "DELAYED", snap_idx - po.promised_idx.to_numpy(), 0)),
    })
    shipments["is_delayed"] = shipments.delay_days > 0

    # Disruptions table (observable once started).
    ev = events[events.start_idx <= snap_idx].copy()
    ev["detected_at"] = _idx_to_date(cal, ev.start_idx)
    ev["resolved_at"] = pd.Series(_idx_to_date(cal, ev.end_idx)).where((ev.end_idx < snap_idx).to_numpy()).to_numpy()
    ev["status"] = np.where(ev.end_idx < snap_idx, "RESOLVED", "ACTIVE")
    # Impact estimated from the simulation: value of shipments ordered from affected suppliers during the event.
    impacts = []
    for e in ev.itertuples():
        affected = bridge.loc[bridge.event_id == e.event_id, "supplier_id"]
        m = po.supplier_id.isin(affected) & po.order_idx.between(e.start_idx, e.end_idx)
        impacts.append(float((po.loc[m, "qty_ordered"] * po.loc[m, "unit_cost"]).sum()))
    ev["estimated_impact_usd"] = np.round(impacts, 0)
    ev["affected_entities"] = [json.dumps(list(bridge.loc[bridge.event_id == e, "supplier_id"])) for e in ev.event_id]
    disruptions = ev[["event_id", "event_type", "severity", "title", "scope", "affected_category", "detected_at",
                      "resolved_at", "status", "capacity_factor", "lead_time_add_days", "demand_multiplier",
                      "estimated_impact_usd", "affected_entities"]].reset_index(drop=True)
    disruption_suppliers = bridge[bridge.event_id.isin(disruptions.event_id)].reset_index(drop=True)

    # Daily network KPIs, derived from the simulation (replaces NEXUS's built-in trend, D7).
    dd = demand_daily.groupby("date").agg(demand=("demand_units", "sum"), sales=("sales_units", "sum"))
    stockout = (demand_daily.lost_sales_units > 0).groupby(demand_daily.date).mean()
    cost_map = products.set_index("sku_id").unit_cost
    inv_val = (inventory_daily.on_hand_units * inventory_daily.sku_id.map(cost_map)).groupby(inventory_daily.date).sum()
    sh = shipments[shipments.status == "DELIVERED"].copy()
    sh["qty_ordered"] = po.qty_ordered.to_numpy()[shipments.status.to_numpy() == "DELIVERED"]
    sh["on_time_in_full"] = (sh.delay_days <= 0) & (sh.quantity_units >= 0.98 * sh.qty_ordered)
    otif = sh.groupby("actual_arrival").on_time_in_full.mean()
    active = np.zeros(n_days, dtype=int)
    for e in events.itertuples():
        active[e.start_idx : e.end_idx + 1] += 1
    kpi_daily = pd.DataFrame({"date": cal.date})
    kpi_daily["fill_rate_pct"] = (100 * dd.sales / dd.demand).round(2).to_numpy()
    kpi_daily["stockout_rate_pct"] = (100 * stockout).round(2).to_numpy()
    kpi_daily["inventory_value_usd"] = inv_val.round(0).to_numpy()
    kpi_daily["warehouse_utilization_pct"] = warehouse_daily.groupby("date").utilization_pct.mean().round(2).to_numpy()
    kpi_daily["otif_pct"] = (100 * otif.reindex(cal.date).rolling(7, min_periods=1).mean()).round(2).to_numpy()
    kpi_daily["active_disruptions"] = active
    kpi_daily = kpi_daily[kpi_daily.date <= pd.Timestamp(cfg.snapshot)]

    # Supplier master: computed statistics replace NEXUS random draws.
    otd = sh.groupby("supplier_id").apply(lambda g: (g.delay_days <= 0).mean(), include_groups=False)
    spend_2025 = po[po.order_date >= pd.Timestamp("2025-01-01")].assign(v=lambda x: x.qty_ordered * x.unit_cost) \
        .groupby("supplier_id").v.sum()
    suppliers["on_time_delivery_rate"] = suppliers.supplier_id.map(otd).round(3)
    suppliers["annual_spend_usd"] = suppliers.supplier_id.map(spend_2025).fillna(0).round(0)
    daily_units = demand_daily.groupby(demand_daily.sku_id.map(products.set_index("sku_id").primary_supplier_id)) \
        .demand_units.sum() / n_days
    suppliers["capacity_units_per_day"] = [int(round(daily_units.get(s, 500) * f, -1)) for s, f in
                                           zip(suppliers.supplier_id, np.random.default_rng(cfg.seed + 7).uniform(1.3, 2.0, len(suppliers)), strict=True)]

    # Hidden truth (never a model input).
    truth_health = pd.DataFrame({
        "date": np.tile(dates, len(suppliers)),
        "supplier_id": np.repeat(sup_ids, n_days),
        "latent_health": np.round(health.reshape(-1), 4),
    })
    an = anomalies.copy()
    an.insert(0, "anomaly_id", [f"ANM-{k:04d}" for k in range(1, len(an) + 1)])
    an["entity_id"] = [sup_ids[r.entity_idx] if r.entity_type == "supplier"
                       else f"{series.sku_id.iloc[r.entity_idx]}@{series.warehouse_id.iloc[r.entity_idx]}"
                       for r in an.itertuples()]
    an["start_date"] = _idx_to_date(cal, an.start_idx)
    an["end_date"] = _idx_to_date(cal, an.start_idx + an.length - 1)
    injected = an[["anomaly_id", "type", "entity_type", "entity_id", "start_date", "end_date", "magnitude"]]
    disruption_starts = events[events.event_type != "DEMAND"][["event_id", "event_type", "severity", "start_idx", "end_idx"]].copy()
    disruption_starts["start_date"] = _idx_to_date(cal, disruption_starts.start_idx)
    disruption_starts["end_date"] = _idx_to_date(cal, disruption_starts.end_idx)
    disruption_truth = disruption_starts.drop(columns=["start_idx", "end_idx"]).merge(bridge, on="event_id")

    suppliers = suppliers.drop(columns=["_risk_base"])
    tables = {
        "suppliers": suppliers, "warehouses": warehouses, "products": products, "sku_warehouse": series,
        "promotions": promotions, "calendar": cal, "demand_daily": demand_daily,
        "inventory_daily": inventory_daily, "warehouse_daily": warehouse_daily,
        "purchase_orders": purchase_orders, "shipments": shipments, "disruptions": disruptions,
        "disruption_suppliers": disruption_suppliers, "kpi_daily": kpi_daily,
    }
    # Observable tables stop at the snapshot date.
    for name in ("demand_daily", "inventory_daily", "warehouse_daily"):
        tables[name] = tables[name][tables[name].date <= pd.Timestamp(cfg.snapshot)].reset_index(drop=True)
    truth = {"supplier_latent_daily": truth_health, "injected_anomalies": injected, "product_profiles": profiles,
             "all_disruption_events": disruption_truth}
    return {"tables": tables, "truth": truth}


# ─────────────────────────────────────────────────────────────────────────────
# nexus_v1: original schema and distributions (for comparison)
# ─────────────────────────────────────────────────────────────────────────────
def generate_v1(cfg: GeneratorConfig) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    snap = pd.Timestamp(cfg.snapshot)
    countries = list(ref.COUNTRIES)
    sup_rows = []
    for i in range(50):
        c = countries[rng.integers(12)]
        cat = ref.PRODUCT_CATEGORIES[rng.integers(10)]
        rb = ref.COUNTRIES[c]["risk_base"] + rng.uniform(-0.1, 0.2)
        score = int(np.clip((1 - rb) * 100 + rng.normal(0, 8), 5, 99))
        sup_rows.append({"id": f"SUP-{i + 1:04d}", "country": c, "region": ref.COUNTRIES[c]["region"],
                         "category": cat["name"], "risk_score": score, "risk_level": _risk_level(score),
                         "on_time_delivery_rate": round(float(np.clip(1 - rb + rng.normal(0, 0.05), 0.4, 0.99)), 3),
                         "avg_lead_time_days": cat["lead_time_days"] + int(rng.integers(-5, 16)),
                         "annual_spend_usd": int(rng.integers(500_000, 15_000_001)),
                         "primary_port": str(rng.choice(ref.COUNTRIES[c]["ports"]))})
    suppliers = pd.DataFrame(sup_rows)
    wh = make_warehouses().rename(columns={"warehouse_id": "id"})
    util = rng.uniform(0.45, 0.95, len(wh))
    wh["capacity_units"] = rng.choice(ref.NEXUS_WAREHOUSE_CAPACITIES, len(wh))
    wh["utilization_pct"] = np.round(util * 100, 1)
    statuses = ["IN_TRANSIT", "AT_PORT", "CUSTOMS", "DELIVERED", "DELAYED", "EXCEPTION"]
    st = rng.choice(statuses, 300, p=[0.45, 0.15, 0.10, 0.20, 0.08, 0.02])
    shipments = pd.DataFrame({
        "id": [f"SHP-{i + 1:05d}" for i in range(300)],
        "supplier_id": rng.choice(suppliers.id, 300), "destination_warehouse_id": rng.choice(wh.id, 300),
        "status": st, "delay_days": np.where(np.isin(st, ["DELAYED", "EXCEPTION"]), rng.integers(1, 9, 300), 0),
        "delay_risk_score": np.round(rng.uniform(0, 1, 300), 3),  # D2: pure noise, as in NEXUS
        "value_usd": rng.integers(50_000, 5_000_001, 300),
    })
    shipments["is_delayed"] = shipments.delay_days > 0
    sev = rng.choice(SEVERITIES, 50, p=SEV_P)
    days_ago = rng.integers(0, 61, 50)
    dur = np.array([ref.SEVERITY_DURATION_DAYS[s] for s in sev])
    disruptions = pd.DataFrame({
        "id": [f"DISR-{i + 1:04d}" for i in range(50)],
        "event_type": rng.choice(list(ref.DISRUPTION_TYPES), 50), "severity": sev,
        "status": np.where(days_ago < dur, "ACTIVE", "RESOLVED"),
        "affected_entities": [json.dumps([f"SUP-{int(x):04d}" for x in rng.integers(1, 51, rng.integers(1, 5))]) for _ in range(50)],
        "detected_at": [snap - pd.Timedelta(days=int(d)) for d in days_ago],
    })
    dates = pd.date_range(end=snap, periods=365)
    cats = [c["name"] for c in ref.PRODUCT_CATEGORIES]
    rows = []
    t = (dates - dates[0]).days.to_numpy()
    dowf = np.array(ref.DOW_FACTORS)[dates.dayofweek]
    seasonal = 1 + ref.MONTHLY_SINE_AMPLITUDE * np.sin(2 * np.pi * dates.month.to_numpy() / 12)
    bf = (dates.month == 11) & (dates.day >= 22) & (dates.dayofweek == 4)  # NEXUS rule kept in v1
    cmd = (dates.month == 11) & (dates.day >= 25) & (dates.dayofweek == 0)
    for k in range(100):
        base = rng.uniform(100, 2000)
        cat = cats[rng.integers(10)]
        for w in wh.id[:4]:  # D4: only 4 warehouses, as in NEXUS
            q = base * (1 + ref.ANNUAL_TREND * t / 365) * dowf * seasonal * np.where(bf, 3.4, 1) * np.where(cmd, 2.8, 1) \
                * rng.lognormal(0, ref.DEMAND_NOISE_SIGMA, len(dates))
            rows.append(pd.DataFrame({"date": dates, "sku_id": f"SKU-{k:04d}", "warehouse_id": w, "category": cat,
                                      "quantity": np.round(q).astype(int),
                                      "price": np.round(rng.uniform(10, 1000, len(dates)), 2)}))  # D1
    demand = pd.concat(rows, ignore_index=True)
    kd = pd.date_range(end=snap, periods=90)
    prog = np.arange(90) / 90
    kpi = pd.DataFrame({"date": kd, "otif_pct": np.round(np.clip(82 + prog * 12 + rng.normal(0, 2, 90), 70, 99), 2),
                        "fill_rate_pct": np.round(np.clip(88 + prog * 8 + rng.normal(0, 1.5, 90), 80, 99), 2)})  # D7
    return {"tables": {"suppliers": suppliers, "warehouses": wh, "shipments": shipments, "disruptions": disruptions,
                       "demand_history": demand, "kpi_history": kpi}, "truth": {}}


# ─────────────────────────────────────────────────────────────────────────────
def write(result: dict, out_dir: Path, truth_dir: Path | None, cfg: GeneratorConfig, mode: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"mode": mode, "generator_version": __version__, "config": json.loads(json.dumps(asdict(cfg), default=str)),
                "config_fingerprint": cfg.fingerprint(), "tables": {}, "truth_tables": {}}
    for name, df in result["tables"].items():
        path = out_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        manifest["tables"][name] = {"rows": len(df), "columns": list(df.columns),
                                    "sha256": hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:16]}
    if truth_dir is not None and result["truth"]:
        truth_dir.mkdir(parents=True, exist_ok=True)
        for name, df in result["truth"].items():
            df.to_parquet(truth_dir / f"{name}.parquet", index=False)
            manifest["truth_tables"][name] = {"rows": len(df)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["nexus_v2", "nexus_v1"], default="nexus_v2")
    parser.add_argument("--out", default=None, help="output folder (default: data/<mode>)")
    args = parser.parse_args()
    settings = get_settings()
    cfg = GeneratorConfig.from_settings()
    out = Path(args.out) if args.out else settings.data_dir / args.mode
    result = generate_v2(cfg) if args.mode == "nexus_v2" else generate_v1(cfg)
    truth_dir = (out.parent / "truth") if args.mode == "nexus_v2" else None
    manifest = write(result, out, truth_dir, cfg, args.mode)
    for name, info in manifest["tables"].items():
        print(f"  {name:<22}{info['rows']:>10,} rows")
    for name, info in manifest["truth_tables"].items():
        print(f"  truth/{name:<16}{info['rows']:>10,} rows")
    print(f"Written to {out}  (fingerprint {manifest['config_fingerprint']})")


if __name__ == "__main__":
    main()
