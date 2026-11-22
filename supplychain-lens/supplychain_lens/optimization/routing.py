# Idea adapted from NEXUS-SCCT route_optimizer.py (haversine distance, per-mode speed/cost/CO2
# scoring); reimplemented with recomputed options instead of fixed +12% / +18% deltas.
"""Transport-mode options for a shipment lane (supplier country -> warehouse).

Mode parameters are planning assumptions (documented, adjustable), not carrier quotes:
  speed (km/day), cost index per unit-km relative to sea, CO2 index relative to sea.
"""

from __future__ import annotations

import pandas as pd

from supplychain_lens.data.reference import COUNTRY_COORDS
from supplychain_lens.optimization.transfers import haversine_km

MODES = {
    "SEA": {"speed_km_day": 550, "handling_days": 6, "cost_index": 1.0, "co2_index": 1.0},
    "RAIL": {"speed_km_day": 900, "handling_days": 3, "cost_index": 1.8, "co2_index": 0.7},
    "ROAD": {"speed_km_day": 700, "handling_days": 1, "cost_index": 3.0, "co2_index": 2.5},
    "AIR": {"speed_km_day": 8000, "handling_days": 2, "cost_index": 12.0, "co2_index": 25.0},
}
SEA_COST_PER_UNIT_PER_1000KM_SHARE = 0.002  # sea freight ≈ 0.2% of unit value per 1,000 km (assumption)


def mode_options(origin_country: str, dest_lat: float, dest_lng: float, dest_region: str, origin_region: str,
                 unit_value: float, units: float) -> pd.DataFrame:
    lat, lng = COUNTRY_COORDS[origin_country]
    km = haversine_km(lat, lng, dest_lat, dest_lng) * 1.25  # route factor over great-circle distance
    rows = []
    for mode, p in MODES.items():
        if mode in ("ROAD", "RAIL") and origin_region != dest_region:
            continue  # no land route between regions in this model
        days = p["handling_days"] + km / p["speed_km_day"]
        cost = units * unit_value * SEA_COST_PER_UNIT_PER_1000KM_SHARE * (km / 1000) * p["cost_index"]
        rows.append({"mode": mode, "distance_km": round(km), "transit_days": round(days, 1),
                     "freight_cost_usd": round(cost, 0), "co2_index": p["co2_index"]})
    return pd.DataFrame(rows).sort_values("transit_days").reset_index(drop=True)
