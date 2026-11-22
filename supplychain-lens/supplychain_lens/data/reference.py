# Design adapted from NEXUS-SCCT by Syed Zain Ahmed
# (https://github.com/ahmedzainsyed/nexus-scct), seed_data.py.
# Reimplemented and extended in SupplyChain Lens.
"""Reference tables of the NEXUS-SCCT data model.

The values below are *facts and parameters* of the NEXUS synthetic world (country risk
baselines, ports, category values and lead times, disruption types, carriers and the
12 warehouse locations). They are kept identical to NEXUS so that SupplyChain Lens uses
the same underlying data model. Additions are marked "Extension".
"""

from __future__ import annotations

# Country risk baseline (0 = safe, 1 = risky), region, currency and ports — NEXUS values.
COUNTRIES: dict[str, dict] = {
    "China": {"region": "APAC", "risk_base": 0.45, "currency": "CNY", "ports": ["Shanghai", "Shenzhen", "Guangzhou"]},
    "Vietnam": {"region": "APAC", "risk_base": 0.35, "currency": "VND", "ports": ["Ho Chi Minh City", "Haiphong"]},
    "Germany": {"region": "EMEA", "risk_base": 0.08, "currency": "EUR", "ports": ["Hamburg", "Bremen"]},
    "India": {"region": "APAC", "risk_base": 0.32, "currency": "INR", "ports": ["Mumbai", "Chennai"]},
    "Brazil": {"region": "LATAM", "risk_base": 0.52, "currency": "BRL", "ports": ["Santos", "Rio de Janeiro"]},
    "South Korea": {"region": "APAC", "risk_base": 0.12, "currency": "KRW", "ports": ["Busan", "Incheon"]},
    "Netherlands": {"region": "EMEA", "risk_base": 0.06, "currency": "EUR", "ports": ["Rotterdam", "Amsterdam"]},
    "Japan": {"region": "APAC", "risk_base": 0.10, "currency": "JPY", "ports": ["Yokohama", "Tokyo", "Nagoya"]},
    "Turkey": {"region": "EMEA", "risk_base": 0.48, "currency": "TRY", "ports": ["Istanbul", "Mersin"]},
    "Mexico": {"region": "LATAM", "risk_base": 0.38, "currency": "MXN", "ports": ["Manzanillo", "Veracruz"]},
    "USA": {"region": "Americas", "risk_base": 0.07, "currency": "USD", "ports": ["Los Angeles", "Houston", "New York"]},
    "Taiwan": {"region": "APAC", "risk_base": 0.55, "currency": "TWD", "ports": ["Kaohsiung", "Keelung"]},
}

# Extension: approximate country centroids, used only to draw supplier markers on maps.
COUNTRY_COORDS: dict[str, tuple[float, float]] = {
    "China": (31.0, 117.0), "Vietnam": (14.1, 108.3), "Germany": (51.2, 10.4), "India": (20.6, 78.9),
    "Brazil": (-14.2, -51.9), "South Korea": (36.5, 127.9), "Netherlands": (52.1, 5.3), "Japan": (36.2, 138.3),
    "Turkey": (39.0, 35.2), "Mexico": (23.6, -102.6), "USA": (39.8, -98.6), "Taiwan": (23.7, 121.0),
}

# Category average unit value (USD), fragility and nominal supplier lead time (days) — NEXUS values.
PRODUCT_CATEGORIES: list[dict] = [
    {"name": "Electronics", "avg_value": 850, "fragile": True, "lead_time_days": 45},
    {"name": "Semiconductors", "avg_value": 2200, "fragile": True, "lead_time_days": 60},
    {"name": "Textiles", "avg_value": 85, "fragile": False, "lead_time_days": 30},
    {"name": "Machinery", "avg_value": 15000, "fragile": False, "lead_time_days": 90},
    {"name": "Auto Parts", "avg_value": 320, "fragile": False, "lead_time_days": 35},
    {"name": "Chemicals", "avg_value": 450, "fragile": False, "lead_time_days": 25},
    {"name": "Raw Materials", "avg_value": 180, "fragile": False, "lead_time_days": 20},
    {"name": "Precision Parts", "avg_value": 1200, "fragile": True, "lead_time_days": 50},
    {"name": "Metals", "avg_value": 280, "fragile": False, "lead_time_days": 28},
    {"name": "Components", "avg_value": 650, "fragile": True, "lead_time_days": 40},
]
CATEGORY_BY_NAME = {c["name"]: c for c in PRODUCT_CATEGORIES}

# Disruption types and titles — NEXUS values.
DISRUPTION_TYPES: dict[str, list[str]] = {
    "WEATHER": ["Typhoon in East China Sea", "Hurricane in Gulf of Mexico", "Severe flooding in Vietnam",
                "Blizzard affecting Northeast US ports"],
    "GEOPOLITICAL": ["Trade restrictions imposed", "Port workers strike", "Customs delays at border",
                     "Sanctions affecting supplier region"],
    "SUPPLIER": ["Factory shutdown", "Quality recall issued", "Capacity reduction", "Financial distress reported"],
    "PORT": ["Port congestion", "Equipment failure at terminal", "Cyber attack on port systems", "Customs backlog"],
    "DEMAND": ["Unexpected demand surge", "Product launch demand spike", "Stockout risk identified",
               "Demand drop due to recession signal"],
    "LOGISTICS": ["Container shortage", "Carrier capacity reduction", "Fuel price spike disrupting economics",
                  "Driver shortage"],
}

# Extension (fixes audit defect D6): which countries each weather title can hit.
WEATHER_FOOTPRINT: dict[str, list[str]] = {
    "Typhoon in East China Sea": ["China", "Taiwan", "Japan", "South Korea"],
    "Hurricane in Gulf of Mexico": ["Mexico", "USA"],
    "Severe flooding in Vietnam": ["Vietnam"],
    "Blizzard affecting Northeast US ports": ["USA"],
}

SEVERITY_WEIGHTS = {"LOW": 0.25, "MEDIUM": 0.35, "HIGH": 0.28, "CRITICAL": 0.12}  # NEXUS
SEVERITY_DURATION_DAYS = {"LOW": 2, "MEDIUM": 5, "HIGH": 12, "CRITICAL": 21}  # NEXUS (v1); v2 draws around these

CARRIER_NAMES = ["Maersk", "MSC", "CMA CGM", "COSCO", "Evergreen", "Hapag-Lloyd", "ONE", "Yang Ming"]  # NEXUS
CERTIFICATIONS = ["ISO9001", "ISO14001", "CTPAT", "AEO", "SA8000"]  # NEXUS

# The 12 NEXUS warehouse locations. Names follow NEXUS: first 4 "Hub", next 4 "DC", last 4 "Gateway".
WAREHOUSE_LOCATIONS: list[dict] = [
    {"city": "Chicago", "state": "IL", "country": "USA", "lat": 41.878, "lng": -87.629, "region": "Americas"},
    {"city": "Dallas", "state": "TX", "country": "USA", "lat": 32.776, "lng": -96.797, "region": "Americas"},
    {"city": "Atlanta", "state": "GA", "country": "USA", "lat": 33.749, "lng": -84.388, "region": "Americas"},
    {"city": "Seattle", "state": "WA", "country": "USA", "lat": 47.606, "lng": -122.332, "region": "Americas"},
    {"city": "Los Angeles", "state": "CA", "country": "USA", "lat": 34.052, "lng": -118.244, "region": "Americas"},
    {"city": "Frankfurt", "state": "HE", "country": "Germany", "lat": 50.110, "lng": 8.682, "region": "EMEA"},
    {"city": "Amsterdam", "state": "NH", "country": "Netherlands", "lat": 52.370, "lng": 4.895, "region": "EMEA"},
    {"city": "London", "state": "ENG", "country": "UK", "lat": 51.508, "lng": -0.128, "region": "EMEA"},
    {"city": "Singapore", "state": "SG", "country": "Singapore", "lat": 1.352, "lng": 103.820, "region": "APAC"},
    {"city": "Dubai", "state": "DU", "country": "UAE", "lat": 25.204, "lng": 55.270, "region": "EMEA"},
    {"city": "Shanghai", "state": "SH", "country": "China", "lat": 31.224, "lng": 121.469, "region": "APAC"},
    {"city": "Tokyo", "state": "TK", "country": "Japan", "lat": 35.689, "lng": 139.692, "region": "APAC"},
]
NEXUS_WAREHOUSE_CAPACITIES = [30000, 40000, 50000, 60000, 75000, 100000]  # NEXUS choices (v1)

# NEXUS demand shape: day-of-week factors (Mon..Sun), monthly sine amplitude, holiday multipliers.
DOW_FACTORS = [1.15, 1.10, 1.05, 1.0, 0.90, 0.72, 0.65]
MONTHLY_SINE_AMPLITUDE = 0.20
BLACK_FRIDAY_MULTIPLIER = 3.4
CYBER_MONDAY_MULTIPLIER = 2.8
ANNUAL_TREND = 0.015  # +1.5% per year
DEMAND_NOISE_SIGMA = 0.12  # lognormal

# NEXUS warehouse policy parameters: reorder point 30%, safety stock 15%.
NEXUS_SAFETY_STOCK_PCT = 0.15
NEXUS_REORDER_POINT_PCT = 0.30
