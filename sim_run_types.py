# sim_run_grok_types.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class StoreConfig:
    key: str
    capacity: float
    opening_low: float
    opening_high: float

@dataclass
class ProductionCandidate:
    product: str
    out_store_key: str
    in_store_key: Optional[str] = None
    rate_tph: float = 0.0
    consumption_pct: float = 1.0

@dataclass
class MakeUnit:
    location: str
    equipment: str
    candidates: List[ProductionCandidate]
    choice_rule: str = "min_fill_pct"
    step_hours: float = 1.0

@dataclass
class TransportRoute:
    product: str
    origin_location: str
    dest_location: str
    origin_stores: List[str]
    dest_stores: List[str]
    n_units: int
    payload_t: float
    load_rate_tph: float
    unload_rate_tph: float
    to_min: float
    back_min: float
    mode: str = "TRAIN"
    route_group: Optional[str] = None
    speed_knots: Optional[float] = None
    hulls_per_vessel: Optional[int] = None
    payload_per_hull_t: Optional[float] = None
    itineraries: Optional[List[List[dict]]] = None
    berth_info: Optional[Dict[str, dict]] = None
    nm_distance: Optional[Dict[tuple, float]] = None

@dataclass
class Demand:
    store_key: str
    rate_per_hour: float