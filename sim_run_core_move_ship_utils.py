# sim_run_core_move_ship_utils.py
"""
Ship transport utility functions.

Helper functions for berths, pilot hours, distances, store rates, and itinerary parsing.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import simpy


def get_berth(env: simpy.Environment, port_berths: Dict[str, simpy.Resource],
              berth_info: Dict[str, dict], location: str) -> simpy.Resource:
    """Get or create FIFO berth queue for a location."""
    if location not in port_berths:
        info = berth_info.get(location, {})
        cap = max(1, int(info.get('berths', 1) or 1))
        port_berths[location] = simpy.Resource(env, capacity=cap)
    return port_berths[location]


def get_pilot_hours(berth_info: Dict[str, dict], location: str, phase: str) -> float:
    """Get pilot hours for a location (in or out)."""
    info = berth_info.get(location, {})
    if phase == 'in':
        return float(info.get('pilot_in_h', 0.0) or 0.0)
    else:
        return float(info.get('pilot_out_h', 0.0) or 0.0)


def get_nm_distance(nm_distances: Dict[Tuple[str, str], float], loc_a: str, loc_b: str) -> float:
    """Get nautical mile distance between two locations."""
    return float(nm_distances.get((loc_a, loc_b), 0.0) or nm_distances.get((loc_b, loc_a), 0.0) or 0.0)


def get_store_rates(store_rates: Dict[str, Tuple[float, float]], store_key: str,
                    default_load: float, default_unload: float) -> Tuple[float, float]:
    """Get load/unload rates for a store. Returns (load_rate, unload_rate)."""
    if store_key in store_rates:
        rates = store_rates[store_key]
        return (float(rates[0] or default_load), float(rates[1] or default_unload))
    return (default_load, default_unload)


def get_start_location(itinerary: List[Dict]) -> Optional[str]:
    """Extract start location from itinerary."""
    for step in itinerary:
        if step.get('kind') == 'start':
            return step.get('location')
    return None


def get_route_id_from_itinerary(itinerary: List[Dict]) -> Optional[str]:
    """Extract specific route_id from itinerary start step."""
    for step in itinerary:
        if step.get('kind') == 'start' and 'route_id' in step:
            return str(step['route_id'])
    return None

