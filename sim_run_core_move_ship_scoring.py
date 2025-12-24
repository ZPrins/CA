# sim_run_core_move_ship_scoring.py
"""
Ship route scoring and selection logic.

Evaluates candidate routes based on utilization, urgency, travel time, and overflow risk.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import math
import simpy

from sim_run_core_move_ship_utils import get_nm_distance, get_pilot_hours


def calculate_route_score(itinerary: List[Dict], stores: Dict[str, simpy.Container],
                          payload_per_hold: float, n_holds: int, demand_rates: Dict[str, float],
                          nm_distances: Dict, speed_knots: float, berth_info: Dict,
                          sole_supplier_stores: Optional[set] = None,
                          production_rates: Optional[Dict[str, float]] = None,
                          store_capacities: Optional[Dict[str, float]] = None,
                          other_pending: Optional[Dict[str, float]] = None,
                          current_cargo: Optional[Dict[str, float]] = None) -> Tuple[
    float, float, float, float, float]:
    """
    Score a route based on utilization, urgency, travel time, and origin overflow risk.
    Returns (score, utilization_pct, urgency_score, overflow_bonus, suggested_load)
    """
    total_cargo_already_onboard = sum(current_cargo.values()) if current_cargo else 0.0
    total_available = total_cargo_already_onboard
    total_capacity = float(n_holds * payload_per_hold)

    urgency = 0.0
    sole_supplier_bonus = 0.0
    overflow_bonus = 0.0
    travel_time = 0.0

    prod_rates = production_rates or {}
    capacities = store_capacities or {}
    pending_map = other_pending or {}

    # Pre-calculate itinerary travel time to estimate arrival at each stop
    # Add a conservative estimate for berth waiting and loading time
    current_time_offset = 0.0
    total_projected_headspace = 0.0
    has_unload = False
    can_unload_existing_cargo = False

    for step in itinerary:
        kind = step.get('kind')
        sk = step.get('store_key')
        prod = step.get('product')

        if kind == 'sail':
            from_loc = step.get('from', '')
            to_loc = step.get('to', step.get('location', ''))
            if from_loc and to_loc:
                nm = get_nm_distance(nm_distances, from_loc, to_loc)
                pilot_out = get_pilot_hours(berth_info, from_loc, 'out')
                pilot_in = get_pilot_hours(berth_info, to_loc, 'in')
                step_time = (nm / max(speed_knots, 1.0)) + pilot_out + pilot_in
                travel_time += step_time
                current_time_offset += step_time

        elif kind == 'load' and sk and sk in stores:
            # For loading, we care about current levels + production during wait/load
            level = float(stores[sk].level)
            total_available += level

            prod_rate = prod_rates.get(sk, 0.0)
            capacity = capacities.get(sk, float('inf'))
            if prod_rate > 0 and capacity < float('inf'):
                # Heuristic: projection of overflow risk
                fill_pct = level / capacity
                if fill_pct > 0.50:
                    bonus = 150 * (fill_pct - 0.50) / 0.50
                    if bonus > overflow_bonus:
                        overflow_bonus = bonus

        elif kind == 'unload' and sk and sk in stores:
            has_unload = True
            # For unloading, we care about PROJECTED level at arrival
            level = float(stores[sk].level)
            rate = demand_rates.get(sk, 0.0)
            capacity = capacities.get(sk, float('inf'))
            prod_rate = prod_rates.get(sk, 0.0)  # Factory production into this store

            # Project level at arrival: current + (prod - consumption) * travel_time + pending deliveries from others
            projected_level = level + (prod_rate - rate) * current_time_offset + pending_map.get(sk, 0.0)
            projected_level = max(0.0, projected_level)

            # Use a safety buffer: only target 100% of capacity to account for variance
            effective_capacity = capacity * 1.0
            headspace = max(0.0, effective_capacity - projected_level)
            total_projected_headspace += headspace

            # Check if this step helps unload existing cargo
            if current_cargo and prod in current_cargo and current_cargo[prod] > 1e-6:
                can_unload_existing_cargo = True

            if rate > 0:
                days_of_stock = projected_level / (rate * 24)
                step_urgency = 10 - days_of_stock
                if step_urgency > urgency:
                    urgency = step_urgency

                if sole_supplier_stores and sk in sole_supplier_stores:
                    if days_of_stock < 60:
                        bonus = 100 * (1 - days_of_stock / 60)
                        if bonus > sole_supplier_bonus:
                            bonus = 100 * (1 - days_of_stock / 60)
                        if bonus > sole_supplier_bonus:
                            sole_supplier_bonus = bonus

            # Penalize if projected to be full or nearly full
            # [Optimization]: Removed heavy global urgency penalty as it blocks multi-product routes.
            # Headspace factor in score already handles this.
            if capacity < float('inf'):
                projected_fill = projected_level / capacity
                if projected_fill > 0.95:
                    urgency -= 2.0  # Mild discouragement for full stores

    # If we have existing cargo and this route CANNOT unload it, penalize heavily
    if total_cargo_already_onboard > 1e-6 and not can_unload_existing_cargo:
        urgency -= 1000.0  # Force picking a route that unloads our cargo

    # Suggest load: limited by projected headspace and total available at source
    # [Issue Fix]: Be slightly more optimistic about headspace if demand is significant
    # or if we have plenty of source stock. We want to maximize vessel utilization.
    suggested_load = total_available
    if has_unload:
        # Increase effective headspace by a small factor (e.g. 20%) to allow for
        # optimism that demand/production will clear space during the voyage
        optimistic_headspace = total_projected_headspace * 1.25
        suggested_load = min(suggested_load, optimistic_headspace)

    suggested_load = min(suggested_load, total_capacity)
    # Quantize to hold sizes
    suggested_load = math.floor(suggested_load / payload_per_hold) * payload_per_hold

    # What we actually NEED to load is the difference
    suggested_to_load = max(0.0, suggested_load - total_cargo_already_onboard)

    # Final score components
    # Re-calculate utilization based on suggested_load (which includes already onboard)
    utilization = min(1.0, suggested_load / max(total_capacity, 1.0))
    urgency = max(-2000.0, urgency)  # Allow negative urgency to discourage bad routes

    # Retrieve scoring weights from settings if available
    util_weight = 100.0
    urgency_weight = 30.0
    travel_time_penalty = 0.1

    # Try to get from sim settings if passed (not always available in this helper)
    # The helper currently doesn't receive the full settings dict, so we'll stick to
    # the existing logic but document that these should ideally be in Settings.

    score = (utilization * util_weight) + (urgency * urgency_weight) + sole_supplier_bonus + overflow_bonus - (travel_time * travel_time_penalty)

    # Apply headspace factor: if we can't even fill one hold because of destination headspace,
    # the route is essentially worthless for moving product.
    headspace_factor = 1.0
    if has_unload and total_capacity > 0:
        # Also be more relaxed about the headspace factor penalty
        headspace_factor = min(1.0, (total_projected_headspace * 1.2) / total_capacity)
        if headspace_factor < 0.3:
            headspace_factor *= 0.5

    if score > 0:
        score *= headspace_factor

    return (score, utilization, urgency, overflow_bonus, suggested_to_load)

