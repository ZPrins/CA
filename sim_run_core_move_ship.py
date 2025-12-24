# sim_run_core_move_ship.py
"""
Ship transport module implementing the Shipping Model Specification.

Ships move through explicit states:
- IDLE: Waiting at origin for route assignment
- LOADING: Loading cargo from origin stores
- IN_TRANSIT: Sailing between ports
- WAITING_FOR_BERTH: Queued at destination berth
- UNLOADING: Unloading cargo to destination stores
- WAITING_FOR_SPACE: Waiting for destination store capacity
- ERROR: Unable to proceed (logged, remains idle)

Planning logic:
- Evaluate candidate routes based on utilization (hold fill %) and urgency
- At least 60% holds must be filled before departure
- Travel time = distance / speed + pilot_in + pilot_out hours

Execution:
- FIFO berth queues at each port
- Load/unload times based on store rates
- Ships complete full route sequence then return to origin pool

This module is refactored into smaller sub-modules:
- sim_run_core_move_ship_types: ShipState enum and ShipContext dataclass
- sim_run_core_move_ship_utils: Utility functions for berths, distances, rates
- sim_run_core_move_ship_scoring: Route scoring logic
- sim_run_core_move_ship_handlers: State machine handlers
"""
from __future__ import annotations
from typing import Callable, Dict, Tuple, Optional
import simpy

from sim_run_types import TransportRoute

# Re-export public API from sub-modules for backward compatibility
from sim_run_core_move_ship_types import ShipState, ShipContext
from sim_run_core_move_ship_utils import (
    get_berth as _get_berth,
    get_pilot_hours as _get_pilot_hours,
    get_nm_distance as _get_nm_distance,
    get_store_rates as _get_store_rates,
    get_start_location as _get_start_location,
    get_route_id_from_itinerary as _get_route_id_from_itinerary,
)
from sim_run_core_move_ship_scoring import calculate_route_score as _calculate_route_score
from sim_run_core_move_ship_handlers import (
    handle_idle_state,
    handle_loading_state,
    handle_in_transit_state,
    handle_waiting_for_berth_state,
    handle_unloading_state,
    handle_error_state,
)


def transporter(env: simpy.Environment, route: TransportRoute,
                stores: Dict[str, simpy.Container],
                port_berths: Dict[str, simpy.Resource],
                log_func: Callable,
                store_rates: Dict[str, Tuple[float, float]],
                require_full: bool = True,
                demand_rates: Optional[Dict[str, float]] = None,
                vessel_id: int = 1,
                sole_supplier_stores: Optional[set] = None,
                production_rates: Optional[Dict[str, float]] = None,
                store_capacity_map: Optional[Dict[str, float]] = None,
                sim=None,
                t_state: Optional[dict] = None):
    """
    Main ship vessel process. This is called once per vessel by the simulation core.
    Follows the state machine: IDLE -> LOADING (-> WAITING_FOR_PRODUCT -> LOADING) -> IN_TRANSIT -> WAITING_FOR_BERTH -> UNLOADING (-> WAITING_FOR_SPACE -> UNLOADING) -> IDLE

    sole_supplier_stores: Set of store keys that only have ONE route serving them.
    production_rates: Dict mapping store_key -> production rate (tons/hour) INTO that store.
    store_capacities: Dict mapping store_key -> capacity.
    """
    if not getattr(route, 'itineraries', None):
        while True:
            wait_h = 24.0
            if sim and hasattr(sim, 'settings'):
                wait_h = float(sim.settings.get('ship_idle_wait_h', 24.0))
            yield env.timeout(wait_h)

    # Extract route configuration
    speed_knots = float(getattr(route, 'speed_knots', 10.0) or 10.0)
    n_holds = int(getattr(route, 'holds_per_vessel', 0) or 0)
    payload_per_hold = float(getattr(route, 'payload_per_hold_t', 0.0) or 0.0)

    if n_holds <= 0 or payload_per_hold <= 0:
        payload_t = float(getattr(route, 'payload_t', 25000.0) or 25000.0)
        n_holds = 1
        payload_per_hold = payload_t

    route_group = getattr(route, 'route_group', 'Ship')
    berth_info = getattr(route, 'berth_info', {}) or {}
    nm_distances = getattr(route, 'nm_distance', {}) or {}
    default_load_rate = float(getattr(route, 'load_rate_tph', 500.0) or 500.0)
    default_unload_rate = float(getattr(route, 'unload_rate_tph', 500.0) or 500.0)

    max_wait_product_h = float(getattr(route, 'max_wait_product_h', 0.0) or 0.0)
    if max_wait_product_h <= 0:
        max_wait_product_h = float(sim.settings.get('ship_max_wait_product_h', 24.0)) if sim else 24.0

    min_utilization = 0.60 if require_full else 0.0
    origin_location = route.origin_location

    # Initialize cargo tracking
    cargo = {}
    if t_state is not None:
        t_state['cargo'] = cargo

    # Create context object with all state
    ctx = ShipContext(
        env=env,
        stores=stores,
        port_berths=port_berths,
        log_func=log_func,
        store_rates=store_rates,
        speed_knots=speed_knots,
        n_holds=n_holds,
        payload_per_hold=payload_per_hold,
        route_group=route_group,
        berth_info=berth_info,
        nm_distances=nm_distances,
        default_load_rate=default_load_rate,
        default_unload_rate=default_unload_rate,
        max_wait_product_h=max_wait_product_h,
        min_utilization=min_utilization,
        vessel_id=vessel_id,
        origin_location=origin_location,
        current_location=origin_location,
        state=ShipState.IDLE,
        cargo=cargo,
        demand_rates=demand_rates or {},
        sole_supplier_stores=sole_supplier_stores,
        production_rates=production_rates,
        store_capacity_map=store_capacity_map,
        sim=sim,
        t_state=t_state,
        itineraries=route.itineraries,
    )

    # Log initial state
    ctx.log_state_change(ctx.state, origin_location)

    # Main state machine loop
    while True:
        if ctx.state == ShipState.IDLE:
            result = yield from handle_idle_state(ctx)
            if result == 'continue':
                continue

        elif ctx.state == ShipState.LOADING:
            result = yield from handle_loading_state(ctx)
            if result == 'continue':
                continue

        elif ctx.state == ShipState.IN_TRANSIT:
            result = yield from handle_in_transit_state(ctx)
            if result == 'continue':
                continue

        elif ctx.state == ShipState.WAITING_FOR_BERTH:
            result = yield from handle_waiting_for_berth_state(ctx)
            if result == 'continue':
                continue

        elif ctx.state == ShipState.UNLOADING:
            result = yield from handle_unloading_state(ctx)
            if result == 'continue':
                continue

        elif ctx.state == ShipState.ERROR:
            result = yield from handle_error_state(ctx)
            if result == 'continue':
                continue

