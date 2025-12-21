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
"""
from __future__ import annotations
from enum import Enum
from typing import Callable, Dict, List, Tuple, Optional, Any
import math
import random
from collections import defaultdict
import simpy

from sim_run_types import TransportRoute


class ShipState(Enum):
    IDLE = "IDLE"
    LOADING = "LOADING"
    WAITING_FOR_PRODUCT = "WAITING_FOR_PRODUCT"
    IN_TRANSIT = "IN_TRANSIT"
    WAITING_FOR_BERTH = "WAITING_FOR_BERTH"
    UNLOADING = "UNLOADING"
    WAITING_FOR_SPACE = "WAITING_FOR_SPACE"
    ERROR = "ERROR"


def _get_berth(env: simpy.Environment, port_berths: Dict[str, simpy.Resource],
               berth_info: Dict[str, dict], location: str) -> simpy.Resource:
    """Get or create FIFO berth queue for a location."""
    if location not in port_berths:
        info = berth_info.get(location, {})
        cap = max(1, int(info.get('berths', 1) or 1))
        port_berths[location] = simpy.Resource(env, capacity=cap)
    return port_berths[location]


def _get_pilot_hours(berth_info: Dict[str, dict], location: str, phase: str) -> float:
    """Get pilot hours for a location (in or out)."""
    info = berth_info.get(location, {})
    if phase == 'in':
        return float(info.get('pilot_in_h', 0.0) or 0.0)
    else:
        return float(info.get('pilot_out_h', 0.0) or 0.0)


def _get_nm_distance(nm_distances: Dict[Tuple[str, str], float], loc_a: str, loc_b: str) -> float:
    """Get nautical mile distance between two locations."""
    return float(nm_distances.get((loc_a, loc_b), 0.0) or nm_distances.get((loc_b, loc_a), 0.0) or 0.0)


def _get_store_rates(store_rates: Dict[str, Tuple[float, float]], store_key: str,
                     default_load: float, default_unload: float) -> Tuple[float, float]:
    """Get load/unload rates for a store. Returns (load_rate, unload_rate)."""
    if store_key in store_rates:
        rates = store_rates[store_key]
        return (float(rates[0] or default_load), float(rates[1] or default_unload))
    return (default_load, default_unload)


def _get_start_location(itinerary: List[Dict]) -> Optional[str]:
    """Extract start location from itinerary."""
    for step in itinerary:
        if step.get('kind') == 'start':
            return step.get('location')
    return None


def _calculate_route_score(itinerary: List[Dict], stores: Dict[str, simpy.Container],
                           payload_per_hold: float, n_holds: int, demand_rates: Dict[str, float],
                           nm_distances: Dict, speed_knots: float, berth_info: Dict,
                           sole_supplier_stores: Optional[set] = None,
                           production_rates: Optional[Dict[str, float]] = None,
                           store_capacities: Optional[Dict[str, float]] = None,
                           pending_deliveries: Optional[Dict[Tuple[int, str], float]] = None,
                           current_vessel_id: int = -1,
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

    # Aggregate pending deliveries from OTHER vessels
    other_pending = defaultdict(float)
    if pending_deliveries:
        for (vid, sk), qty in pending_deliveries.items():
            if vid != current_vessel_id:
                other_pending[sk] += qty

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
                nm = _get_nm_distance(nm_distances, from_loc, to_loc)
                pilot_out = _get_pilot_hours(berth_info, from_loc, 'out')
                pilot_in = _get_pilot_hours(berth_info, to_loc, 'in')
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
            projected_level = level + (prod_rate - rate) * current_time_offset + other_pending.get(sk, 0.0)
            projected_level = max(0.0, projected_level)

            # Use a safety buffer: only target 85% of capacity to account for variance
            effective_capacity = capacity * 0.85
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
                            sole_supplier_bonus = bonus

            # Penalize if projected to be full or nearly full
            if capacity < float('inf'):
                projected_fill = projected_level / capacity
                if projected_fill > 0.80:
                    # Heavy penalty if already full or nearly full
                    urgency -= 300 * (projected_fill - 0.80) / 0.20

    # If we have existing cargo and this route CANNOT unload it, penalize heavily
    if total_cargo_already_onboard > 1e-6 and not can_unload_existing_cargo:
        urgency -= 1000.0  # Force picking a route that unloads our cargo

    # Suggest load: limited by projected headspace and total available at source
    suggested_load = total_available
    if has_unload:
        suggested_load = min(suggested_load, total_projected_headspace)

    suggested_load = min(suggested_load, total_capacity)
    # Quantize to hold sizes
    suggested_load = math.floor(suggested_load / payload_per_hold) * payload_per_hold

    # What we actually NEED to load is the difference
    suggested_to_load = max(0.0, suggested_load - total_cargo_already_onboard)

    # Final score components
    # Re-calculate utilization based on suggested_load (which includes already onboard)
    utilization = min(1.0, suggested_load / max(total_capacity, 1.0))
    urgency = max(-2000.0, urgency)  # Allow negative urgency to discourage bad routes

    score = (utilization * 50) + (urgency * 30) + sole_supplier_bonus + overflow_bonus - (travel_time * 0.1)

    # Apply headspace factor: if we can't even fill one hold because of destination headspace, 
    # the route is essentially worthless for moving product.
    headspace_factor = 1.0
    if has_unload and total_capacity > 0:
        headspace_factor = min(1.0, total_projected_headspace / total_capacity)
        if headspace_factor < 0.5:
            headspace_factor *= 0.2

    if score > 0:
        score *= headspace_factor

    return (score, utilization, urgency, overflow_bonus, suggested_to_load)


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
            yield env.timeout(24)

    speed_knots = float(getattr(route, 'speed_knots', 10.0) or 10.0)
    n_holds = int(getattr(route, 'holds_per_vessel', 0) or 0)
    payload_per_hold = float(getattr(route, 'payload_per_hold_t', 0.0) or 0.0)

    if n_holds <= 0 or payload_per_hold <= 0:
        # Fallback to general payload if hold info is missing
        payload_t = float(getattr(route, 'payload_t', 25000.0) or 25000.0)
        n_holds = 1
        payload_per_hold = payload_t

    route_group = getattr(route, 'route_group', 'Ship')
    berth_info = getattr(route, 'berth_info', {}) or {}
    nm_distances = getattr(route, 'nm_distance', {}) or {}
    default_load_rate = float(getattr(route, 'load_rate_tph', 500.0) or 500.0)
    default_unload_rate = float(getattr(route, 'unload_rate_tph', 500.0) or 500.0)

    # Get max wait for product from route or settings
    max_wait_product_h = float(getattr(route, 'max_wait_product_h', 0.0) or 0.0)
    if max_wait_product_h <= 0:
        # Fallback to general settings or hard default
        max_wait_product_h = float(sim.settings.get('ship_max_wait_product_h', 24.0)) if sim else 24.0

    min_utilization = 0.60 if require_full else 0.0

    origin_location = route.origin_location
    current_location = origin_location
    state = ShipState.IDLE
    cargo = {}
    if t_state is not None:
        t_state['cargo'] = cargo
    chosen_itinerary = None
    itinerary_idx = 0
    demand_rates_map = demand_rates or {}
    active_berth = None
    active_berth_req = None
    current_route_id = None  # Specific route ID (e.g., 1.1, 1.2)
    total_waited_at_location = 0.0
    last_prob_wait_location = None  # Tracks the last location where we performed a probabilistic wait check

    def _get_route_id_from_itinerary(itinerary: List[Dict]) -> Optional[str]:
        """Extract specific route_id from itinerary start step."""
        for step in itinerary:
            if step.get('kind') == 'start' and 'route_id' in step:
                return str(step['route_id'])
        return None

    def log_state_change(new_state: ShipState, location: str = None):
        log_func(
            process="ShipState",
            event="StateChange",
            location=location or current_location,
            equipment="Ship",
            product=None,
            qty=None,
            from_store=None,
            from_level=None,
            to_store=None,
            to_level=None,
            route_id=current_route_id or route_group,
            vessel_id=vessel_id,
            ship_state=new_state.value,
            override_time_h=env.now
        )

    log_state_change(state, origin_location)

    while True:
        if state == ShipState.IDLE:
            # Ensure t_state is synced with current cargo
            if t_state is not None:
                t_state['cargo'] = cargo

            candidate_its = [it for it in route.itineraries
                             if _get_start_location(it) == current_location]

            if not candidate_its:
                current_location = origin_location
                yield env.timeout(1)
                continue

            best_it = None
            best_score = -float('inf')
            best_suggested_load = 0.0

            prod_rates = production_rates or {}
            capacities = store_capacity_map or {}

            for it in candidate_its:
                # [Issue Fix]: Don't select a route if another vessel is already delivering to its target store
                is_already_served = False
                if sim and hasattr(sim, 'pending_deliveries'):
                    for step in it:
                        if step.get('kind') == 'unload':
                            sk = step.get('store_key')
                            if sk:
                                # Check if ANY other vessel has a pending delivery to this store
                                if any(qty > 1e-6 for (vid, s_key), qty in sim.pending_deliveries.items() 
                                       if s_key == sk and vid != vessel_id):
                                    is_already_served = True
                                    break
                
                if is_already_served:
                    continue

                score, util, _, overflow_bonus, suggested_to_load = _calculate_route_score(
                    it, stores, payload_per_hold, n_holds,
                    demand_rates_map, nm_distances, speed_knots, berth_info,
                    sole_supplier_stores, prod_rates, capacities,
                    getattr(sim, 'pending_deliveries', None),
                    vessel_id,
                    cargo
                )

                required_util = min_utilization

                if overflow_bonus > 60:
                    required_util = 0.20

                if sole_supplier_stores:
                    for step in it:
                        if step.get('kind') == 'unload':
                            sk = step.get('store_key')
                            if sk and sk in sole_supplier_stores and sk in stores:
                                level = float(stores[sk].level)
                                rate = demand_rates_map.get(sk, 0.0)
                                if rate > 0:
                                    days_of_stock = level / (rate * 24)
                                    if days_of_stock < 60:
                                        required_util = 0.20
                                        break

                if util >= required_util and score > best_score:
                    best_score = score
                    best_it = it
                    best_suggested_load = suggested_to_load

            if best_it is None:
                # If we have cargo but can't find a route, we MUST wait or find any route that unloads.
                # If we keep cargo, we just wait.
                reason = "No suitable itinerary"
                if cargo:
                    # Check if we are waiting to unload somewhere but can't find a path or have no space
                    # This is a bit complex for ships, but "No suitable itinerary" covers it.
                    pass

                log_func(
                    process="Move",
                    event=reason,
                    location=current_location,
                    equipment="Ship",
                    product=str(list(cargo.keys())) if cargo else None,
                    time=1.0,
                    qty=sum(cargo.values()) if cargo else None,
                    from_store=None,
                    to_store=None,
                    route_id=current_route_id or route_group,
                    vessel_id=vessel_id,
                    ship_state=state.value,
                    override_time_h=env.now
                )
                if sim:
                    yield sim.wait_for_step(7)
                else:
                    yield env.timeout(1)
                continue

            chosen_itinerary = best_it
            current_route_id = _get_route_id_from_itinerary(chosen_itinerary)
            
            # [Issue Fix]: Clear existing cargo if it doesn't match the new itinerary's products
            # This prevents vessels from carrying over phantom cargo that inflates utilization
            # or causes discrepancies in unload quantities.
            if cargo:
                new_products = {step.get('product') for step in chosen_itinerary if step.get('kind') in ('load', 'unload')}
                mismatch = any(p not in new_products for p in cargo.keys())
                if mismatch:
                    log_func(
                        process="Move",
                        event="Clear Cargo",
                        location=current_location,
                        equipment="Ship",
                        product=str(list(cargo.keys())),
                        qty=sum(cargo.values()),
                        time=0.0,
                        from_store=None,
                        to_store=None,
                        route_id=current_route_id or route_group,
                        vessel_id=vessel_id,
                        ship_state=state.value,
                        override_time_h=env.now
                    )
                    cargo = {}
                    if t_state is not None:
                        t_state['cargo'] = cargo

            if t_state is not None:
                t_state['route_id'] = current_route_id or route_group
            state = ShipState.LOADING
            log_state_change(state)

            # Register pending deliveries
            if sim and hasattr(sim, 'pending_deliveries'):
                # Track how much of each product we WILL have after all planned loads
                projected_cargo = cargo.copy()
                # Simplified: assume all planned load is the 'primary' product if multi-product not specified
                # In reality, best_suggested_load is a total. 
                # Let's see if we can identify which product it belongs to.
                
                # Find the first load step's product as the primary one for the suggested load
                primary_load_product = None
                for step in chosen_itinerary:
                    if step.get('kind') == 'load':
                        primary_load_product = step.get('product')
                        break
                
                if primary_load_product:
                    projected_cargo[primary_load_product] = projected_cargo.get(primary_load_product, 0.0) + best_suggested_load
                
                for step in chosen_itinerary:
                    if step.get('kind') == 'unload':
                        sk = step.get('store_key')
                        prod = step.get('product')
                        if sk and prod in projected_cargo:
                            # We only add the amount of THIS product that we are carrying
                            qty_to_drop = projected_cargo[prod]
                            sim.pending_deliveries[(vessel_id, sk)] += qty_to_drop

            # Track planned load to respect it during loading phase
            planned_load_remaining = best_suggested_load
            itinerary_idx = 0
            total_waited_at_location = 0.0
            last_prob_wait_location = None  # Reset on new itinerary

            for i, step in enumerate(chosen_itinerary):
                if step.get('kind') == 'start':
                    itinerary_idx = i + 1
                    break

        elif state == ShipState.LOADING:
            if itinerary_idx >= len(chosen_itinerary):
                if active_berth is not None:
                    active_berth.release(active_berth_req)
                    active_berth = None
                    active_berth_req = None
                state = ShipState.IN_TRANSIT
                log_state_change(state)
                continue

            # Wait for Step 3: Reduce inventory by the "Ship Load" Qty
            # (Step 3 is now handled inside the berth request to be closer to the action)

            step = chosen_itinerary[itinerary_idx]
            kind = step.get('kind')

            if kind == 'load':
                store_key = step.get('store_key')
                product = step.get('product')
                location = step.get('location', current_location)

                if store_key and store_key in stores:
                    # Request berth if not already held
                    if active_berth is None:
                        # 1. Probabilistic Berth Wait (Check ONLY on arrival at new location)
                        if last_prob_wait_location != location:
                            loc_info = berth_info.get(location, {})
                            p_occ = loc_info.get('p_occupied', 0.0)

                            while random.random() < p_occ:
                                log_func(
                                    process="Move",
                                    event="Wait for Berth",
                                    location=location,
                                    equipment="Ship",
                                    product=product,
                                    time=1.0,
                                    from_store=None,
                                    to_store=None,
                                    route_id=current_route_id or route_group,
                                    vessel_id=vessel_id,
                                    ship_state=ShipState.WAITING_FOR_BERTH.value,
                                    override_time_h=env.now
                                )
                                if sim:
                                    yield sim.wait_for_step(7)
                                else:
                                    yield env.timeout(1.0)

                            last_prob_wait_location = location

                        # 2. Physical Berth Request (FIFO Resource)
                        berth = _get_berth(env, port_berths, berth_info, location)
                        req = berth.request()
                        # Log waiting for berth if it's not immediate
                        if not req.triggered:
                            while not req.triggered:
                                log_func(
                                    process="Move",
                                    event="Wait for Berth",
                                    location=location,
                                    equipment="Ship",
                                    product=product,
                                    time=1.0,
                                    from_store=None,
                                    to_store=None,
                                    route_id=current_route_id or route_group,
                                    vessel_id=vessel_id,
                                    ship_state=ShipState.WAITING_FOR_BERTH.value,
                                    override_time_h=env.now
                                )
                                if sim:
                                    yield sim.env.any_of([req, sim.wait_for_step(7)])
                                else:
                                    yield sim.env.any_of([req, env.timeout(1)])
                        else:
                            # Immediate berth acquisition - still log it for transparency
                            log_func(
                                process="Move",
                                event="Wait for Berth",
                                location=location,
                                equipment="Ship",
                                product=product,
                                time=0.0,
                                from_store=None,
                                to_store=None,
                                route_id=current_route_id or route_group,
                                vessel_id=vessel_id,
                                ship_state=ShipState.WAITING_FOR_BERTH.value,
                                override_time_h=env.now
                            )

                        active_berth = berth
                        active_berth_req = req
                        # Don't log state change here as it's already LOADING or UNLOADING

                    cont = stores[store_key]
                    load_rate, _ = _get_store_rates(store_rates, store_key, default_load_rate, default_unload_rate)

                    already_loaded = sum(cargo.values())
                    remaining_cap = max(0, (n_holds * payload_per_hold) - already_loaded)
                    # Also limit by planned_load_remaining
                    remaining_cap = min(remaining_cap, planned_load_remaining)
                    holds_remaining = int(remaining_cap // payload_per_hold)

                    total_loaded_this_stop = 0.0
                    time_per_hold = payload_per_hold / max(load_rate, 1.0)

                    for hold_num in range(holds_remaining):
                        # Wait for Step 3: Reduce inventory by the "Ship Load" Qty
                        if sim:
                            yield sim.wait_for_step(3)

                        if cont.level >= payload_per_hold:
                            # IMPORTANT: GET THE INVENTORY FIRST
                            yield cont.get(payload_per_hold)
                            
                            log_func(
                                process="Move",
                                event="Load",
                                location=location,
                                equipment="Ship",
                                product=product,
                                qty=payload_per_hold,
                                time=round(time_per_hold, 2),
                                qty_out=payload_per_hold,
                                from_store=store_key,
                                from_level=float(cont.level),
                                from_fill_pct=float(cont.level) / cont.capacity if cont.capacity > 0 else 0.0,
                                to_store=None,
                                to_level=None,
                                route_id=current_route_id or route_group,
                                vessel_id=vessel_id,
                                ship_state=state.value,
                                override_time_h=env.now
                            )

                            yield env.timeout(time_per_hold)
                            total_loaded_this_stop += payload_per_hold
                            planned_load_remaining -= payload_per_hold
                        else:
                            waited_this_step = 0.0
                            while cont.level < payload_per_hold and total_waited_at_location < max_wait_product_h:
                                log_func(
                                    process="Move",
                                    event="Waiting for Product",
                                    location=location,
                                    equipment="Ship",
                                    product=product,
                                    time=1.0,
                                    from_store=store_key,
                                    from_level=float(cont.level),
                                    from_fill_pct=float(cont.level) / cont.capacity if cont.capacity > 0 else 0.0,
                                    to_store=None,
                                    route_id=current_route_id or route_group,
                                    vessel_id=vessel_id,
                                    ship_state=ShipState.WAITING_FOR_PRODUCT.value,
                                    override_time_h=env.now
                                )
                                if sim:
                                    yield sim.wait_for_step(7)
                                else:
                                    yield env.timeout(1.0)
                                waited_this_step += 1.0
                                total_waited_at_location += 1.0

                            if cont.level >= payload_per_hold:
                                # Wait for Step 3: Reduce inventory by the "Ship Load" Qty
                                if sim:
                                    yield sim.wait_for_step(3)

                                # IMPORTANT: GET THE INVENTORY FIRST
                                yield cont.get(payload_per_hold)

                                log_func(
                                    process="Move",
                                    event="Load",
                                    location=location,
                                    equipment="Ship",
                                    product=product,
                                    qty=payload_per_hold,
                                    time=round(time_per_hold, 2),
                                    qty_out=payload_per_hold,
                                    from_store=store_key,
                                    from_level=float(cont.level),
                                    from_fill_pct=float(cont.level) / cont.capacity if cont.capacity > 0 else 0.0,
                                    to_store=None,
                                    to_level=None,
                                    route_id=current_route_id or route_group,
                                    vessel_id=vessel_id,
                                    ship_state=state.value,
                                    override_time_h=env.now
                                )

                                yield env.timeout(time_per_hold)
                                total_loaded_this_stop += payload_per_hold
                                planned_load_remaining -= payload_per_hold
                            else:
                                break

                    if total_loaded_this_stop > 0:
                        cargo[product] = cargo.get(product, 0.0) + total_loaded_this_stop

                itinerary_idx += 1

            elif kind == 'sail':
                total_loaded = sum(cargo.values())
                utilization = total_loaded / max(n_holds * payload_per_hold, 1.0)

                # Logic: If we are at the end of loading at this location (reached 'sail')
                # and we don't have enough cargo to meet min_utilization, we have a choice:
                # 1. Stay and wait more (already done by the load loop's wait)
                # 2. Go anyway if we have SOME cargo (if allowed)
                # 3. Abort/Idle if we have NO cargo or insufficient cargo.

                if utilization >= min_utilization:
                    if active_berth is not None:
                        # Release berth only when we are actually proceeding with transit
                        active_berth.release(active_berth_req)
                        active_berth = None
                        active_berth_req = None

                    # If we loaded less than planned, adjust pending deliveries
                    if planned_load_remaining > 1e-6 and sim and hasattr(sim, 'pending_deliveries'):
                        # Find which product we failed to load fully
                        primary_load_product = None
                        for step in chosen_itinerary:
                            if step.get('kind') == 'load':
                                primary_load_product = step.get('product')
                                break
                        
                        if primary_load_product:
                            for step in chosen_itinerary:
                                if step.get('kind') == 'unload' and step.get('product') == primary_load_product:
                                    sk = step.get('store_key')
                                    if (vessel_id, sk) in sim.pending_deliveries:
                                        sim.pending_deliveries[(vessel_id, sk)] = max(0.0, sim.pending_deliveries[(vessel_id, sk)] - planned_load_remaining)

                    state = ShipState.IN_TRANSIT
                    log_state_change(state)
                elif total_loaded > 1e-6 and total_waited_at_location < max_wait_product_h:
                    # We have SOME cargo but didn't hit 60% and we haven't timed out yet.
                    # Find the first 'load' step at the current location to retry
                    found_retry_step = False
                    for i in range(itinerary_idx - 1, -1, -1):
                        s = chosen_itinerary[i]
                        if s.get('kind') == 'load' and s.get('location', current_location) == current_location:
                            itinerary_idx = i
                            found_retry_step = True
                            break
                        elif s.get('kind') in ('sail', 'start'):
                            break

                    log_func(
                        process="Move",
                        event="Utilization Low - Retrying Load",
                        location=current_location,
                        equipment="Ship",
                        product=None,
                        time=1.0,
                        qty=0,
                        from_store=None,
                        to_store=None,
                        route_id=current_route_id or route_group,
                        vessel_id=vessel_id,
                        ship_state=state.value,
                        override_time_h=env.now
                    )
                    
                    # Reset planned load to allow filling remaining capacity during retry
                    total_loaded = sum(cargo.values())
                    planned_load_remaining = max(0.0, (n_holds * payload_per_hold) - total_loaded)

                    if sim:
                        yield sim.wait_for_step(7)
                    else:
                        yield env.timeout(1)
                    total_waited_at_location += 1.0
                else:
                    # EITHER utilization hit, OR we have SOME cargo and timed out, OR we have NO cargo.
                    if total_loaded > 1e-6:
                        # We have SOME cargo. If we are here, it means either we hit utilization (handled above)
                        # OR we timed out waiting for more.
                        # If we timed out, we should probably just go even if below min_utilization,
                        # UNLESS the user strictly wants 60%. 
                        # Given the prompt "i set the max wait time for product to 6 hours", 
                        # it implies they want the ship to DO something after 6 hours.

                        log_func(
                            process="Move",
                            event="Wait Limit Reached - Proceeding with Partial Load",
                            location=current_location,
                            equipment="Ship",
                            product=None,
                            time=0.0,
                            qty=0,
                            from_store=None,
                            to_store=None,
                            route_id=current_route_id or route_group,
                            vessel_id=vessel_id,
                            ship_state=state.value,
                            override_time_h=env.now
                        )
                        # If we loaded less than planned, adjust pending deliveries
                        if planned_load_remaining > 1e-6 and sim and hasattr(sim, 'pending_deliveries'):
                            # Find which product we failed to load fully
                            primary_load_product = None
                            for step in chosen_itinerary:
                                if step.get('kind') == 'load':
                                    primary_load_product = step.get('product')
                                    break
                            
                            if primary_load_product:
                                for step in chosen_itinerary:
                                    if step.get('kind') == 'unload' and step.get('product') == primary_load_product:
                                        sk = step.get('store_key')
                                        if (vessel_id, sk) in sim.pending_deliveries:
                                            sim.pending_deliveries[(vessel_id, sk)] = max(0.0, sim.pending_deliveries[(vessel_id, sk)] - planned_load_remaining)

                        if active_berth is not None:
                            active_berth.release(active_berth_req)
                            active_berth = None
                            active_berth_req = None

                        state = ShipState.IN_TRANSIT
                        log_state_change(state)
                    else:
                        # NOTHING loaded and we either finished all load steps or timed out.
                        # Let's reset to IDLE so we can pick a better route.
                        # Check for lost cargo before resetting
                        if active_berth is not None:
                            active_berth.release(active_berth_req)
                            active_berth = None
                            active_berth_req = None

                        total_cargo = sum(cargo.values())
                        if total_cargo > 1e-6:
                            log_func(
                                process="Move",
                                event="Cargo Lost at Sea",
                                location=current_location,
                                equipment="Ship",
                                product=str(list(cargo.keys())),
                                qty=total_cargo,
                                time=0.0,
                                from_store=None,
                                to_store=None,
                                route_id=current_route_id or route_group,
                                vessel_id=vessel_id,
                                ship_state=state.value,
                                override_time_h=env.now
                            )

                        # Unregister pending deliveries before clearing
                        if sim and hasattr(sim, 'pending_deliveries'):
                            # Clear all pending deliveries for this vessel
                            keys_to_del = [k for k in sim.pending_deliveries.keys() if k[0] == vessel_id]
                            for k in keys_to_del:
                                del sim.pending_deliveries[k]

                        cargo = {}
                        if t_state is not None:
                            t_state['cargo'] = cargo
                            t_state['route_id'] = route_group
                        chosen_itinerary = None
                        current_route_id = None
                        state = ShipState.IDLE
                        log_state_change(state)

                        if sim:
                            yield sim.wait_for_step(7)
                        else:
                            yield env.timeout(1)
                        continue
            else:
                itinerary_idx += 1

        elif state == ShipState.IN_TRANSIT:
            if itinerary_idx >= len(chosen_itinerary):
                if active_berth is not None:
                    active_berth.release(active_berth_req)
                    active_berth = None
                    active_berth_req = None

                current_location = origin_location
                state = ShipState.IDLE
                current_route_id = None
                if t_state is not None:
                    t_state['route_id'] = route_group
                log_state_change(state)
                # DO NOT reset cargo here - keep it for next route
                # Unregister pending deliveries
                if sim and hasattr(sim, 'pending_deliveries'):
                    # Clear all pending deliveries for this vessel
                    keys_to_del = [k for k in sim.pending_deliveries.keys() if k[0] == vessel_id]
                    for k in keys_to_del:
                        del sim.pending_deliveries[k]

                if sim:
                    yield sim.wait_for_step(7)
                else:
                    yield env.timeout(1)
                continue

            step = chosen_itinerary[itinerary_idx]
            kind = step.get('kind')

            if kind == 'sail':
                from_loc = step.get('from', current_location)
                to_loc = step.get('to', step.get('location'))

                if to_loc:
                    if from_loc == to_loc:
                        # Skip transit if already at location
                        current_location = to_loc
                        itinerary_idx += 1
                        continue

                    # Check if there's any remaining work in the itinerary
                    # If we have no cargo and no more loads, we should probably just go home
                    has_future_work = False
                    total_cargo = sum(cargo.values())

                    for i in range(itinerary_idx, len(chosen_itinerary)):
                        future_step = chosen_itinerary[i]
                        f_kind = future_step.get('kind')
                        if f_kind == 'load':
                            has_future_work = True
                            break
                        if f_kind == 'unload':
                            f_prod = future_step.get('product')
                            if f_prod in cargo and cargo[f_prod] > 1e-6:
                                has_future_work = True
                                break

                    if not has_future_work and to_loc != origin_location:
                        # No more loads or relevant unloads. 
                        # Find if there's a return leg to origin in the itinerary
                        found_return = False
                        for i in range(itinerary_idx + 1, len(chosen_itinerary)):
                            if chosen_itinerary[i].get('kind') == 'sail' and \
                                    chosen_itinerary[i].get('location') == origin_location:
                                itinerary_idx = i
                                found_return = True
                                break

                        if found_return:
                            continue  # Jump to return leg
                        else:
                            # No return leg found, just finish
                            itinerary_idx = len(chosen_itinerary)
                            continue

                    nm = _get_nm_distance(nm_distances, from_loc, to_loc)
                    travel_hours = nm / max(speed_knots, 1.0)

                    pilot_out = _get_pilot_hours(berth_info, from_loc, 'out')
                    pilot_in = _get_pilot_hours(berth_info, to_loc, 'in')
                    total_time = travel_hours + pilot_out + pilot_in

                    if total_time > 0:
                        log_func(
                            process="Move",
                            event="Transit",
                            location=from_loc,
                            equipment="Ship",
                            product=None,
                            time=round(total_time, 2),
                            from_store=None,
                            to_store=None,
                            route_id=current_route_id or route_group,
                            vessel_id=vessel_id,
                            ship_state=state.value,
                            override_time_h=env.now
                        )
                        yield env.timeout(total_time)

                    current_location = to_loc

                itinerary_idx += 1

            elif kind == 'unload':
                # Transition to WAITING_FOR_BERTH but keep the berth if it's the same location
                unload_loc = step.get('location', current_location)
                if active_berth is not None and unload_loc == current_location:
                    state = ShipState.UNLOADING
                else:
                    # If we are not at the same location, check if we need to sail first
                    if unload_loc != current_location:
                        # Auto-sail to the destination if no explicit sail step exists
                        # or if we are skipping it.
                        nm = _get_nm_distance(nm_distances, current_location, unload_loc)
                        if nm > 0:
                            travel_hours = nm / max(speed_knots, 1.0)
                            pilot_out = _get_pilot_hours(berth_info, current_location, 'out')
                            pilot_in = _get_pilot_hours(berth_info, unload_loc, 'in')
                            total_time = travel_hours + pilot_out + pilot_in
                            
                            log_func(
                                process="Move",
                                event="Transit",
                                location=current_location,
                                equipment="Ship",
                                product=None,
                                time=round(total_time, 2),
                                from_store=None,
                                to_store=None,
                                route_id=current_route_id or route_group,
                                vessel_id=vessel_id,
                                ship_state=state.value,
                                override_time_h=env.now
                            )
                            yield env.timeout(total_time)
                            current_location = unload_loc
                    
                    state = ShipState.WAITING_FOR_BERTH
                log_state_change(state, current_location)
                continue

            elif kind == 'load':
                # If we already have a berth at this location, go straight to LOADING
                load_loc = step.get('location', current_location)
                if active_berth is not None and load_loc == current_location:
                    state = ShipState.LOADING
                else:
                    # If we are not at the same location, check if we need to sail first
                    if load_loc != current_location:
                        nm = _get_nm_distance(nm_distances, current_location, load_loc)
                        if nm > 0:
                            travel_hours = nm / max(speed_knots, 1.0)
                            pilot_out = _get_pilot_hours(berth_info, current_location, 'out')
                            pilot_in = _get_pilot_hours(berth_info, load_loc, 'in')
                            total_time = travel_hours + pilot_out + pilot_in
                            
                            log_func(
                                process="Move",
                                event="Transit",
                                location=current_location,
                                equipment="Ship",
                                product=None,
                                time=round(total_time, 2),
                                from_store=None,
                                to_store=None,
                                route_id=current_route_id or route_group,
                                vessel_id=vessel_id,
                                ship_state=state.value,
                                override_time_h=env.now
                            )
                            yield env.timeout(total_time)
                            current_location = load_loc

                    state = ShipState.LOADING
                log_state_change(state, current_location)
                continue

            else:
                itinerary_idx += 1

        elif state == ShipState.WAITING_FOR_BERTH:
            # We can now go straight to UNLOADING because UNLOADING handles its own berth acquisition
            state = ShipState.UNLOADING
            log_state_change(state)

        elif state == ShipState.UNLOADING:
            if itinerary_idx >= len(chosen_itinerary):
                if active_berth is not None:
                    active_berth.release(active_berth_req)
                    active_berth = None
                    active_berth_req = None
                current_location = origin_location
                state = ShipState.IDLE
                log_state_change(state)
                # DO NOT reset cargo here - keep it for next route
                # Unregister pending deliveries
                if sim and hasattr(sim, 'pending_deliveries'):
                    # Clear all pending deliveries for this vessel
                    keys_to_del = [k for k in sim.pending_deliveries.keys() if k[0] == vessel_id]
                    for k in keys_to_del:
                        del sim.pending_deliveries[k]

                if sim:
                    yield sim.wait_for_step(7)
                else:
                    yield env.timeout(1)
                continue

            # Wait for Step 7: Increase inventory by the "Ship Offload" Qty
            # (Step 7 is now handled inside the berth request to be closer to the action)

            step = chosen_itinerary[itinerary_idx]
            kind = step.get('kind')

            if kind == 'unload':
                store_key = step.get('store_key')
                product = step.get('product')
                unload_location = step.get('location', current_location)

                # Ensure we have a berth at the UNLOADING location
                if active_berth is None:
                    # 1. Probabilistic Berth Wait (Check ONLY on arrival at new location)
                    if last_prob_wait_location != unload_location:
                        loc_info = berth_info.get(unload_location, {})
                        p_occ = loc_info.get('p_occupied', 0.0)

                        while random.random() < p_occ:
                            log_func(
                                process="Move",
                                event="Wait for Berth",
                                location=unload_location,
                                equipment="Ship",
                                product=None,
                                time=1.0,
                                from_store=None,
                                to_store=None,
                                route_id=current_route_id or route_group,
                                vessel_id=vessel_id,
                                ship_state=ShipState.WAITING_FOR_BERTH.value,
                                override_time_h=env.now
                            )
                            if sim:
                                yield sim.wait_for_step(7)
                            else:
                                yield env.timeout(1.0)

                        last_prob_wait_location = unload_location

                    # 2. Physical Berth Request (FIFO Resource)
                    berth = _get_berth(env, port_berths, berth_info, unload_location)
                    req = berth.request()
                    if not req.triggered:
                        while not req.triggered:
                            log_func(
                                process="Move",
                                event="Wait for Berth",
                                location=unload_location,
                                equipment="Ship",
                                product=None,
                                time=1.0,
                                from_store=None,
                                to_store=None,
                                route_id=current_route_id or route_group,
                                vessel_id=vessel_id,
                                ship_state=ShipState.WAITING_FOR_BERTH.value,
                                override_time_h=env.now
                            )
                            if sim:
                                yield env.any_of([req, sim.wait_for_step(7)])
                            else:
                                yield env.any_of([req, env.timeout(1)])
                    else:
                        # Immediate berth acquisition - log it for transparency
                        log_func(
                            process="Move",
                            event="Wait for Berth",
                            location=unload_location,
                            equipment="Ship",
                            product=None,
                            time=0.0,
                            from_store=None,
                            to_store=None,
                            route_id=current_route_id or route_group,
                            vessel_id=vessel_id,
                            ship_state=ShipState.WAITING_FOR_BERTH.value,
                            override_time_h=env.now
                        )
                    active_berth = berth
                    active_berth_req = req

                if store_key and store_key in stores and product in cargo:
                    cont = stores[store_key]
                    _, unload_rate = _get_store_rates(store_rates, store_key, default_load_rate, default_unload_rate)

                    carried = float(cargo.get(product, 0.0))

                    # Wait until there's room for the FULL cargo (no partial unloads)
                    # Max wait: 48 hours to avoid indefinite blocking
                    wait_hours = 0
                    max_wait = 48

                    original_state = state
                    while wait_hours < max_wait:
                        room = float(cont.capacity - cont.level)
                        
                        # [Issue Fix] Better space projection:
                        # If inventory is being drawn from the store, we don't need to wait for ALL space 
                        # to be available right now if it will become available during unloading.
                        # However, we must ensure we ONLY unload full holds to satisfy the requirement.
                        
                        # Get consumption rate (demand) at this store
                        demand_rate = demand_rates_map.get(store_key, 0.0)
                        
                        # Required space for at least one hold
                        required_space = min(carried, payload_per_hold)
                        
                        # Calculate if/when space will be available for one hold
                        # room_at_t = room + demand_rate * t
                        # we want room_at_t >= required_space
                        # t = (required_space - room) / demand_rate
                        
                        can_proceed = False
                        if room >= required_space - 1e-6:
                            can_proceed = True
                        elif demand_rate > 1e-6:
                            # How much time would it take to get the required space?
                            time_to_space = (required_space - room) / demand_rate
                            # If it's less than, say, 1 hour, we can consider it "available enough" 
                            # to start the process, assuming the simulation's time-stepping handles it.
                            # But more robustly, we check if room + demand * 1h >= required_space
                            if room + demand_rate * 1.0 >= required_space - 1e-6:
                                can_proceed = True

                        if can_proceed:
                            break

                        if state != ShipState.WAITING_FOR_SPACE:
                            state = ShipState.WAITING_FOR_SPACE
                            log_state_change(state, unload_location)

                        log_func(
                            process="Move",
                            event="Waiting for Space",
                            location=unload_location,
                            equipment="Ship",
                            product=product,
                            time=1.0,
                            from_store=None,
                            to_store=store_key,
                            to_level=float(cont.level),
                            to_fill_pct=float(cont.level) / cont.capacity if cont.capacity > 0 else 0.0,
                            route_id=current_route_id or route_group,
                            vessel_id=vessel_id,
                            ship_state=state.value,
                            override_time_h=env.now
                        )
                        if sim:
                            yield sim.wait_for_step(7)
                        else:
                            yield env.timeout(1.0)
                        wait_hours += 1

                    if state != original_state:
                        state = original_state
                        log_state_change(state, unload_location)

                    # After waiting, unload what we can (strictly in full hold increments)
                    room = float(cont.capacity - cont.level)
                    
                    # Number of FULL holds that can fit in the current room
                    num_holds_to_unload = int(min(carried, room + 1e-6) // payload_per_hold)
                    qty_to_unload = float(num_holds_to_unload * payload_per_hold)

                    # Special case: if we have LESS than one hold left of THIS product on the ship, 
                    # and it's the LAST unload for this product, we MUST unload it to finish the trip.
                    if qty_to_unload < 1e-6 and carried < payload_per_hold and carried > 1e-6:
                        # Check if any future step in the itinerary unloads this product
                        has_future_unload = False
                        for i in range(itinerary_idx + 1, len(chosen_itinerary)):
                            f_step = chosen_itinerary[i]
                            if f_step.get('kind') == 'unload' and f_step.get('product') == product:
                                has_future_unload = True
                                break
                        
                        if not has_future_unload:
                            # It's the last stop for this product, and we have less than a full hold.
                            # Unload the remainder if it fits.
                            if room >= carried - 1e-6:
                                qty_to_unload = carried
                            elif wait_hours >= max_wait:
                                # We timed out and it still doesn't fit? Unload what fits anyway if it's the last stop.
                                qty_to_unload = min(carried, room)
                    
                    # If we have more than one hold, but can't fit even one, we skip and move on 
                    # (unless we already waited 48h and it's the last stop, handled above)

                    if qty_to_unload > 1e-6:
                        unload_time = qty_to_unload / max(unload_rate, 1.0)

                        # Wait for Step 7: Increase inventory by the "Ship Offload" Qty
                        if sim:
                            yield sim.wait_for_step(7)

                        # IMPORTANT: PUT THE INVENTORY FIRST
                        yield cont.put(qty_to_unload)

                        log_func(
                            process="Move",
                            event="Unload",
                            location=unload_location,
                            equipment="Ship",
                            product=product,
                            qty=qty_to_unload,
                            time=round(unload_time, 2),
                            qty_in=qty_to_unload,
                            from_store=None,
                            from_level=None,
                            to_store=store_key,
                            to_level=float(cont.level),
                            to_fill_pct=float(cont.level) / cont.capacity if cont.capacity > 0 else 0.0,
                            route_id=current_route_id or route_group,
                            vessel_id=vessel_id,
                            ship_state=state.value,
                            override_time_h=env.now
                        )

                        yield env.timeout(unload_time)

                        cargo[product] = carried - qty_to_unload

                        # Update pending deliveries
                        if sim and hasattr(sim, 'pending_deliveries'):
                            if (vessel_id, store_key) in sim.pending_deliveries:
                                sim.pending_deliveries[(vessel_id, store_key)] = max(0.0, sim.pending_deliveries[
                                    (vessel_id, store_key)] - qty_to_unload)

                itinerary_idx += 1
            elif kind == 'unload':
                # Transition to UNLOADING but keep the berth if it's the same location
                state = ShipState.UNLOADING
                log_state_change(state)
            elif kind == 'load':
                # Transition to LOADING but keep the berth if it's the same location
                state = ShipState.LOADING
                log_state_change(state)
            else:
                if active_berth is not None:
                    active_berth.release(active_berth_req)
                    active_berth = None
                    active_berth_req = None
                state = ShipState.IN_TRANSIT
                log_state_change(state)

        elif state == ShipState.ERROR:
            log_func(
                process="Move",
                event="Error",
                location=current_location,
                equipment="Ship",
                product=None,
                qty=0,
                from_store=None,
                from_level=None,
                to_store=None,
                to_level=None,
                route_id=current_route_id or route_group,
                vessel_id=vessel_id,
                ship_state=state.value,
                override_time_h=env.now
            )
            state = ShipState.IDLE
            cargo = {}
            if t_state is not None:
                t_state['cargo'] = cargo
            if sim:
                for _ in range(24):
                    yield sim.wait_for_step(7)
            else:
                yield env.timeout(24)