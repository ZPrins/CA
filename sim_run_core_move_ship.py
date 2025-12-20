# sim_run_core_move_ship.py
"""
Ship transport module implementing the Shipping Model Specification.

Ships move through explicit states:
- IDLE: Waiting at origin for route assignment
- LOADING: Loading cargo from origin stores
- IN_TRANSIT: Sailing between ports
- WAITING_FOR_BERTH: Queued at destination berth
- UNLOADING: Unloading cargo to destination stores
- UNLOADING_WAITING_FOR_SPACE: Waiting for destination store capacity
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
import simpy

from sim_run_types import TransportRoute


class ShipState(Enum):
    IDLE = "IDLE"
    LOADING = "LOADING"
    LOADING_WAITING_FOR_PRODUCT = "LOADING_WAITING_FOR_PRODUCT"
    IN_TRANSIT = "IN_TRANSIT"
    WAITING_FOR_BERTH = "WAITING_FOR_BERTH"
    UNLOADING = "UNLOADING"
    UNLOADING_WAITING_FOR_SPACE = "UNLOADING_WAITING_FOR_SPACE"
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
                           store_capacities: Optional[Dict[str, float]] = None) -> Tuple[float, float, float, float]:
    """
    Score a route based on utilization, urgency, travel time, and origin overflow risk.
    Returns (score, utilization_pct, urgency_score, overflow_bonus)
    
    sole_supplier_stores: Set of store keys that only have ONE route serving them.
    production_rates: Dict mapping store_key -> production rate (tons/hour) INTO that store.
    store_capacities: Dict mapping store_key -> capacity.
    """
    load_steps = [s for s in itinerary if s.get('kind') == 'load']
    unload_steps = [s for s in itinerary if s.get('kind') == 'unload']
    
    total_available = 0.0
    total_capacity = float(n_holds * payload_per_hold)
    
    for step in load_steps:
        sk = step.get('store_key')
        if sk and sk in stores:
            total_available += float(stores[sk].level)
    
    utilization = min(1.0, total_available / max(total_capacity, 1.0))
    
    urgency = 0.0
    sole_supplier_bonus = 0.0
    for step in unload_steps:
        sk = step.get('store_key')
        if sk and sk in stores:
            level = float(stores[sk].level)
            rate = demand_rates.get(sk, 0.0)
            if rate > 0:
                days_of_stock = level / (rate * 24)
                step_urgency = max(0, 10 - days_of_stock)
                urgency = max(urgency, step_urgency)
                
                if sole_supplier_stores and sk in sole_supplier_stores:
                    if days_of_stock < 60:
                        sole_supplier_bonus = max(sole_supplier_bonus, 100 * (1 - days_of_stock / 60))
    
    overflow_bonus = 0.0
    prod_rates = production_rates or {}
    capacities = store_capacities or {}
    for step in load_steps:
        sk = step.get('store_key')
        if sk and sk in stores:
            level = float(stores[sk].level)
            prod_rate = prod_rates.get(sk, 0.0)
            capacity = capacities.get(sk, float('inf'))
            
            if prod_rate > 0 and capacity < float('inf'):
                space_left = capacity - level
                hours_to_full = space_left / prod_rate if prod_rate > 0 else float('inf')
                fill_pct = level / capacity
                
                if fill_pct > 0.50:
                    overflow_bonus = max(overflow_bonus, 150 * (fill_pct - 0.50) / 0.50)
    
    travel_time = 0.0
    sail_steps = [s for s in itinerary if s.get('kind') == 'sail']
    for step in sail_steps:
        from_loc = step.get('from', '')
        to_loc = step.get('to', step.get('location', ''))
        if from_loc and to_loc:
            nm = _get_nm_distance(nm_distances, from_loc, to_loc)
            pilot_out = _get_pilot_hours(berth_info, from_loc, 'out')
            pilot_in = _get_pilot_hours(berth_info, to_loc, 'in')
            travel_time += (nm / max(speed_knots, 1.0)) + pilot_out + pilot_in
    
    score = (utilization * 50) + (urgency * 30) + sole_supplier_bonus + overflow_bonus - (travel_time * 0.1)
    
    return (score, utilization, urgency, overflow_bonus)


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
                store_capacities: Optional[Dict[str, float]] = None,
                sim=None):
    """
    Main ship vessel process. This is called once per vessel by the simulation core.
    Follows the state machine: IDLE -> LOADING (-> LOADING_WAITING_FOR_PRODUCT -> LOADING) -> IN_TRANSIT -> WAITING_FOR_BERTH -> UNLOADING (-> UNLOADING_WAITING_FOR_SPACE -> UNLOADING) -> IDLE
    
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
    chosen_itinerary = None
    itinerary_idx = 0
    demand_rates_map = demand_rates or {}
    active_berth = None
    active_berth_req = None
    current_route_id = None  # Specific route ID (e.g., 1.1, 1.2)
    total_waited_at_location = 0.0
    last_prob_wait_location = None # Tracks the last location where we performed a probabilistic wait check
    
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
            candidate_its = [it for it in route.itineraries 
                           if _get_start_location(it) == current_location]
            
            if not candidate_its:
                current_location = origin_location
                yield env.timeout(1)
                continue
            
            best_it = None
            best_score = -float('inf')
            
            prod_rates = production_rates or {}
            capacities = store_capacities or {}
            
            for it in candidate_its:
                score, util, _, overflow_bonus = _calculate_route_score(
                    it, stores, payload_per_hold, n_holds, 
                    demand_rates_map, nm_distances, speed_knots, berth_info,
                    sole_supplier_stores, prod_rates, capacities
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
            
            if best_it is None:
                log_func(
                    process="Move",
                    event="Idle",
                    location=current_location,
                    equipment="Ship",
                    product=None,
                    time=1.0,
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
            state = ShipState.LOADING
            log_state_change(state)
            cargo = {}
            itinerary_idx = 0
            total_waited_at_location = 0.0
            last_prob_wait_location = None # Reset on new itinerary
            
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
                        
                        active_berth = berth
                        active_berth_req = req
                        # Don't log state change here as it's already LOADING or UNLOADING

                    cont = stores[store_key]
                    load_rate, _ = _get_store_rates(store_rates, store_key, default_load_rate, default_unload_rate)
                    
                    already_loaded = sum(cargo.values())
                    remaining_cap = max(0, (n_holds * payload_per_hold) - already_loaded)
                    holds_remaining = int(remaining_cap // payload_per_hold)
                    
                    total_loaded_this_stop = 0.0
                    time_per_hold = payload_per_hold / max(load_rate, 1.0)
                    
                    for hold_num in range(holds_remaining):
                        # Wait for Step 3: Reduce inventory by the "Ship Load" Qty
                        if sim:
                            yield sim.wait_for_step(3)

                        if cont.level >= payload_per_hold:
                            log_func(
                                process="Move",
                                event="Load",
                                location=location,
                                equipment="Ship",
                                product=product,
                                qty=payload_per_hold,
                                time=round(time_per_hold, 2),
                                from_store=store_key,
                                from_level=float(cont.level - payload_per_hold),
                                to_store=None,
                                to_level=None,
                                route_id=current_route_id or route_group,
                                vessel_id=vessel_id,
                                ship_state=state.value,
                                override_time_h=env.now
                            )
                            yield cont.get(payload_per_hold)
                            
                            yield env.timeout(time_per_hold)
                            total_loaded_this_stop += payload_per_hold
                        else:
                            waited_this_step = 0.0
                            while cont.level < payload_per_hold and total_waited_at_location < max_wait_product_h:
                                log_func(
                                    process="Move",
                                    event="Loading - Waiting for Product",
                                    location=location,
                                    equipment="Ship",
                                    product=product,
                                    time=1.0,
                                    from_store=store_key,
                                    to_store=None,
                                    route_id=current_route_id or route_group,
                                    vessel_id=vessel_id,
                                    ship_state=ShipState.LOADING_WAITING_FOR_PRODUCT.value,
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
                                
                                log_func(
                                    process="Move",
                                    event="Load",
                                    location=location,
                                    equipment="Ship",
                                    product=product,
                                    qty=payload_per_hold,
                                    time=round(time_per_hold, 2),
                                    from_store=store_key,
                                    from_level=float(cont.level - payload_per_hold),
                                    to_store=None,
                                    to_level=None,
                                    route_id=current_route_id or route_group,
                                    vessel_id=vessel_id,
                                    ship_state=state.value,
                                    override_time_h=env.now
                                )
                                yield cont.get(payload_per_hold)
                            
                                yield env.timeout(time_per_hold)
                                total_loaded_this_stop += payload_per_hold
                            else:
                                break
                    
                    if total_loaded_this_stop > 0:
                        cargo[product] = cargo.get(product, 0.0) + total_loaded_this_stop
                
                itinerary_idx += 1
            
            elif kind == 'sail':
                if active_berth is not None:
                    # Check if the next location is different. If it's a 'sail', it almost certainly is.
                    active_berth.release(active_berth_req)
                    active_berth = None
                    active_berth_req = None
                
                total_loaded = sum(cargo.values())
                utilization = total_loaded / max(n_holds * payload_per_hold, 1.0)
                
                # Logic: If we are at the end of loading at this location (reached 'sail')
                # and we don't have enough cargo to meet min_utilization, we have a choice:
                # 1. Stay and wait more (already done by the load loop's wait)
                # 2. Go anyway if we have SOME cargo (if allowed)
                # 3. Abort/Idle if we have NO cargo or insufficient cargo.
                
                if utilization >= min_utilization:
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
                        qty=total_loaded,
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
                            qty=total_loaded,
                            from_store=None,
                            to_store=None,
                            route_id=current_route_id or route_group,
                            vessel_id=vessel_id,
                            ship_state=state.value,
                            override_time_h=env.now
                        )
                        state = ShipState.IN_TRANSIT
                        log_state_change(state)
                    else:
                        # NOTHING loaded and we either finished all load steps or timed out.
                        # Let's reset to IDLE so we can pick a better route.
                        state = ShipState.IDLE
                        log_state_change(state)
                        chosen_itinerary = None
                        current_route_id = None
                        cargo = {}
                        
                        if sim:
                            yield sim.wait_for_step(7)
                        else:
                            yield env.timeout(1)
            else:
                itinerary_idx += 1
        
        elif state == ShipState.IN_TRANSIT:
            if itinerary_idx >= len(chosen_itinerary):
                current_location = origin_location
                state = ShipState.IDLE
                log_state_change(state)
                cargo = {}
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
                            continue # Jump to return leg
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
                if active_berth is not None and step.get('location', current_location) == current_location:
                    state = ShipState.UNLOADING
                else:
                    state = ShipState.WAITING_FOR_BERTH
                log_state_change(state)
            
            elif kind == 'load':
                # If we already have a berth at this location, go straight to LOADING
                if active_berth is not None and step.get('location', current_location) == current_location:
                    state = ShipState.LOADING
                else:
                    state = ShipState.LOADING
                log_state_change(state)
            
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
                cargo = {}
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
                        # Require at least one hold worth of space (or full cargo if less than one hold)
                        # This avoids the ship waiting 48h when it can unload most of its cargo.
                        required_space = min(carried, payload_per_hold)
                        if room >= required_space - 1e-6:
                            break
                        
                        if state != ShipState.UNLOADING_WAITING_FOR_SPACE:
                            state = ShipState.UNLOADING_WAITING_FOR_SPACE
                            log_state_change(state, unload_location)
                            
                        log_func(
                            process="Move",
                            event="Unloading - Waiting for Space",
                            location=unload_location,
                            equipment="Ship",
                            product=product,
                            time=1.0,
                            from_store=None,
                            to_store=store_key,
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
                    
                    # After waiting, unload what we can (should be full if room became available)
                    room = float(cont.capacity - cont.level)
                    
                    # Quantize unload to multiples of hold size
                    # We unload as many full holds as will fit in the remaining room
                    num_holds_to_unload = int(min(carried, room) // payload_per_hold)
                    qty_to_unload = float(num_holds_to_unload * payload_per_hold)
                    
                    if qty_to_unload > 1e-6:
                        unload_time = qty_to_unload / max(unload_rate, 1.0)
                        
                        # Wait for Step 7: Increase inventory by the "Ship Offload" Qty
                        if sim:
                            yield sim.wait_for_step(7)

                        log_func(
                            process="Move",
                            event="Unload",
                            location=unload_location,
                            equipment="Ship",
                            product=product,
                            qty=qty_to_unload,
                            time=round(unload_time, 2),
                            from_store=None,
                            from_level=None,
                            to_store=store_key,
                            to_level=float(cont.level + qty_to_unload),
                            route_id=current_route_id or route_group,
                            vessel_id=vessel_id,
                            ship_state=state.value,
                            override_time_h=env.now
                        )

                        yield cont.put(qty_to_unload)
                        
                        yield env.timeout(unload_time)
                        
                        cargo[product] = carried - qty_to_unload
                
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
            if sim:
                for _ in range(24):
                    yield sim.wait_for_step(7)
            else:
                yield env.timeout(24)
