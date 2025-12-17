# sim_run_core_move_ship.py
"""
Ship transport module implementing the Shipping Model Specification.

Ships move through explicit states:
- IDLE: Waiting at origin for route assignment
- LOADING: Loading cargo from origin stores
- IN_TRANSIT: Sailing between ports
- WAITING_FOR_BERTH: Queued at destination berth
- UNLOADING: Unloading cargo to destination stores
- ERROR: Unable to proceed (logged, remains idle)

Planning logic:
- Evaluate candidate routes based on utilization (hull fill %) and urgency
- At least 60% hulls must be filled before departure
- Travel time = distance / speed + pilot_in + pilot_out hours

Execution:
- FIFO berth queues at each port
- Load/unload times based on store rates
- Ships complete full route sequence then return to origin pool
"""
from __future__ import annotations
from enum import Enum
from typing import Callable, Dict, List, Tuple, Optional, Any
import simpy

from sim_run_types import TransportRoute


class ShipState(Enum):
    IDLE = "IDLE"
    LOADING = "LOADING"
    IN_TRANSIT = "IN_TRANSIT"
    WAITING_FOR_BERTH = "WAITING_FOR_BERTH"
    UNLOADING = "UNLOADING"
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
                           payload_per_hull: float, n_hulls: int, demand_rates: Dict[str, float],
                           nm_distances: Dict, speed_knots: float, berth_info: Dict,
                           sole_supplier_stores: Optional[set] = None) -> Tuple[float, float, float]:
    """
    Score a route based on utilization, urgency, and travel time.
    Returns (score, utilization_pct, urgency_score)
    
    sole_supplier_stores: Set of store keys that only have ONE route serving them.
    Routes serving these stores get a bonus to ensure they get selected occasionally.
    """
    load_steps = [s for s in itinerary if s.get('kind') == 'load']
    unload_steps = [s for s in itinerary if s.get('kind') == 'unload']
    
    total_available = 0.0
    total_capacity = float(n_hulls * payload_per_hull)
    
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
                
                # Bonus for sole supplier routes - stronger when stock is low
                # Use 60-day threshold to ensure these routes get selected before running out
                if sole_supplier_stores and sk in sole_supplier_stores:
                    if days_of_stock < 60:
                        # Strong bonus that can compete with high-utilization routes
                        sole_supplier_bonus = max(sole_supplier_bonus, 100 * (1 - days_of_stock / 60))
    
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
    
    score = (utilization * 50) + (urgency * 30) + sole_supplier_bonus - (travel_time * 0.1)
    
    return (score, utilization, urgency)


def transporter(env: simpy.Environment, route: TransportRoute,
                stores: Dict[str, simpy.Container],
                port_berths: Dict[str, simpy.Resource],
                log_func: Callable,
                store_rates: Dict[str, Tuple[float, float]],
                require_full: bool = True,
                demand_rates: Optional[Dict[str, float]] = None,
                vessel_id: int = 1,
                sole_supplier_stores: Optional[set] = None):
    """
    Main ship vessel process. This is called once per vessel by the simulation core.
    Follows the state machine: IDLE -> LOADING -> IN_TRANSIT -> WAITING_FOR_BERTH -> UNLOADING -> IDLE
    
    sole_supplier_stores: Set of store keys that only have ONE route serving them.
    """
    if not getattr(route, 'itineraries', None):
        while True:
            yield env.timeout(24)
    
    speed_knots = float(getattr(route, 'speed_knots', 10.0) or 10.0)
    n_hulls = int(getattr(route, 'hulls_per_vessel', 5) or 5)
    payload_per_hull = float(getattr(route, 'payload_per_hull_t', 5000.0) or 5000.0)
    route_group = getattr(route, 'route_group', 'Ship')
    berth_info = getattr(route, 'berth_info', {}) or {}
    nm_distances = getattr(route, 'nm_distance', {}) or {}
    default_load_rate = float(getattr(route, 'load_rate_tph', 500.0) or 500.0)
    default_unload_rate = float(getattr(route, 'unload_rate_tph', 500.0) or 500.0)
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
            ship_state=new_state.value
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
            
            for it in candidate_its:
                score, util, _ = _calculate_route_score(
                    it, stores, payload_per_hull, n_hulls, 
                    demand_rates_map, nm_distances, speed_knots, berth_info,
                    sole_supplier_stores
                )
                
                # Check if this route serves a sole-supplier store that's critically low
                # If so, allow lower utilization (min 1 hull = 20% for 5-hull ship)
                required_util = min_utilization
                if sole_supplier_stores:
                    for step in it:
                        if step.get('kind') == 'unload':
                            sk = step.get('store_key')
                            if sk and sk in sole_supplier_stores and sk in stores:
                                level = float(stores[sk].level)
                                rate = demand_rates_map.get(sk, 0.0)
                                if rate > 0:
                                    days_of_stock = level / (rate * 24)
                                    # If less than 60 days stock at a sole-supplier dest, lower the bar
                                    if days_of_stock < 60:
                                        required_util = 0.20  # Allow with just 1 hull
                                        break
                
                if util >= required_util and score > best_score:
                    best_score = score
                    best_it = it
            
            if best_it is None:
                yield env.timeout(1)
                continue
            
            chosen_itinerary = best_it
            current_route_id = _get_route_id_from_itinerary(chosen_itinerary)
            state = ShipState.LOADING
            log_state_change(state)
            cargo = {}
            itinerary_idx = 0
            
            for i, step in enumerate(chosen_itinerary):
                if step.get('kind') == 'start':
                    itinerary_idx = i + 1
                    break
        
        elif state == ShipState.LOADING:
            if itinerary_idx >= len(chosen_itinerary):
                state = ShipState.IN_TRANSIT
                log_state_change(state)
                continue
            
            step = chosen_itinerary[itinerary_idx]
            kind = step.get('kind')
            
            if kind == 'load':
                store_key = step.get('store_key')
                product = step.get('product')
                location = step.get('location', current_location)
                
                if store_key and store_key in stores:
                    berth = _get_berth(env, port_berths, berth_info, location)
                    
                    with berth.request() as req:
                        yield req
                        
                        cont = stores[store_key]
                        load_rate, _ = _get_store_rates(store_rates, store_key, default_load_rate, default_unload_rate)
                        
                        already_loaded = sum(cargo.values())
                        remaining_cap = max(0, (n_hulls * payload_per_hull) - already_loaded)
                        available = float(cont.level)
                        
                        # Round down to full hulls only (no partial hull loads)
                        max_loadable = min(remaining_cap, available)
                        full_hulls = int(max_loadable // payload_per_hull)
                        qty_to_load = full_hulls * payload_per_hull
                        
                        if qty_to_load >= payload_per_hull and cont.level >= qty_to_load:
                            yield cont.get(qty_to_load)
                            from_level = cont.level
                            
                            load_time = qty_to_load / max(load_rate, 1.0)
                            yield env.timeout(load_time)
                            
                            cargo[product] = cargo.get(product, 0.0) + qty_to_load
                            
                            log_func(
                                process="Move",
                                event="Load",
                                location=location,
                                equipment="Ship",
                                product=product,
                                qty=qty_to_load,
                                from_store=store_key,
                                from_level=from_level,
                                to_store=None,
                                to_level=None,
                                route_id=current_route_id or route_group
                            )
                
                itinerary_idx += 1
            
            elif kind == 'sail':
                total_loaded = sum(cargo.values())
                utilization = total_loaded / max(n_hulls * payload_per_hull, 1.0)
                
                if utilization >= min_utilization:
                    state = ShipState.IN_TRANSIT
                    log_state_change(state)
                else:
                    for i, s in enumerate(chosen_itinerary):
                        if s.get('kind') == 'load':
                            itinerary_idx = i
                            break
                    yield env.timeout(1)
            else:
                itinerary_idx += 1
        
        elif state == ShipState.IN_TRANSIT:
            if itinerary_idx >= len(chosen_itinerary):
                current_location = origin_location
                state = ShipState.IDLE
                log_state_change(state)
                cargo = {}
                yield env.timeout(0.01)
                continue
            
            step = chosen_itinerary[itinerary_idx]
            kind = step.get('kind')
            
            if kind == 'sail':
                from_loc = step.get('from', current_location)
                to_loc = step.get('to', step.get('location'))
                
                if to_loc:
                    nm = _get_nm_distance(nm_distances, from_loc, to_loc)
                    travel_hours = nm / max(speed_knots, 1.0)
                    
                    pilot_out = _get_pilot_hours(berth_info, from_loc, 'out')
                    pilot_in = _get_pilot_hours(berth_info, to_loc, 'in')
                    total_time = travel_hours + pilot_out + pilot_in
                    
                    if total_time > 0:
                        yield env.timeout(total_time)
                    
                    current_location = to_loc
                
                itinerary_idx += 1
            
            elif kind == 'unload':
                state = ShipState.WAITING_FOR_BERTH
                log_state_change(state)
            
            elif kind == 'load':
                state = ShipState.LOADING
                log_state_change(state)
            
            else:
                itinerary_idx += 1
        
        elif state == ShipState.WAITING_FOR_BERTH:
            step = chosen_itinerary[itinerary_idx]
            unload_location = step.get('location', current_location)
            
            berth = _get_berth(env, port_berths, berth_info, unload_location)
            berth_req = berth.request()
            yield berth_req
            
            state = ShipState.UNLOADING
            log_state_change(state, unload_location)
            active_berth = berth
            active_berth_req = berth_req
        
        elif state == ShipState.UNLOADING:
            if itinerary_idx >= len(chosen_itinerary):
                if active_berth is not None:
                    active_berth.release(active_berth_req)
                    active_berth = None
                current_location = origin_location
                state = ShipState.IDLE
                log_state_change(state)
                cargo = {}
                yield env.timeout(0.01)
                continue
            
            step = chosen_itinerary[itinerary_idx]
            kind = step.get('kind')
            
            if kind == 'unload':
                store_key = step.get('store_key')
                product = step.get('product')
                unload_location = step.get('location', current_location)
                
                if store_key and store_key in stores and product in cargo:
                    cont = stores[store_key]
                    _, unload_rate = _get_store_rates(store_rates, store_key, default_load_rate, default_unload_rate)
                    
                    carried = float(cargo.get(product, 0.0))
                    
                    # Wait until there's room for the FULL cargo (no partial unloads)
                    # Max wait: 48 hours to avoid indefinite blocking
                    wait_hours = 0
                    max_wait = 48
                    while wait_hours < max_wait:
                        room = float(cont.capacity - cont.level)
                        if room >= carried - 1e-6:
                            break
                        yield env.timeout(1.0)
                        wait_hours += 1
                    
                    # After waiting, unload what we can (should be full if room became available)
                    room = float(cont.capacity - cont.level)
                    qty_to_unload = min(carried, room)  # Full cargo if room, else max possible
                    
                    if qty_to_unload > 1e-6:
                        unload_time = qty_to_unload / max(unload_rate, 1.0)
                        yield env.timeout(unload_time)
                        
                        yield cont.put(qty_to_unload)
                        to_level = cont.level
                        
                        cargo[product] = carried - qty_to_unload
                        
                        log_func(
                            process="Move",
                            event="Unload",
                            location=unload_location,
                            equipment="Ship",
                            product=product,
                            qty=qty_to_unload,
                            from_store=None,
                            from_level=None,
                            to_store=store_key,
                            to_level=to_level,
                            route_id=current_route_id or route_group
                        )
                
                itinerary_idx += 1
            else:
                if active_berth is not None:
                    active_berth.release(active_berth_req)
                    active_berth = None
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
                route_id=route_group
            )
            state = ShipState.IDLE
            yield env.timeout(24)
