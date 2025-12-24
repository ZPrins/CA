# sim_run_core_move_ship_handlers.py
"""
Ship state handlers for loading, unloading, transit, and idle states.

Each handler processes the corresponding ship state and returns the next state.
"""
from __future__ import annotations
from typing import Optional, Generator, Any
from collections import defaultdict
import random

from sim_run_core_move_ship_types import ShipState, ShipContext
from sim_run_core_move_ship_utils import (
    get_berth, get_pilot_hours, get_nm_distance, get_store_rates,
    get_start_location, get_route_id_from_itinerary
)
from sim_run_core_move_ship_scoring import calculate_route_score


def handle_idle_state(ctx: ShipContext) -> Generator[Any, Any, Optional[str]]:
    """
    Handle IDLE state: select best route and transition to LOADING.

    Returns: 'continue' to restart main loop, None to proceed to next state.
    """
    # Ensure t_state is synced with current cargo
    if ctx.t_state is not None:
        ctx.t_state['cargo'] = ctx.cargo

    candidate_its = [it for it in ctx.itineraries
                     if get_start_location(it) == ctx.current_location]

    if not candidate_its:
        if ctx.current_location != ctx.origin_location:
            nm = get_nm_distance(ctx.nm_distances, ctx.current_location, ctx.origin_location)
            if nm > 0:
                travel_hours = nm / max(ctx.speed_knots, 1.0)
                pilot_out = get_pilot_hours(ctx.berth_info, ctx.current_location, 'out')
                pilot_in = get_pilot_hours(ctx.berth_info, ctx.origin_location, 'in')
                total_time = travel_hours + pilot_out + pilot_in

                ctx.log_func(
                    process="Move",
                    event="Transit",
                    location=ctx.current_location,
                    equipment="Ship",
                    product=None,
                    time=round(total_time, 2),
                    from_store=None,
                    to_store=None,
                    route_id=ctx.current_route_id or ctx.route_group,
                    vessel_id=ctx.vessel_id,
                    ship_state=ctx.state.value,
                    override_time_h=ctx.env.now
                )
                yield ctx.env.timeout(total_time)
        ctx.current_location = ctx.origin_location
        # Retrieve small wait timeout from settings if available, else 1.0
        wait_h = 1.0
        if ctx.sim and hasattr(ctx.sim, 'settings'):
            wait_h = float(ctx.sim.settings.get('ship_idle_wait_h', 1.0))
        yield ctx.env.timeout(wait_h)
        return 'continue'

    best_it = None
    best_score = -float('inf')
    best_suggested_load = 0.0

    prod_rates = ctx.production_rates or {}
    capacities = ctx.store_capacity_map or {}

    # Pre-aggregate pending deliveries from OTHER vessels once per IDLE step
    other_pending_map = defaultdict(float)
    if ctx.sim and hasattr(ctx.sim, 'pending_deliveries'):
        for (vid, sk), qty in ctx.sim.pending_deliveries.items():
            if vid != ctx.vessel_id:
                other_pending_map[sk] += qty

    for it in candidate_its:
        # Don't select a route if another vessel is already delivering to its target store
        is_already_served = False
        for step in it:
            if step.get('kind') == 'unload':
                sk = step.get('store_key')
                if sk and other_pending_map[sk] > 1e-6:
                    is_already_served = True
                    break

        if is_already_served:
            continue

        score, util, _, overflow_bonus, suggested_to_load = calculate_route_score(
            it, ctx.stores, ctx.payload_per_hold, ctx.n_holds,
            ctx.demand_rates, ctx.nm_distances, ctx.speed_knots, ctx.berth_info,
            ctx.sole_supplier_stores, prod_rates, capacities,
            other_pending_map,
            ctx.cargo
        )

        required_util = ctx.min_utilization

        if overflow_bonus > 60:
            required_util = 0.20

        if ctx.sole_supplier_stores:
            for step in it:
                if step.get('kind') == 'unload':
                    sk = step.get('store_key')
                    if sk and sk in ctx.sole_supplier_stores and sk in ctx.stores:
                        level = float(ctx.stores[sk].level)
                        rate = ctx.demand_rates.get(sk, 0.0)
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
        reason = "No suitable itinerary"
        ctx.log_func(
            process="Move",
            event=reason,
            location=ctx.current_location,
            equipment="Ship",
            product=str(list(ctx.cargo.keys())) if ctx.cargo else None,
            time=1.0,
            unmet_demand=0.0,
            qty=0,
            from_store=None,
            to_store=None,
            route_id=ctx.current_route_id or ctx.route_group,
            vessel_id=ctx.vessel_id,
            ship_state=ctx.state.value,
            override_time_h=ctx.env.now
        )
        if ctx.sim:
            yield ctx.sim.wait_for_step(7)
        else:
            yield ctx.env.timeout(1)
        return 'continue'

    ctx.chosen_itinerary = best_it
    ctx.current_route_id = get_route_id_from_itinerary(ctx.chosen_itinerary)

    # Clear existing cargo if it doesn't match the new itinerary's products
    if ctx.cargo:
        new_products = {step.get('product') for step in ctx.chosen_itinerary if step.get('kind') in ('load', 'unload')}
        mismatch = any(p not in new_products for p in ctx.cargo.keys())
        if mismatch:
            ctx.log_func(
                process="Move",
                event="Clear Cargo",
                location=ctx.current_location,
                equipment="Ship",
                product=str(list(ctx.cargo.keys())),
                qty=sum(ctx.cargo.values()),
                time=0.0,
                unmet_demand=0.0,
                from_store=None,
                to_store=None,
                route_id=ctx.current_route_id or ctx.route_group,
                vessel_id=ctx.vessel_id,
                ship_state=ctx.state.value,
                override_time_h=ctx.env.now
            )
            ctx.cargo = {}
            if ctx.t_state is not None:
                ctx.t_state['cargo'] = ctx.cargo

    if ctx.t_state is not None:
        ctx.t_state['route_id'] = ctx.current_route_id or ctx.route_group
    ctx.state = ShipState.LOADING
    ctx.log_state_change(ctx.state)

    # Register pending deliveries
    if ctx.sim and hasattr(ctx.sim, 'pending_deliveries'):
        projected_cargo = ctx.cargo.copy()
        primary_load_product = None
        for step in ctx.chosen_itinerary:
            if step.get('kind') == 'load':
                primary_load_product = step.get('product')
                break

        if primary_load_product:
            projected_cargo[primary_load_product] = projected_cargo.get(primary_load_product, 0.0) + best_suggested_load

        for step in ctx.chosen_itinerary:
            if step.get('kind') == 'unload':
                sk = step.get('store_key')
                prod = step.get('product')
                if sk and prod in projected_cargo:
                    qty_to_drop = projected_cargo[prod]
                    ctx.sim.pending_deliveries[(ctx.vessel_id, sk)] += qty_to_drop

    # Track planned load to respect it during loading phase
    ctx.planned_load_remaining = best_suggested_load
    ctx.itinerary_idx = 0
    ctx.total_waited_at_location = 0.0
    ctx.last_prob_wait_location = None

    for i, step in enumerate(ctx.chosen_itinerary):
        if step.get('kind') == 'start':
            ctx.itinerary_idx = i + 1
            break

    return None


def _acquire_berth(ctx: ShipContext, location: str, product: Optional[str] = None) -> Generator[Any, Any, None]:
    """
    Acquire a berth at the given location with probabilistic and FIFO waiting.
    """
    # 1. Probabilistic Berth Wait (Check ONLY on arrival at new location)
    if ctx.last_prob_wait_location != location:
        loc_info = ctx.berth_info.get(location, {})
        p_occ = loc_info.get('p_occupied', 0.0)

        while random.random() < p_occ:
            ctx.log_func(
                process="Move",
                event="Wait for Berth",
                location=location,
                equipment="Ship",
                product=product,
                time=1.0,
                unmet_demand=0.0,
                from_store=None,
                to_store=None,
                route_id=ctx.current_route_id or ctx.route_group,
                vessel_id=ctx.vessel_id,
                ship_state=ShipState.WAITING_FOR_BERTH.value,
                override_time_h=ctx.env.now
            )
            if ctx.sim:
                yield ctx.sim.wait_for_step(7)
            else:
                yield ctx.env.timeout(1.0)

        ctx.last_prob_wait_location = location

    # 2. Physical Berth Request (FIFO Resource)
    berth = get_berth(ctx.env, ctx.port_berths, ctx.berth_info, location)
    req = berth.request()

    if not req.triggered:
        while not req.triggered:
            ctx.log_func(
                process="Move",
                event="Wait for Berth",
                location=location,
                equipment="Ship",
                product=product,
                time=1.0,
                unmet_demand=0.0,
                from_store=None,
                to_store=None,
                route_id=ctx.current_route_id or ctx.route_group,
                vessel_id=ctx.vessel_id,
                ship_state=ShipState.WAITING_FOR_BERTH.value,
                override_time_h=ctx.env.now
            )
            if ctx.sim:
                yield ctx.sim.env.any_of([req, ctx.sim.wait_for_step(7)])
            else:
                yield ctx.env.any_of([req, ctx.env.timeout(1)])
    else:
        # Immediate berth acquisition - log it for transparency
        ctx.log_func(
            process="Move",
            event="Wait for Berth",
            location=location,
            equipment="Ship",
            product=product,
            time=0.0,
            unmet_demand=0.0,
            from_store=None,
            to_store=None,
            route_id=ctx.current_route_id or ctx.route_group,
            vessel_id=ctx.vessel_id,
            ship_state=ShipState.WAITING_FOR_BERTH.value,
            override_time_h=ctx.env.now
        )

    ctx.active_berth = berth
    ctx.active_berth_req = req


def handle_loading_state(ctx: ShipContext) -> Generator[Any, Any, Optional[str]]:
    """
    Handle LOADING state: load cargo from stores.

    Returns: 'continue' to restart main loop, None to proceed to next state.
    """
    if ctx.itinerary_idx >= len(ctx.chosen_itinerary):
        if ctx.active_berth is not None:
            ctx.active_berth.release(ctx.active_berth_req)
            ctx.active_berth = None
            ctx.active_berth_req = None
        ctx.state = ShipState.IN_TRANSIT
        ctx.log_state_change(ctx.state)
        return 'continue'

    step = ctx.chosen_itinerary[ctx.itinerary_idx]
    kind = step.get('kind')

    if kind == 'load':
        store_key = step.get('store_key')
        product = step.get('product')
        location = step.get('location', ctx.current_location)

        if store_key and store_key in ctx.stores:
            # Request berth if not already held
            if ctx.active_berth is None:
                yield from _acquire_berth(ctx, location, product)

            cont = ctx.stores[store_key]
            load_rate, _ = get_store_rates(ctx.store_rates, store_key, ctx.default_load_rate, ctx.default_unload_rate)

            already_loaded = sum(ctx.cargo.values())
            remaining_cap = max(0, (ctx.n_holds * ctx.payload_per_hold) - already_loaded)
            remaining_cap = min(remaining_cap, ctx.planned_load_remaining)
            holds_remaining = int(remaining_cap // ctx.payload_per_hold)

            total_loaded_this_stop = 0.0
            time_per_hold = ctx.payload_per_hold / max(load_rate, 1.0)

            for hold_num in range(holds_remaining):
                if ctx.sim:
                    yield ctx.sim.wait_for_step(3)

                if cont.level >= ctx.payload_per_hold:
                    start_hold_t = ctx.env.now
                    yield cont.get(ctx.payload_per_hold)
                    yield ctx.env.timeout(time_per_hold)

                    ctx.log_func(
                        process="Move",
                        event="Load",
                        location=location,
                        equipment="Ship",
                        product=product,
                        qty=ctx.payload_per_hold,
                        time=round(time_per_hold, 2),
                        unmet_demand=0.0,
                        qty_out=ctx.payload_per_hold,
                        from_store=store_key,
                        from_level=float(cont.level),
                        from_fill_pct=float(cont.level) / cont.capacity if cont.capacity > 0 else 0.0,
                        to_store=None,
                        to_level=None,
                        route_id=ctx.current_route_id or ctx.route_group,
                        vessel_id=ctx.vessel_id,
                        ship_state=ctx.state.value,
                        override_time_h=start_hold_t
                    )

                    total_loaded_this_stop += ctx.payload_per_hold
                    ctx.planned_load_remaining -= ctx.payload_per_hold
                else:
                    waited_this_step = 0.0
                    while cont.level < ctx.payload_per_hold and ctx.total_waited_at_location < ctx.max_wait_product_h:
                        ctx.log_func(
                            process="Move",
                            event="Waiting for Product",
                            location=location,
                            equipment="Ship",
                            product=product,
                            time=1.0,
                            unmet_demand=0.0,
                            from_store=store_key,
                            from_level=float(cont.level),
                            from_fill_pct=float(cont.level) / cont.capacity if cont.capacity > 0 else 0.0,
                            to_store=None,
                            route_id=ctx.current_route_id or ctx.route_group,
                            vessel_id=ctx.vessel_id,
                            ship_state=ShipState.WAITING_FOR_PRODUCT.value,
                            override_time_h=ctx.env.now
                        )
                        if ctx.sim:
                            yield ctx.sim.wait_for_step(7)
                        else:
                            yield ctx.env.timeout(1.0)
                        waited_this_step += 1.0
                        ctx.total_waited_at_location += 1.0

                    if cont.level >= ctx.payload_per_hold:
                        if ctx.sim:
                            yield ctx.sim.wait_for_step(3)

                        start_hold_t = ctx.env.now
                        yield cont.get(ctx.payload_per_hold)
                        yield ctx.env.timeout(time_per_hold)

                        ctx.log_func(
                            process="Move",
                            event="Load",
                            location=location,
                            equipment="Ship",
                            product=product,
                            qty=ctx.payload_per_hold,
                            time=round(time_per_hold, 2),
                            unmet_demand=0.0,
                            qty_out=ctx.payload_per_hold,
                            from_store=store_key,
                            from_level=float(cont.level),
                            from_fill_pct=float(cont.level) / cont.capacity if cont.capacity > 0 else 0.0,
                            to_store=None,
                            to_level=None,
                            route_id=ctx.current_route_id or ctx.route_group,
                            vessel_id=ctx.vessel_id,
                            ship_state=ctx.state.value,
                            override_time_h=start_hold_t
                        )

                        total_loaded_this_stop += ctx.payload_per_hold
                        ctx.planned_load_remaining -= ctx.payload_per_hold
                    else:
                        break

            if total_loaded_this_stop > 0:
                ctx.cargo[product] = ctx.cargo.get(product, 0.0) + total_loaded_this_stop

        ctx.itinerary_idx += 1

    elif kind == 'sail':
        total_loaded = sum(ctx.cargo.values())
        utilization = total_loaded / max(ctx.n_holds * ctx.payload_per_hold, 1.0)

        if utilization >= ctx.min_utilization:
            if ctx.active_berth is not None:
                ctx.active_berth.release(ctx.active_berth_req)
                ctx.active_berth = None
                ctx.active_berth_req = None

            # Adjust pending deliveries if loaded less than planned
            if ctx.planned_load_remaining > 1e-6 and ctx.sim and hasattr(ctx.sim, 'pending_deliveries'):
                primary_load_product = None
                for step in ctx.chosen_itinerary:
                    if step.get('kind') == 'load':
                        primary_load_product = step.get('product')
                        break

                if primary_load_product:
                    for step in ctx.chosen_itinerary:
                        if step.get('kind') == 'unload' and step.get('product') == primary_load_product:
                            sk = step.get('store_key')
                            if (ctx.vessel_id, sk) in ctx.sim.pending_deliveries:
                                ctx.sim.pending_deliveries[(ctx.vessel_id, sk)] = max(0.0, ctx.sim.pending_deliveries[(ctx.vessel_id, sk)] - ctx.planned_load_remaining)

            ctx.state = ShipState.IN_TRANSIT
            ctx.log_state_change(ctx.state)
        elif total_loaded > 1e-6 and ctx.total_waited_at_location < ctx.max_wait_product_h:
            # Retry loading
            found_retry_step = False
            for i in range(ctx.itinerary_idx - 1, -1, -1):
                s = ctx.chosen_itinerary[i]
                if s.get('kind') == 'load' and s.get('location', ctx.current_location) == ctx.current_location:
                    ctx.itinerary_idx = i
                    found_retry_step = True
                    break
                elif s.get('kind') in ('sail', 'start'):
                    break

            ctx.log_func(
                process="Move",
                event="Utilization Low - Retrying Load",
                location=ctx.current_location,
                equipment="Ship",
                product=None,
                qty=0,
                time=1.0,
                unmet_demand=0.0,
                from_store=None,
                to_store=None,
                route_id=ctx.current_route_id or ctx.route_group,
                vessel_id=ctx.vessel_id,
                ship_state=ctx.state.value,
                override_time_h=ctx.env.now
            )

            total_loaded = sum(ctx.cargo.values())
            ctx.planned_load_remaining = max(0.0, (ctx.n_holds * ctx.payload_per_hold) - total_loaded)

            if ctx.sim:
                yield ctx.sim.wait_for_step(7)
            else:
                yield ctx.env.timeout(1)
            ctx.total_waited_at_location += 1.0
        else:
            if total_loaded > 1e-6:
                ctx.log_func(
                    process="Move",
                    event="Wait Limit Reached - Proceeding with Partial Load",
                    location=ctx.current_location,
                    equipment="Ship",
                    product=None,
                    qty=0,
                    time=0.0,
                    unmet_demand=0.0,
                    from_store=None,
                    to_store=None,
                    route_id=ctx.current_route_id or ctx.route_group,
                    vessel_id=ctx.vessel_id,
                    ship_state=ctx.state.value,
                    override_time_h=ctx.env.now
                )

                if ctx.planned_load_remaining > 1e-6 and ctx.sim and hasattr(ctx.sim, 'pending_deliveries'):
                    primary_load_product = None
                    for step in ctx.chosen_itinerary:
                        if step.get('kind') == 'load':
                            primary_load_product = step.get('product')
                            break

                    if primary_load_product:
                        for step in ctx.chosen_itinerary:
                            if step.get('kind') == 'unload' and step.get('product') == primary_load_product:
                                sk = step.get('store_key')
                                if (ctx.vessel_id, sk) in ctx.sim.pending_deliveries:
                                    ctx.sim.pending_deliveries[(ctx.vessel_id, sk)] = max(0.0, ctx.sim.pending_deliveries[(ctx.vessel_id, sk)] - ctx.planned_load_remaining)

                if ctx.active_berth is not None:
                    ctx.active_berth.release(ctx.active_berth_req)
                    ctx.active_berth = None
                    ctx.active_berth_req = None

                ctx.state = ShipState.IN_TRANSIT
                ctx.log_state_change(ctx.state)
            else:
                # Reset to IDLE
                if ctx.active_berth is not None:
                    ctx.active_berth.release(ctx.active_berth_req)
                    ctx.active_berth = None
                    ctx.active_berth_req = None

                total_cargo = sum(ctx.cargo.values())
                if total_cargo > 1e-6:
                    ctx.log_func(
                        process="Move",
                        event="Cargo Lost at Sea",
                        location=ctx.current_location,
                        equipment="Ship",
                        product=str(list(ctx.cargo.keys())),
                        qty=total_cargo,
                        time=0.0,
                        unmet_demand=0.0,
                        from_store=None,
                        to_store=None,
                        route_id=ctx.current_route_id or ctx.route_group,
                        vessel_id=ctx.vessel_id,
                        ship_state=ctx.state.value,
                        override_time_h=ctx.env.now
                    )

                # Clear pending deliveries
                if ctx.sim and hasattr(ctx.sim, 'pending_deliveries'):
                    keys_to_del = [k for k in ctx.sim.pending_deliveries.keys() if k[0] == ctx.vessel_id]
                    for k in keys_to_del:
                        del ctx.sim.pending_deliveries[k]

                ctx.cargo = {}
                if ctx.t_state is not None:
                    ctx.t_state['cargo'] = ctx.cargo
                    ctx.t_state['route_id'] = ctx.route_group
                ctx.chosen_itinerary = None
                ctx.current_route_id = None
                ctx.state = ShipState.IDLE
                ctx.log_state_change(ctx.state)

                if ctx.sim:
                    yield ctx.sim.wait_for_step(7)
                else:
                    yield ctx.env.timeout(1)
                return 'continue'
    else:
        ctx.itinerary_idx += 1

    return None


def handle_in_transit_state(ctx: ShipContext) -> Generator[Any, Any, Optional[str]]:
    """
    Handle IN_TRANSIT state: sail between ports.

    Returns: 'continue' to restart main loop, None to proceed to next state.
    """
    if ctx.itinerary_idx >= len(ctx.chosen_itinerary):
        if ctx.active_berth is not None:
            ctx.active_berth.release(ctx.active_berth_req)
            ctx.active_berth = None
            ctx.active_berth_req = None

        # Return to origin
        if ctx.current_location != ctx.origin_location:
            nm = get_nm_distance(ctx.nm_distances, ctx.current_location, ctx.origin_location)
            if nm > 0:
                travel_hours = nm / max(ctx.speed_knots, 1.0)
                pilot_out = get_pilot_hours(ctx.berth_info, ctx.current_location, 'out')
                pilot_in = get_pilot_hours(ctx.berth_info, ctx.origin_location, 'in')
                total_time = travel_hours + pilot_out + pilot_in

                ctx.log_func(
                    process="Move",
                    event="Transit",
                    location=ctx.current_location,
                    equipment="Ship",
                    product=None,
                    time=round(total_time, 2),
                    from_store=None,
                    to_store=None,
                    route_id=ctx.current_route_id or ctx.route_group,
                    vessel_id=ctx.vessel_id,
                    ship_state=ctx.state.value,
                    override_time_h=ctx.env.now
                )
                yield ctx.env.timeout(total_time)

        ctx.current_location = ctx.origin_location
        ctx.state = ShipState.IDLE
        ctx.current_route_id = None
        if ctx.t_state is not None:
            ctx.t_state['route_id'] = ctx.route_group
        ctx.log_state_change(ctx.state)

        # Clear pending deliveries
        if ctx.sim and hasattr(ctx.sim, 'pending_deliveries'):
            keys_to_del = [k for k in ctx.sim.pending_deliveries.keys() if k[0] == ctx.vessel_id]
            for k in keys_to_del:
                del ctx.sim.pending_deliveries[k]

        if ctx.sim:
            yield ctx.sim.wait_for_step(7)
        else:
            yield ctx.env.timeout(1)
        return 'continue'

    step = ctx.chosen_itinerary[ctx.itinerary_idx]
    kind = step.get('kind')

    if kind == 'sail':
        from_loc = step.get('from', ctx.current_location)
        to_loc = step.get('to', step.get('location'))

        if to_loc:
            if from_loc == to_loc:
                ctx.current_location = to_loc
                ctx.itinerary_idx += 1
                return 'continue'

            # Check for future work
            has_future_work = False
            total_cargo = sum(ctx.cargo.values())

            for i in range(ctx.itinerary_idx, len(ctx.chosen_itinerary)):
                future_step = ctx.chosen_itinerary[i]
                f_kind = future_step.get('kind')
                if f_kind == 'load':
                    has_future_work = True
                    break
                if f_kind == 'unload':
                    f_prod = future_step.get('product')
                    if f_prod in ctx.cargo and ctx.cargo[f_prod] > 1e-6:
                        has_future_work = True
                        break

            if not has_future_work and to_loc != ctx.origin_location:
                found_return = False
                for i in range(ctx.itinerary_idx + 1, len(ctx.chosen_itinerary)):
                    if ctx.chosen_itinerary[i].get('kind') == 'sail' and \
                            ctx.chosen_itinerary[i].get('location') == ctx.origin_location:
                        ctx.itinerary_idx = i
                        found_return = True
                        break

                if found_return:
                    return 'continue'
                else:
                    ctx.itinerary_idx = len(ctx.chosen_itinerary)
                    return 'continue'

            nm = get_nm_distance(ctx.nm_distances, from_loc, to_loc)
            travel_hours = nm / max(ctx.speed_knots, 1.0)

            pilot_out = get_pilot_hours(ctx.berth_info, from_loc, 'out')
            pilot_in = get_pilot_hours(ctx.berth_info, to_loc, 'in')
            total_time = travel_hours + pilot_out + pilot_in

            if total_time > 0:
                ctx.log_func(
                    process="Move",
                    event="Transit",
                    location=from_loc,
                    equipment="Ship",
                    product=None,
                    time=round(total_time, 2),
                    from_store=None,
                    to_store=None,
                    route_id=ctx.current_route_id or ctx.route_group,
                    vessel_id=ctx.vessel_id,
                    ship_state=ctx.state.value,
                    override_time_h=ctx.env.now
                )
                yield ctx.env.timeout(total_time)

            ctx.current_location = to_loc

        ctx.itinerary_idx += 1

    elif kind == 'unload':
        unload_loc = step.get('location', ctx.current_location)
        if ctx.active_berth is not None and unload_loc == ctx.current_location:
            ctx.state = ShipState.UNLOADING
        else:
            if unload_loc != ctx.current_location:
                nm = get_nm_distance(ctx.nm_distances, ctx.current_location, unload_loc)
                if nm > 0:
                    travel_hours = nm / max(ctx.speed_knots, 1.0)
                    pilot_out = get_pilot_hours(ctx.berth_info, ctx.current_location, 'out')
                    pilot_in = get_pilot_hours(ctx.berth_info, unload_loc, 'in')
                    total_time = travel_hours + pilot_out + pilot_in

                    ctx.log_func(
                        process="Move",
                        event="Transit",
                        location=ctx.current_location,
                        equipment="Ship",
                        product=None,
                        time=round(total_time, 2),
                        from_store=None,
                        to_store=None,
                        route_id=ctx.current_route_id or ctx.route_group,
                        vessel_id=ctx.vessel_id,
                        ship_state=ctx.state.value,
                        override_time_h=ctx.env.now
                    )
                    yield ctx.env.timeout(total_time)
                    ctx.current_location = unload_loc

            ctx.state = ShipState.WAITING_FOR_BERTH
        ctx.log_state_change(ctx.state, ctx.current_location)
        return 'continue'

    elif kind == 'load':
        load_loc = step.get('location', ctx.current_location)
        if ctx.active_berth is not None and load_loc == ctx.current_location:
            ctx.state = ShipState.LOADING
        else:
            if load_loc != ctx.current_location:
                nm = get_nm_distance(ctx.nm_distances, ctx.current_location, load_loc)
                if nm > 0:
                    travel_hours = nm / max(ctx.speed_knots, 1.0)
                    pilot_out = get_pilot_hours(ctx.berth_info, ctx.current_location, 'out')
                    pilot_in = get_pilot_hours(ctx.berth_info, load_loc, 'in')
                    total_time = travel_hours + pilot_out + pilot_in

                    ctx.log_func(
                        process="Move",
                        event="Transit",
                        location=ctx.current_location,
                        equipment="Ship",
                        product=None,
                        time=round(total_time, 2),
                        from_store=None,
                        to_store=None,
                        route_id=ctx.current_route_id or ctx.route_group,
                        vessel_id=ctx.vessel_id,
                        ship_state=ctx.state.value,
                        override_time_h=ctx.env.now
                    )
                    yield ctx.env.timeout(total_time)
                    ctx.current_location = load_loc

            ctx.state = ShipState.LOADING
        ctx.log_state_change(ctx.state, ctx.current_location)
        return 'continue'

    else:
        ctx.itinerary_idx += 1

    return None


def handle_waiting_for_berth_state(ctx: ShipContext) -> Generator[Any, Any, Optional[str]]:
    """
    Handle WAITING_FOR_BERTH state: transition to UNLOADING.
    """
    ctx.state = ShipState.UNLOADING
    ctx.log_state_change(ctx.state)
    if False:
        yield
    return None


def handle_unloading_state(ctx: ShipContext) -> Generator[Any, Any, Optional[str]]:
    """
    Handle UNLOADING state: unload cargo to destination stores.

    Returns: 'continue' to restart main loop, None to proceed to next state.
    """
    if ctx.itinerary_idx >= len(ctx.chosen_itinerary):
        if ctx.active_berth is not None:
            ctx.active_berth.release(ctx.active_berth_req)
            ctx.active_berth = None
            ctx.active_berth_req = None

        # Return to origin
        if ctx.current_location != ctx.origin_location:
            nm = get_nm_distance(ctx.nm_distances, ctx.current_location, ctx.origin_location)
            if nm > 0:
                travel_hours = nm / max(ctx.speed_knots, 1.0)
                pilot_out = get_pilot_hours(ctx.berth_info, ctx.current_location, 'out')
                pilot_in = get_pilot_hours(ctx.berth_info, ctx.origin_location, 'in')
                total_time = travel_hours + pilot_out + pilot_in

                ctx.log_func(
                    process="Move",
                    event="Transit",
                    location=ctx.current_location,
                    equipment="Ship",
                    product=None,
                    time=round(total_time, 2),
                    from_store=None,
                    to_store=None,
                    route_id=ctx.current_route_id or ctx.route_group,
                    vessel_id=ctx.vessel_id,
                    ship_state=ctx.state.value,
                    override_time_h=ctx.env.now
                )
                yield ctx.env.timeout(total_time)

        ctx.current_location = ctx.origin_location
        ctx.state = ShipState.IDLE
        ctx.log_state_change(ctx.state)

        # Clear pending deliveries
        if ctx.sim and hasattr(ctx.sim, 'pending_deliveries'):
            keys_to_del = [k for k in ctx.sim.pending_deliveries.keys() if k[0] == ctx.vessel_id]
            for k in keys_to_del:
                del ctx.sim.pending_deliveries[k]

        if ctx.sim:
            yield ctx.sim.wait_for_step(7)
        else:
            yield ctx.env.timeout(1)
        return 'continue'

    step = ctx.chosen_itinerary[ctx.itinerary_idx]
    kind = step.get('kind')

    if kind == 'unload':
        store_key = step.get('store_key')
        product = step.get('product')
        unload_location = step.get('location', ctx.current_location)

        # Acquire berth if needed
        if ctx.active_berth is None:
            yield from _acquire_berth(ctx, unload_location, None)

        if store_key and store_key in ctx.stores and product in ctx.cargo:
            cont = ctx.stores[store_key]
            _, unload_rate = get_store_rates(ctx.store_rates, store_key, ctx.default_load_rate, ctx.default_unload_rate)

            carried = float(ctx.cargo.get(product, 0.0))

            # Wait for space
            wait_hours = 0
            max_wait = 48

            original_state = ctx.state
            while wait_hours < max_wait:
                room = float(cont.capacity - cont.level)

                demand_rate = ctx.demand_rates.get(store_key, 0.0)
                required_space = min(carried, ctx.payload_per_hold)

                can_proceed = False
                if room >= required_space - 1e-6:
                    can_proceed = True
                elif demand_rate > 1e-6:
                    if room + demand_rate * 1.0 >= required_space:
                        can_proceed = True

                if can_proceed:
                    break

                if ctx.state != ShipState.WAITING_FOR_SPACE:
                    ctx.state = ShipState.WAITING_FOR_SPACE
                    ctx.log_state_change(ctx.state, unload_location)

                ctx.log_func(
                    process="Move",
                    event="Waiting for Space",
                    location=unload_location,
                    equipment="Ship",
                    product=product,
                    time=1.0,
                    unmet_demand=0.0,
                    from_store=None,
                    to_store=store_key,
                    to_level=float(cont.level),
                    to_fill_pct=float(cont.level) / cont.capacity if cont.capacity > 0 else 0.0,
                    route_id=ctx.current_route_id or ctx.route_group,
                    vessel_id=ctx.vessel_id,
                    ship_state=ctx.state.value,
                    override_time_h=ctx.env.now
                )
                if ctx.sim:
                    yield ctx.sim.wait_for_step(7)
                else:
                    yield ctx.env.timeout(1.0)
                wait_hours += 1

            if ctx.state != original_state:
                ctx.state = original_state
                ctx.log_state_change(ctx.state, unload_location)

            # Calculate unload quantity
            room = float(cont.capacity - cont.level)
            num_holds_to_unload = int(min(carried, room + 1e-6) // ctx.payload_per_hold)
            qty_to_unload = float(num_holds_to_unload * ctx.payload_per_hold)

            # Handle remainder
            if qty_to_unload < 1e-6 and carried < ctx.payload_per_hold and carried > 1e-6:
                has_future_unload = False
                for i in range(ctx.itinerary_idx + 1, len(ctx.chosen_itinerary)):
                    f_step = ctx.chosen_itinerary[i]
                    if f_step.get('kind') == 'unload' and f_step.get('product') == product:
                        has_future_unload = True
                        break

                if not has_future_unload:
                    if room >= carried - 1e-6:
                        qty_to_unload = carried
                    elif wait_hours >= max_wait:
                        qty_to_unload = min(carried, room)

            if qty_to_unload > 1e-6:
                unload_time = qty_to_unload / max(unload_rate, 1.0)

                if ctx.sim:
                    yield ctx.sim.wait_for_step(7)

                start_hold_t = ctx.env.now
                yield cont.put(qty_to_unload)
                yield ctx.env.timeout(unload_time)

                ctx.log_func(
                    process="Move",
                    event="Unload",
                    location=unload_location,
                    equipment="Ship",
                    product=product,
                    qty=qty_to_unload,
                    time=round(unload_time, 2),
                    unmet_demand=0.0,
                    qty_in=qty_to_unload,
                    from_store=None,
                    from_level=None,
                    to_store=store_key,
                    to_level=float(cont.level),
                    to_fill_pct=float(cont.level) / cont.capacity if cont.capacity > 0 else 0.0,
                    route_id=ctx.current_route_id or ctx.route_group,
                    vessel_id=ctx.vessel_id,
                    ship_state=ctx.state.value,
                    override_time_h=start_hold_t
                )

                ctx.cargo[product] = carried - qty_to_unload

                # Update pending deliveries
                if ctx.sim and hasattr(ctx.sim, 'pending_deliveries'):
                    if (ctx.vessel_id, store_key) in ctx.sim.pending_deliveries:
                        ctx.sim.pending_deliveries[(ctx.vessel_id, store_key)] = max(0.0, ctx.sim.pending_deliveries[
                            (ctx.vessel_id, store_key)] - qty_to_unload)

                ctx.itinerary_idx += 1
            elif wait_hours >= max_wait:
                ctx.log_func(
                    process="Move",
                    event="Skip Unload - Wait Limit Reached",
                    location=unload_location,
                    equipment="Ship",
                    product=product,
                    qty=0,
                    time=0.0,
                    unmet_demand=0.0,
                    from_store=None,
                    to_store=None,
                    route_id=ctx.current_route_id or ctx.route_group,
                    vessel_id=ctx.vessel_id,
                    ship_state=ctx.state.value,
                    override_time_h=ctx.env.now
                )
                ctx.itinerary_idx += 1
            else:
                if ctx.sim:
                    yield ctx.sim.wait_for_step(7)
                else:
                    yield ctx.env.timeout(1.0)
        else:
            ctx.itinerary_idx += 1
    elif kind == 'unload':
        ctx.state = ShipState.UNLOADING
        ctx.log_state_change(ctx.state)
    elif kind == 'load':
        ctx.state = ShipState.LOADING
        ctx.log_state_change(ctx.state)
    else:
        if ctx.active_berth is not None:
            ctx.active_berth.release(ctx.active_berth_req)
            ctx.active_berth = None
            ctx.active_berth_req = None
        ctx.state = ShipState.IN_TRANSIT
        ctx.log_state_change(ctx.state)

    return None


def handle_error_state(ctx: ShipContext) -> Generator[Any, Any, Optional[str]]:
    """
    Handle ERROR state: log error and reset to IDLE.
    """
    ctx.log_func(
        process="Move",
        event="Error",
        location=ctx.current_location,
        equipment="Ship",
        product=None,
        qty=0,
        unmet_demand=0.0,
        from_store=None,
        from_level=None,
        to_store=None,
        to_level=None,
        route_id=ctx.current_route_id or ctx.route_group,
        vessel_id=ctx.vessel_id,
        ship_state=ctx.state.value,
        override_time_h=ctx.env.now
    )
    ctx.state = ShipState.IDLE
    ctx.cargo = {}
    if ctx.t_state is not None:
        ctx.t_state['cargo'] = ctx.cargo
    if ctx.sim:
        for _ in range(24):
            yield ctx.sim.wait_for_step(7)
    else:
        yield ctx.env.timeout(24)
    return None

