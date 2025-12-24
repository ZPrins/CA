# sim_run_core_move_train.py
from __future__ import annotations
from typing import Callable, Dict, Optional
import math
import simpy

from sim_run_types import TransportRoute


def transporter(env, route: TransportRoute,
                stores: Dict[str, simpy.Container],
                log_func: Callable,
                require_full: bool = True,
                debug_full: bool = False,
                sim=None,
                t_state: Optional[dict] = None,
                vessel_id: int = 1):
    route_id_str = f"{route.origin_location}->{route.dest_location}"

    # Helper to log accumulated idle time
    idle_start_time = None
    idle_start_day = None
    idle_location = None
    idle_store = None
    idle_reason = None

    def log_idle(location, store=None, reason=None):
        nonlocal idle_start_time, idle_start_day, idle_location, idle_store, idle_reason
        
        # Log immediately instead of accumulating to keep time field sensible and hourly
        log_day = int(env.now / 24) % 365 + 1
        
        # Get current store level if a store is provided
        from_lvl = None
        from_fill = None
        to_lvl = None
        to_fill = None
        
        if store and store in stores:
            cont = stores[store]
            level = float(cont.level)
            fill = level / cont.capacity if cont.capacity > 0 else 0.0
            if location == route.origin_location:
                from_lvl = level
                from_fill = fill
            else:
                to_lvl = level
                to_fill = fill

        log_func(
            process="Move",
            event=reason if reason else "Idle",
            location=location,
            equipment=route.mode.capitalize() if route.mode else "Train",
            product=route.product,
            qty=0,
            time=1.0,
            unmet_demand=0.0,
            from_store=store if location == route.origin_location else None,
            from_level=from_lvl,
            from_fill_pct=from_fill,
            to_store=store if location == route.dest_location else None,
            to_level=to_lvl,
            to_fill_pct=to_fill,
            route_id=route_id_str,
            vessel_id=vessel_id,
            override_day=log_day,
            override_time_h=env.now
        )

    while True:
        # Before starting a new trip, sync to hour boundary if we finished a trip mid-hour
        # or if we are just starting. Actually, Step 2 will sync us.

        # 1. Validation
        if route.payload_t <= 0 or route.load_rate_tph <= 0 or route.unload_rate_tph <= 0:
            # Retrieve small wait timeout from settings if available, else 1.0
            wait_h = 1.0
            if sim and hasattr(sim, 'settings'):
                wait_h = float(sim.settings.get('transporter_wait_h', 1.0))
            yield env.timeout(wait_h)
            continue

        origin_cont = None
        origin_key = None
        dest_cont = None
        dest_key = None

        # Wait for Step 2: Reduce the Inventory by the "Train Load" Qty
        # Step 2 is now handled just before .get() to be closer to the action
        # and avoid blocking if no inventory is available yet.

        # 2. Select Source/Dest
        if require_full:
            # Require: FULL load available at origin AND FULL space available at destination
            for ok in route.origin_stores:
                oc = stores[ok]
                if oc.level + 1e-6 >= route.payload_t:
                    for dk in route.dest_stores:
                        dc = stores[dk]
                        free_space = (dc.capacity - dc.level)
                        if free_space + 1e-6 >= route.payload_t:
                            # Account for other pending deliveries to this destination
                            other_pending = 0.0
                            if sim and hasattr(sim, 'pending_deliveries'):
                                for (vid, sk), qty in sim.pending_deliveries.items():
                                    # vid < 1000 for ships, vid >= 1000 for trains (using a convention)
                                    if sk == dk:
                                        other_pending += qty
                            
                            if free_space - other_pending + 1e-6 >= route.payload_t:
                                origin_key, origin_cont = ok, oc
                                dest_key, dest_cont = dk, dc
                                break
                    if origin_cont is not None:
                        break

            if origin_cont is None or dest_cont is None:
                # Either no origin had enough stock, or no destination had enough space yet.
                reason = "Waiting for Product" if origin_cont is None else "Waiting for Space"
                # For Waiting for Product/Space, we use the candidate store key for logging if possible
                log_store = None
                if origin_cont is None and route.origin_stores:
                    log_store = route.origin_stores[0]
                elif dest_cont is None and route.dest_stores:
                    log_store = route.dest_stores[0]
                
                log_idle(route.origin_location, store=log_store, reason=reason)
                if sim:
                    yield sim.wait_for_step(7)
                else:
                    # Retrieve small wait timeout from settings if available, else 1.0
                    wait_h = 1.0
                    if sim and hasattr(sim, 'settings'):
                        wait_h = float(sim.settings.get('transporter_wait_h', 1.0))
                    yield env.timeout(wait_h)
                continue

            # Final re-check immediately before securing inventory to reduce race-condition issues
            if origin_cont.level + 1e-6 < route.payload_t or ((dest_cont.capacity - dest_cont.level) + 1e-6 < route.payload_t):
                reason = "Waiting for Product" if origin_cont.level + 1e-6 < route.payload_t else "Waiting for Space"
                log_idle(route.origin_location, store=origin_key if reason == "Waiting for Product" else dest_key, reason=reason)
                if sim:
                    yield sim.wait_for_step(7)
                else:
                    # Retrieve small wait timeout from settings if available, else 1.0
                    wait_h = 1.0
                    if sim and hasattr(sim, 'settings'):
                        wait_h = float(sim.settings.get('transporter_wait_h', 1.0))
                    yield env.timeout(wait_h)
                continue

            # 3. Register Pending Delivery
            if sim and hasattr(sim, 'pending_deliveries'):
                sim.pending_deliveries[(vessel_id, dest_key)] += route.payload_t

            take = float(route.payload_t)

        else:
            for ok in route.origin_stores:
                oc = stores[ok]
                if oc.level > 1e-6:
                    origin_key, origin_cont = ok, oc
                    break
            if origin_cont is None:
                log_idle(route.origin_location, store=route.origin_stores[0] if route.origin_stores else None, reason="Waiting for Product")
                if sim:
                    yield sim.wait_for_step(7)
                else:
                    # Retrieve small wait timeout from settings if available, else 1.0
                    wait_h = 1.0
                    if sim and hasattr(sim, 'settings'):
                        wait_h = float(sim.settings.get('transporter_wait_h', 1.0))
                    yield env.timeout(wait_h)
                continue

            dest_key = route.dest_stores[0]
            dest_cont = stores[dest_key]

            origin_stock = float(origin_cont.level)
            take = min(max(0.0, route.payload_t), origin_stock)

            if take <= 1e-6:
                log_idle(route.origin_location, store=origin_key, reason="Waiting for Product")
                if sim:
                    yield sim.wait_for_step(7)
                else:
                    # Retrieve small wait timeout from settings if available, else 1.0
                    wait_h = 1.0
                    if sim and hasattr(sim, 'settings'):
                        wait_h = float(sim.settings.get('transporter_wait_h', 1.0))
                    yield env.timeout(wait_h)
                continue

        # 3. LOAD (Secure & Log)
        # Wait for Step 2: Reduce the Inventory by the "Train Load" Qty
        if sim:
            yield sim.wait_for_step(2)

        # Non-blocking get with logging
        while True:
            cont = origin_cont
            if cont.level >= take - 1e-6:
                break
            else:
                log_idle(route.origin_location, origin_key, reason="Waiting for Product")
                if sim:
                    yield sim.wait_for_step(7)
                else:
                    # Retrieve small wait timeout from settings if available, else 1.0
                    wait_h = 1.0
                    if sim and hasattr(sim, 'settings'):
                        wait_h = float(sim.settings.get('transporter_wait_h', 1.0))
                    yield env.timeout(wait_h)
        
        load_h = take / max(route.load_rate_tph, 1e-6)
        
        start_load_t = env.now
        yield origin_cont.get(take)
        if load_h > 0:
            yield env.timeout(load_h)

        log_func(
            process="Move",
            event="Load",
            location=route.origin_location,
            equipment=route.mode.capitalize() if route.mode else "Train",
            product=route.product,
            qty=take,
            time=round(load_h, 2),
            unmet_demand=0.0,
            qty_out=take,
            from_store=origin_key,
            from_level=float(origin_cont.level),
            from_fill_pct=float(origin_cont.level) / origin_cont.capacity if origin_cont.capacity > 0 else 0.0,
            to_store=None,  # On Train
            to_level=None,
            route_id=route_id_str,
            vessel_id=vessel_id,
            override_time_h=start_load_t
        )

        if t_state is not None:
            t_state['cargo'][route.product] = t_state['cargo'].get(route.product, 0) + take

        # 4. TRAVEL
        travel_h = route.to_min / 60.0 if route.to_min > 0 else 0
        if travel_h > 0:
            log_func(
                process="Move",
                event="Transit",
                location=route.origin_location,
                equipment="Train",
                product=route.product,
                time=round(travel_h, 2),
                from_store=None,
                to_store=None,
                route_id=route_id_str,
                vessel_id=vessel_id,
                override_time_h=env.now
            )
            yield env.timeout(travel_h)

        # Wait for Step 6: Increase inventory by the "Train Offload" Qty
        if sim:
            yield sim.wait_for_step(6)

        # 5. UNLOAD (Put & Log)
        # Unload time is also busy time - let's log it hourly if it's > 1h
        unload_h = take / max(route.unload_rate_tph, 1e-6)
        
        # Non-blocking put with logging
        while True:
            cont = dest_cont
            if cont.capacity - cont.level >= take - 1e-6:
                break
            else:
                log_idle(route.dest_location, dest_key, reason="Waiting for Space")
                if sim:
                    yield sim.wait_for_step(7)
                else:
                    # Retrieve small wait timeout from settings if available, else 1.0
                    wait_h = 1.0
                    if sim and hasattr(sim, 'settings'):
                        wait_h = float(sim.settings.get('transporter_wait_h', 1.0))
                    yield env.timeout(wait_h)

        start_unload_t = env.now
        yield dest_cont.put(take)
        if unload_h > 0:
            yield env.timeout(unload_h)

        log_func(
            process="Move",
            event="Unload",
            location=route.dest_location,
            equipment=route.mode.capitalize() if route.mode else "Train",
            product=route.product,
            qty=take,
            time=round(unload_h, 2),
            unmet_demand=0.0,
            qty_in=take,
            from_store=None,  # Off Train
            from_level=None,
            to_store=dest_key,
            to_level=float(dest_cont.level),
            to_fill_pct=float(dest_cont.level) / dest_cont.capacity if dest_cont.capacity > 0 else 0.0,
            route_id=route_id_str,
            vessel_id=vessel_id,
            override_time_h=start_unload_t
        )

        if t_state is not None:
            t_state['cargo'][route.product] = max(0.0, t_state['cargo'].get(route.product, 0) - take)

        # 6. Unregister Pending Delivery
        if sim and hasattr(sim, 'pending_deliveries'):
            if (vessel_id, dest_key) in sim.pending_deliveries:
                sim.pending_deliveries[(vessel_id, dest_key)] = max(0.0, sim.pending_deliveries[(vessel_id, dest_key)] - take)
                if sim.pending_deliveries[(vessel_id, dest_key)] < 1e-6:
                    del sim.pending_deliveries[(vessel_id, dest_key)]

        # 7. RETURN
        return_h = route.back_min / 60.0 if route.back_min > 0 else 0
        if return_h > 0:
            log_func(
                process="Move",
                event="Return",
                location=route.dest_location,
                equipment="Train",
                product=route.product,
                time=round(return_h, 2),
                from_store=None,
                to_store=None,
                route_id=route_id_str,
                vessel_id=vessel_id,
                override_time_h=env.now
            )
            yield env.timeout(return_h)