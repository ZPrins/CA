# sim_run_core_move_train.py
from __future__ import annotations
from typing import Callable, Dict
import math
import simpy

from sim_run_types import TransportRoute


def transporter(env, route: TransportRoute,
                stores: Dict[str, simpy.Container],
                log_func: Callable,
                require_full: bool = True,
                debug_full: bool = False,
                sim=None):
    route_id_str = f"{route.origin_location}->{route.dest_location}"

    while True:
        # Before starting a new trip, sync to hour boundary if we finished a trip mid-hour
        # or if we are just starting. Actually, Step 2 will sync us.

        # 1. Validation
        if route.payload_t <= 0 or route.load_rate_tph <= 0 or route.unload_rate_tph <= 0:
            yield env.timeout(1)
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
                            origin_key, origin_cont = ok, oc
                            dest_key, dest_cont = dk, dc
                            break
                    if origin_cont is not None:
                        break

            if origin_cont is None or dest_cont is None:
                # Either no origin had enough stock, or no destination had enough space yet.
                log_func(
                    process="Move",
                    event="Idle",
                    location=route.origin_location,
                    equipment="Train",
                    product=route.product,
                    qty=0,
                    from_store=None,
                    to_store=None,
                    route_id=route_id_str
                )
                if sim:
                    yield sim.wait_for_step(7)
                else:
                    yield env.timeout(1)
                continue

            # Final re-check immediately before securing inventory to reduce race-condition issues
            if origin_cont.level + 1e-6 < route.payload_t or ((dest_cont.capacity - dest_cont.level) + 1e-6 < route.payload_t):
                log_func(
                    process="Move",
                    event="Idle",
                    location=route.origin_location,
                    equipment="Train",
                    product=route.product,
                    qty=0,
                    from_store=None,
                    to_store=None,
                    route_id=route_id_str
                )
                if sim:
                    yield sim.wait_for_step(7)
                else:
                    yield env.timeout(1)
                continue

            take = float(route.payload_t)

        else:
            for ok in route.origin_stores:
                oc = stores[ok]
                if oc.level > 1e-6:
                    origin_key, origin_cont = ok, oc
                    break
            if origin_cont is None:
                log_func(
                    process="Move",
                    event="Idle",
                    location=route.origin_location,
                    equipment="Train",
                    product=route.product,
                    qty=0,
                    from_store=None,
                    to_store=None,
                    route_id=route_id_str
                )
                if sim:
                    yield sim.wait_for_step(7)
                else:
                    yield env.timeout(1)
                continue

            dest_key = route.dest_stores[0]
            dest_cont = stores[dest_key]

            origin_stock = float(origin_cont.level)
            take = min(max(0.0, route.payload_t), origin_stock)

            if take <= 1e-6:
                log_func(
                    process="Move",
                    event="Idle",
                    location=route.origin_location,
                    equipment="Train",
                    product=route.product,
                    qty=0,
                    from_store=None,
                    to_store=None,
                    route_id=route_id_str
                )
                if sim:
                    yield sim.wait_for_step(7)
                else:
                    yield env.timeout(1)
                continue

        # 3. LOAD (Secure & Log)
        # Wait for Step 2: Reduce the Inventory by the "Train Load" Qty
        if sim:
            yield sim.wait_for_step(2)

        yield origin_cont.get(take)
        origin_bal = float(origin_cont.level)  # Snapshot level after take

        # Log LOAD immediately after getting from store, before the timeout
        log_func(
            process="Move",
            event="Load",
            location=route.origin_location,
            equipment="Train",
            product=route.product,
            qty=take,
            from_store=origin_key,
            from_level=origin_bal,
            to_store=None,  # On Train
            to_level=None,
            route_id=route_id_str
        )

        yield env.timeout(math.ceil(take / max(route.load_rate_tph, 1e-6)))

        # 4. TRAVEL
        yield env.timeout(math.ceil(route.to_min / 60.0) if route.to_min > 0 else 0)

        # Wait for Step 6: Increase inventory by the "Train Offload" Qty
        if sim:
            yield sim.wait_for_step(6)

        # 5. UNLOAD (Put & Log)
        yield env.timeout(math.ceil(take / max(route.unload_rate_tph, 1e-6)))

        yield dest_cont.put(take)
        dest_bal = dest_cont.level  # Snapshot level after put

        log_func(
            process="Move",
            event="Unload",
            location=route.dest_location,
            equipment="Train",
            product=route.product,
            qty=take,
            from_store=None,  # Off Train
            from_level=None,
            to_store=dest_key,
            to_level=dest_bal,
            route_id=route_id_str
        )

        # 6. RETURN
        yield env.timeout(math.ceil(route.back_min / 60.0) if route.back_min > 0 else 0)