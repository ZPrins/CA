# sim_run_grok_core_move_train.py
from __future__ import annotations
import simpy

from sim_run_grok_core import TransportRoute


def transporter(sim, route: TransportRoute):
    """
    TRAIN transporter process. Implements full/partial load policy based on
    sim.settings['require_full_payload'] and logs with mode="TRAIN".
    Uses sim.env, sim.stores, sim.log.
    """
    while True:
        # Skip invalid/degenerate routes
        if route.payload_t <= 0 or route.load_rate_tph <= 0 or route.unload_rate_tph <= 0:
            yield sim.env.timeout(1)
            continue

        require_full = bool(sim.settings.get("require_full_payload", True))
        debug_full = bool(sim.settings.get("debug_full_payload", False))

        # Choose origin and destination stores based on policy
        origin_cont = None
        origin_key = None
        dest_cont = None
        dest_key = None

        if require_full:
            # Require an origin with at least one full payload and a destination with room
            for ok in route.origin_stores:
                oc = sim.stores[ok]
                if oc.level + 1e-6 >= route.payload_t:
                    for dk in route.dest_stores:
                        dc = sim.stores[dk]
                        room = dc.capacity - dc.level
                        if room + 1e-6 >= route.payload_t:
                            origin_key, origin_cont = ok, oc
                            dest_key, dest_cont = dk, dc
                            break
                    if origin_cont is not None:
                        break
            if origin_cont is None or dest_cont is None:
                if debug_full:
                    try:
                        o_stock = max((float(sim.stores[k].level) for k in route.origin_stores), default=0.0)
                        d_room = max((float(sim.stores[k].capacity - sim.stores[k].level) for k in route.dest_stores), default=0.0)
                    except Exception:
                        o_stock = d_room = 0.0
                    sim.log(
                        "WaitFullPayload",
                        mode="TRAIN",
                        product=route.product,
                        origin_location=route.origin_location,
                        dest_location=route.dest_location,
                        payload_t=route.payload_t,
                        max_origin_stock=o_stock,
                        max_dest_room=d_room,
                    )
                yield sim.env.timeout(1)
                continue
            take = float(route.payload_t)
        else:
            # Partial-load policy: pick any stock/room and move up to payload
            for ok in route.origin_stores:
                oc = sim.stores[ok]
                if oc.level > 1e-6:
                    origin_key, origin_cont = ok, oc
                    break
            if origin_cont is None:
                yield sim.env.timeout(1)
                continue
            for dk in route.dest_stores:
                dc = sim.stores[dk]
                if (dc.capacity - dc.level) > 1e-6:
                    dest_key, dest_cont = dk, dc
                    break
            if dest_cont is None:
                yield sim.env.timeout(1)
                continue
            origin_stock = float(origin_cont.level)
            dest_room = float(dest_cont.capacity - dest_cont.level)
            take = min(max(0.0, route.payload_t), origin_stock, dest_room)
            if take <= 1e-6:
                yield sim.env.timeout(1)
                continue

        # Load
        load_time = take / max(route.load_rate_tph, 1e-6)
        yield sim.env.timeout(load_time)
        yield origin_cont.get(take)
        sim.log("Loaded", mode="TRAIN", product=route.product, from_store=origin_key, to_location=route.dest_location, qty=take)

        # Transit to destination
        yield sim.env.timeout(route.to_min / 60 if route.to_min > 0 else 0)

        # Unload in chunks capped by unload rate per hour
        remaining = take
        while remaining > 1e-6:
            eff_rate = max(route.unload_rate_tph, 1e-6)
            chunk = min(remaining, eff_rate)
            if chunk <= 1e-6:
                break
            unload_time = chunk / eff_rate
            yield sim.env.timeout(unload_time)
            yield dest_cont.put(chunk)
            remaining -= chunk
            sim.log("Unloaded", mode="TRAIN", product=route.product, to_store=dest_key, qty=chunk)

        # Return transit
        yield sim.env.timeout(route.back_min / 60 if route.back_min > 0 else 0)
