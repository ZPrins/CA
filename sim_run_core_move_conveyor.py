# sim_run_core_move_conveyor.py
from __future__ import annotations
from typing import Callable, Dict, Optional
import simpy
from sim_run_types import TransportRoute

def transporter(env, route: TransportRoute,
                stores: Dict[str, simpy.Container],
                log_func: Callable,
                sim=None,
                t_state: Optional[dict] = None,
                vessel_id: int = 1):
    """
    Handles continuous movement via conveyor.
    Moves up to 'conveyor_speed' per hour, limited by source availability and target space.
    """
    route_id_str = f"{route.origin_location}->{route.dest_location}"
    
    while True:
        # 1. Sync with simulation step (Step 6 is for conveyor transfers)
        if sim:
            yield sim.wait_for_step(6)

        # 2. Get the speed from the route (load_rate_tph) or fallback to settings/1000
        speed_tph = float(route.load_rate_tph)
        if speed_tph <= 0:
            if sim and hasattr(sim, 'settings'):
                speed_tph = float(sim.settings.get('conveyor_speed', 1000.0))
            else:
                speed_tph = 1000.0

        # 3. Identify source and destination containers
        # We pick the best source (highest level) and best destination (highest available space)
        origin_key = None
        dest_key = None
        
        max_avail = -1.0
        for ok in route.origin_stores:
            if ok in stores:
                avail = float(stores[ok].level)
                if avail > max_avail:
                    max_avail = avail
                    origin_key = ok
        
        max_space = -1.0
        for dk in route.dest_stores:
            if dk in stores:
                space = float(stores[dk].capacity - stores[dk].level)
                if space > max_space:
                    max_space = space
                    dest_key = dk

        if not origin_key or not dest_key:
            yield env.timeout(1.0)
            continue

        origin_cont = stores.get(origin_key)
        dest_cont = stores.get(dest_key)

        if not origin_cont or not dest_cont:
            yield env.timeout(1.0)
            continue

        # 4. Calculate how much we CAN move this hour
        # Min of (Source Level, Destination Available Space, Conveyor Capacity/hr)
        avail_at_source = float(origin_cont.level)
        space_at_dest = float(dest_cont.capacity - dest_cont.level)
        
        qty_to_move = min(avail_at_source, space_at_dest, speed_tph)

        if qty_to_move > 0.01:
            # Execute the transfer
            yield origin_cont.get(qty_to_move)
            yield dest_cont.put(qty_to_move)

            # Log the movement
            log_func(
                process="Move",
                event="Transfer",
                location=route.origin_location,
                equipment="Conveyor",
                product=route.product,
                qty=qty_to_move,
                time=1.0,
                from_store=origin_key,
                from_level=origin_cont.level,
                to_store=dest_key,
                to_level=dest_cont.level,
                route_id=route_id_str,
                vessel_id=vessel_id
            )
        else:
            # Log Idle/Waiting state
            reason = "Waiting for Product" if avail_at_source <= 0.01 else "Waiting for Space"
            log_func(
                process="Move",
                event=reason,
                location=route.origin_location,
                equipment="Conveyor",
                product=route.product,
                qty=0,
                time=1.0,
                from_store=origin_key,
                to_store=dest_key,
                route_id=route_id_str,
                vessel_id=vessel_id
            )

        # 5. Advance simulation time by 1 hour (minus step overhead)
        yield env.timeout(1.0)
