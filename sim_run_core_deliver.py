# sim_run_core_deliver.py
from __future__ import annotations
from typing import Callable, Dict


def consumer(env, stores, demand_key: str, demand_rate: float,
             truck_load: float, step_h: float,
             log_func: Callable, unmet_dict: Dict[str, float], sim=None):
    # Ensure stores is a list
    if not isinstance(stores, list):
        stores = [stores]

    while True:
        # Wait for Step 1: Reduce the Inventory by the "Deliver" Qty
        if sim:
            yield sim.wait_for_step(1)
        
        per_step = float(demand_rate) * step_h
        remaining_need = round(per_step, 2)
        unmet_total = 0.0

        # Helper to extract product/loc from key
        try:
            parts = demand_key.split('|')
            prod = parts[0]
            loc = parts[1]
        except:
            prod = "Unknown"
            loc = "Unknown"

        n_full = int(remaining_need // truck_load) if truck_load > 0 else 0

        # Batch delivery logic
        for _ in range(max(0, n_full)):
            take = 0.0
            chosen_store = None
            
            # Try to take from stores in order
            for store in stores:
                if float(store.level) > 0:
                    take = min(float(store.level), truck_load)
                    chosen_store = store
                    break
            
            # If no stock, chosen_store is None, take is 0
            if not chosen_store and stores:
                chosen_store = stores[0] # Default to first store for logging unmet

            unmet_this_truck = max(0.0, truck_load - take) if take < truck_load else 0.0
            
            if take > 0 or unmet_this_truck > 0:
                if take > 0:
                    yield chosen_store.get(take)
                
                level_after = chosen_store.level
                log_func(
                    process="Deliver",
                    event="Demand",
                    location=loc,
                    equipment="Truck",
                    product=prod,
                    qty=take,
                    time=0.0,
                    unmet_demand=unmet_this_truck,
                    qty_out=take,
                    from_store=getattr(chosen_store, 'store_key', demand_key),
                    from_level=level_after,
                    from_fill_pct=float(level_after) / chosen_store.capacity if chosen_store.capacity > 0 else 0.0,
                    to_store=None,
                    to_level=None,
                    route_id=None,
                    override_time_h=env.now
                )
                if unmet_this_truck > 0:
                    unmet_total += unmet_this_truck

            remaining_need -= truck_load
            if remaining_need <= 0: break

        # Remainder
        remainder = round(max(0.0, remaining_need), 2)
        if remainder > 0:
            take = 0.0
            chosen_store = None
            for store in stores:
                if float(store.level) > 0:
                    take = min(float(store.level), remainder)
                    chosen_store = store
                    break
            
            if not chosen_store and stores:
                chosen_store = stores[0]

            unmet_remainder = max(0.0, remainder - take)
            if take > 0 or unmet_remainder > 0:
                if take > 0:
                    yield chosen_store.get(take)
                
                level_after = chosen_store.level
                log_func(
                    process="Deliver",
                    event="Demand",
                    location=loc,
                    equipment="Truck",
                    product=prod,
                    qty=take,
                    time=0.0,
                    unmet_demand=unmet_remainder,
                    qty_out=take,
                    from_store=getattr(chosen_store, 'store_key', demand_key),
                    from_level=level_after,
                    from_fill_pct=float(level_after) / chosen_store.capacity if chosen_store.capacity > 0 else 0.0,
                    to_store=None,
                    to_level=None,
                    route_id=None,
                    override_time_h=env.now
                )
                if unmet_remainder > 0:
                    unmet_total += unmet_remainder

        if unmet_total > 0:
            unmet_dict[demand_key] = round(unmet_dict.get(demand_key, 0.0) + unmet_total, 2)

        if not sim:
            yield env.timeout(step_h)