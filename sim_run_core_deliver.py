# sim_run_core_deliver.py
from __future__ import annotations
from typing import Callable, Dict


def consumer(env, store, demand_key: str, demand_rate: float,
             truck_load: float, step_h: float,
             log_func: Callable, unmet_dict: Dict[str, float]):
    while True:
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
            take = min(float(store.level), truck_load)
            if take > 0:
                yield store.get(take)
                log_func(
                    process="Deliver",
                    event="Demand",
                    location=loc,
                    equipment="Truck",
                    product=prod,
                    qty=take,
                    from_store=demand_key,
                    from_level=store.level,
                    to_store=None,
                    to_level=None,
                    route_id=None
                )

            remaining_need -= truck_load
            if remaining_need <= 0: break

        # Remainder
        remainder = round(max(0.0, remaining_need), 2)
        if remainder > 0:
            take = min(float(store.level), remainder)
            if take > 0:
                yield store.get(take)
                log_func(
                    process="Deliver",
                    event="Demand",
                    location=loc,
                    equipment="Truck",
                    product=prod,
                    qty=take,
                    from_store=demand_key,
                    from_level=store.level,
                    to_store=None,
                    to_level=None,
                    route_id=None
                )

            # Track unmet
            if take < remainder:
                unmet_total += (remainder - take)

        if unmet_total > 0:
            unmet_dict[demand_key] = round(unmet_dict.get(demand_key, 0.0) + unmet_total, 2)

        yield env.timeout(step_h)