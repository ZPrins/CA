# sim_run_grok_core_deliver.py
from __future__ import annotations
from typing import Callable, Dict


def consumer(env, store, demand_key: str, demand_rate: float,
             truck_load: float, step_h: float,
             log_func: Callable, unmet_dict: Dict[str, float]):
    """
    Deliver (TRUCK) process.
    Refactored to be pure: depends only on explicit arguments, not the global 'sim' object.
    """
    while True:
        per_step = float(demand_rate) * step_h
        remaining_need = round(per_step, 2)
        unmet_total = 0.0

        # Full truckloads
        n_full = int(remaining_need // truck_load) if truck_load > 0 else 0
        for _ in range(max(0, n_full)):
            qty_req = truck_load
            qty_avail = float(store.level)
            take_amt = min(qty_avail, qty_req)
            take_amt = round(take_amt, 2)
            if take_amt > 0:
                yield store.get(take_amt)
                log_func("Delivered", mode="TRUCK", store_key=demand_key, qty_t=round(take_amt, 2))

            short = round(qty_req - take_amt, 2)
            if short > 0:
                unmet_total += short

            remaining_need = round(remaining_need - qty_req, 2)
            if remaining_need <= 0:
                break

        # Remainder (partial truck)
        remainder = round(max(0.0, remaining_need), 2)
        if remainder > 0:
            qty_req = remainder
            qty_avail = float(store.level)
            take_amt = min(qty_avail, qty_req)
            take_amt = round(take_amt, 2)
            if take_amt > 0:
                yield store.get(take_amt)
                log_func("Delivered", mode="TRUCK", store_key=demand_key, qty_t=round(take_amt, 2))

            short = round(qty_req - take_amt, 2)
            if short > 0:
                unmet_total += short

        unmet_total = round(unmet_total, 2)
        if unmet_total > 0:
            unmet_dict[demand_key] = round(unmet_dict.get(demand_key, 0.0) + unmet_total, 2)
            log_func("Unmet_Demand", store_key=demand_key, unmet=unmet_total)

        yield env.timeout(step_h)