# sim_run_grok_core_deliver.py
from __future__ import annotations

from sim_run_grok_core import Demand


def consumer(sim, demand: Demand):
    """
    Deliver (TRUCK) process that services demand in truck-sized chunks per step.
    Uses sim.env, sim.stores, sim.settings, sim.log and updates sim.unmet.
    """
    store = sim.stores[demand.store_key]
    truck_load = float(sim.settings.get("demand_truck_load_tons", 25.0) or 25.0)
    step_h = float(sim.settings.get("demand_step_hours", 1.0) or 1.0)
    while True:
        per_step = float(demand.rate_per_hour) * step_h
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
                sim.log("Delivered", mode="TRUCK", store_key=demand.store_key, qty_t=round(take_amt, 2))
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
                sim.log("Delivered", mode="TRUCK", store_key=demand.store_key, qty_t=round(take_amt, 2))
            short = round(qty_req - take_amt, 2)
            if short > 0:
                unmet_total += short
        unmet_total = round(unmet_total, 2)
        if unmet_total > 0:
            sim.unmet[demand.store_key] = round(sim.unmet.get(demand.store_key, 0.0) + unmet_total, 2)
            sim.log("Unmet_Demand", store_key=demand.store_key, unmet=unmet_total)
        yield sim.env.timeout(step_h)
