# sim_run_grok_core_make.py
from __future__ import annotations
from typing import List, Callable, Dict
import simpy

from sim_run_grok_core import MakeUnit, ProductionCandidate

def producer(env, resource: simpy.Resource, unit: MakeUnit,
             stores: Dict[str, simpy.Container],
             log_func: Callable,
             choose_output_func: Callable[[str, List[ProductionCandidate]], int]):
    """
    Manufacturing process.
    Refactored to depend on explicit dependencies, not 'sim'.
    """
    while True:
        with resource.request() as req:
            yield req
            eligible: List[ProductionCandidate] = [
                c for c in unit.candidates
                if stores[c.out_store_key].level + c.rate_tph * unit.step_hours <= stores[c.out_store_key].capacity + 1e-6
            ]
            if not eligible:
                yield env.timeout(unit.step_hours)
                continue

            # Choose candidate using the passed selector function
            cand = eligible[choose_output_func(unit.choice_rule, eligible)]
            qty = cand.rate_tph * unit.step_hours

            if cand.in_store_key:
                needed = qty * cand.consumption_pct
                taken = min(stores[cand.in_store_key].level, needed)
                yield stores[cand.in_store_key].get(taken)
                log_func("Consumed", location=unit.location, equipment=unit.equipment, qty=taken, product=cand.product)
                if needed > 0:
                    qty *= taken / needed

            yield stores[cand.out_store_key].put(qty)
            log_func("Produced", location=unit.location, equipment=unit.equipment, qty=qty, product=cand.product, to_store_key=cand.out_store_key)
            yield env.timeout(unit.step_hours)