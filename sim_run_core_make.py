# sim_run_core_make.py
from __future__ import annotations
from typing import List, Callable, Dict
import simpy

from sim_run_types import MakeUnit, ProductionCandidate


def producer(env, resource: simpy.Resource, unit: MakeUnit,
             stores: Dict[str, simpy.Container],
             log_func: Callable,
             choose_output_func: Callable[[str, List[ProductionCandidate]], int]):
    while True:
        with resource.request() as req:
            yield req
            # Check eligibility
            eligible: List[ProductionCandidate] = [
                c for c in unit.candidates
                if
                stores[c.out_store_key].level + c.rate_tph * unit.step_hours <= stores[c.out_store_key].capacity + 1e-6
            ]
            if not eligible:
                yield env.timeout(unit.step_hours)
                continue

            # Select Candidate
            cand = eligible[choose_output_func(unit.choice_rule, eligible)]
            qty = cand.rate_tph * unit.step_hours

            # 1. Consume (Input Side)
            from_store_key = None
            from_store_bal = None

            if cand.in_store_key:
                needed = qty * cand.consumption_pct
                # Check actual availability
                available = stores[cand.in_store_key].level
                taken = min(available, needed)

                yield stores[cand.in_store_key].get(taken)

                from_store_key = cand.in_store_key
                from_store_bal = stores[cand.in_store_key].level

                # Scale output if input was short (though logic usually prevents this)
                if needed > 0:
                    qty *= taken / needed

            # 2. Process Time
            yield env.timeout(unit.step_hours)

            # 3. Produce (Output Side)
            yield stores[cand.out_store_key].put(qty)
            to_store_bal = stores[cand.out_store_key].level

            # 4. Log Unified Event
            log_func(
                process="Make",
                event="Produce",
                location=unit.location,
                equipment=unit.equipment,
                product=cand.product,
                qty=qty,
                from_store=from_store_key,
                from_level=from_store_bal,
                to_store=cand.out_store_key,
                to_level=to_store_bal,
                route_id=None
            )