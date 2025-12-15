# sim_run_grok_core_make.py
from __future__ import annotations
from typing import List
import simpy

from sim_run_grok_core import MakeUnit, ProductionCandidate


def producer(sim, resource: simpy.Resource, unit: MakeUnit):
    """
    Manufacturing process: chooses among output candidates based on rule and
    produces in steps, optionally consuming input stores.
    Uses sim.env, sim.stores, sim.settings, sim.log.
    """
    while True:
        with resource.request() as req:
            yield req
            eligible: List[ProductionCandidate] = [
                c for c in unit.candidates
                if sim.stores[c.out_store_key].level + c.rate_tph * unit.step_hours <= sim.stores[c.out_store_key].capacity + 1e-6
            ]
            if not eligible:
                yield sim.env.timeout(unit.step_hours)
                continue

            # Choose candidate using sim's chooser
            cand = eligible[sim.choose_output(unit.choice_rule, eligible)]
            qty = cand.rate_tph * unit.step_hours

            if cand.in_store_key:
                needed = qty * cand.consumption_pct
                taken = min(sim.stores[cand.in_store_key].level, needed)
                yield sim.stores[cand.in_store_key].get(taken)
                sim.log("Consumed", location=unit.location, equipment=unit.equipment, qty=taken, product=cand.product)
                if needed > 0:
                    qty *= taken / needed

            yield sim.stores[cand.out_store_key].put(qty)
            sim.log("Produced", location=unit.location, equipment=unit.equipment, qty=qty, product=cand.product, to_store_key=cand.out_store_key)
            yield sim.env.timeout(unit.step_hours)
