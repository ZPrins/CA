# sim_run_core_make.py
from __future__ import annotations
from typing import List, Callable, Dict, Optional, Tuple
import simpy

from sim_run_types import MakeUnit, ProductionCandidate


def _select_best_input_store(stores: Dict[str, simpy.Container], 
                              store_keys: List[str], 
                              needed: float) -> Optional[Tuple[str, float]]:
    """Select input store with HIGHEST available inventory that has enough material."""
    best_key = None
    best_level = -1.0
    
    for key in store_keys:
        if key not in stores:
            continue
        level = stores[key].level
        if level >= needed - 1e-6 and level > best_level:
            best_level = level
            best_key = key
    
    if best_key:
        return (best_key, best_level)
    
    # Fallback: if no store has enough, pick the one with most inventory anyway
    for key in store_keys:
        if key not in stores:
            continue
        level = stores[key].level
        if level > best_level:
            best_level = level
            best_key = key
    
    return (best_key, best_level) if best_key else None


def _select_best_output_store(stores: Dict[str, simpy.Container], 
                               store_keys: List[str], 
                               qty: float) -> Optional[Tuple[str, float]]:
    """Select output store with MOST empty space that can fit the output."""
    best_key = None
    best_space = -1.0
    
    for key in store_keys:
        if key not in stores:
            continue
        store = stores[key]
        space = store.capacity - store.level
        if space >= qty - 1e-6 and space > best_space:
            best_space = space
            best_key = key
    
    if best_key:
        return (best_key, best_space)
    
    # Fallback: if no store has enough space, pick the one with most space anyway
    for key in store_keys:
        if key not in stores:
            continue
        store = stores[key]
        space = store.capacity - store.level
        if space > best_space:
            best_space = space
            best_key = key
    
    return (best_key, best_space) if best_key else None


def producer(env, resource: simpy.Resource, unit: MakeUnit,
             stores: Dict[str, simpy.Container],
             log_func: Callable,
             choose_output_func: Callable[[str, List[ProductionCandidate]], int]):
    while True:
        with resource.request() as req:
            yield req
            
            qty_base = unit.candidates[0].rate_tph * unit.step_hours if unit.candidates else 0
            
            # Build list of eligible candidates with best input/output stores selected
            eligible: List[Tuple[ProductionCandidate, str, str]] = []
            
            for cand in unit.candidates:
                qty = cand.rate_tph * unit.step_hours
                needed = qty * cand.consumption_pct
                
                # Determine input store keys (use list if available, else single key)
                in_keys = cand.in_store_keys or ([cand.in_store_key] if cand.in_store_key else [])
                # Determine output store keys (use list if available, else single key)
                out_keys = cand.out_store_keys or ([cand.out_store_key] if cand.out_store_key else [])
                
                # Select best input store (highest inventory)
                selected_in = None
                if in_keys:
                    result = _select_best_input_store(stores, in_keys, needed)
                    if result:
                        selected_in = result[0]
                
                # Select best output store (most empty space)
                selected_out = None
                if out_keys:
                    result = _select_best_output_store(stores, out_keys, qty)
                    if result and result[1] >= qty - 1e-6:  # Must have enough space
                        selected_out = result[0]
                
                # Check eligibility: output store must have space
                if selected_out:
                    # Check input availability if required
                    if in_keys:
                        # Accept if input store has ANY inventory (partial runs allowed - scaling handles it)
                        if selected_in and stores[selected_in].level > 1e-6:
                            eligible.append((cand, selected_in, selected_out))
                    else:
                        # No input required
                        eligible.append((cand, None, selected_out))
            
            if not eligible:
                yield env.timeout(unit.step_hours)
                continue

            # Select best candidate based on output store with most empty space
            best_idx = 0
            best_space = -1.0
            for i, (c, in_key, out_key) in enumerate(eligible):
                store = stores[out_key]
                space = store.capacity - store.level
                if space > best_space:
                    best_space = space
                    best_idx = i
            
            cand, from_store_key, to_store_key = eligible[best_idx]
            qty = cand.rate_tph * unit.step_hours

            # 1. Consume (Input Side) - from SINGLE selected store
            from_store_bal = None
            if from_store_key:
                needed = qty * cand.consumption_pct
                available = stores[from_store_key].level
                taken = min(available, needed)
                
                yield stores[from_store_key].get(taken)
                from_store_bal = stores[from_store_key].level
                
                # Scale output if input was short
                if needed > 0:
                    qty *= taken / needed

            # 2. Process Time
            yield env.timeout(unit.step_hours)

            # 3. Produce (Output Side) - to SINGLE selected store
            yield stores[to_store_key].put(qty)
            to_store_bal = stores[to_store_key].level

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
                to_store=to_store_key,
                to_level=to_store_bal,
                route_id=None
            )