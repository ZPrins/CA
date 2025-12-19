# sim_run_core_make.py
from __future__ import annotations
from typing import List, Callable, Dict, Optional, Tuple, Set
import simpy
import random
import math

from sim_run_types import MakeUnit, ProductionCandidate


def _sample_breakdown_duration(mean_hours: float = 3.0) -> int:
    """Sample breakdown duration from lognormal distribution, minimum 1 hour."""
    sigma = 0.8
    mu = math.log(mean_hours) - (sigma ** 2) / 2
    duration = random.lognormvariate(mu, sigma)
    return max(1, round(duration))


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
    """Select output store with MOST empty space (allow partial if any positive space)."""
    best_key = None
    best_space = -1.0
    
    for key in store_keys:
        if key not in stores:
            continue
        store = stores[key]
        space = store.capacity - store.level
        # Allow selection if any positive space exists (partial allowed)
        if space > 1e-6 and space > best_space:
            best_space = space
            best_key = key
    
    if best_key:
        return (best_key, best_space)
    
    # Fallback: if no store has any space, pick the one with most space anyway (will log blocked)
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
    
    maintenance_days: Set[int] = set(unit.maintenance_days) if unit.maintenance_days else set()
    downtime_pct: float = unit.unplanned_downtime_pct or 0.0
    
    available_hours_seen: float = 0.0
    downtime_consumed: float = 0.0
    breakdown_remaining: int = 0
    mean_breakdown_duration: float = 3.0
    maintenance_window_start = None  # track start of continuous maintenance window
    maintenance_window_hours = 0  # accumulate hours for continuous window

    while True:
        # Calculate time markers ONCE at loop start, before any yields
        current_time = env.now
        log_time = current_time + unit.step_hours  # represent the hour being consumed
        current_day = int(current_time / 24) % 365 + 1
        log_day = int(log_time / 24) % 365 + 1
        next_day = current_day + 1

        if current_day in maintenance_days:
            # Log maintenance hour to maintain continuous time_h sequence
            log_func(
                process="Downtime",
                event="Maintenance",
                location=unit.location,
                equipment=unit.equipment,
                product=None,
                qty=None,
                from_store=None,
                from_level=None,
                to_store=None,
                to_level=None,
                route_id=None,
                override_day=log_day,
                override_time_h=log_time
            )
            # Track window for potential consolidation in reports
            if maintenance_window_start is None:
                maintenance_window_start = log_time
                maintenance_window_hours = unit.step_hours
            else:
                maintenance_window_hours += unit.step_hours

            yield env.timeout(unit.step_hours)
            continue
        else:
            # End of maintenance window - reset tracking (logging already done per-hour)
            if maintenance_window_start is not None:
                maintenance_window_start = None
                maintenance_window_hours = 0

        available_hours_seen += unit.step_hours
        
        if breakdown_remaining > 0:
            # Log breakdown continuation hour to maintain continuous time_h sequence
            log_func(
                process="Downtime",
                event="Breakdown",
                location=unit.location,
                equipment=unit.equipment,
                product=None,
                qty=None,
                from_store=None,
                from_level=None,
                to_store=None,
                to_level=None,
                route_id=None,
                override_day=log_day,
                override_time_h=log_time
            )
            downtime_consumed += unit.step_hours
            breakdown_remaining -= 1
            yield env.timeout(unit.step_hours)
            continue
        
        if downtime_pct > 0:
            target_downtime = available_hours_seen * downtime_pct
            backlog = target_downtime - downtime_consumed
            
            if backlog > 0:
                expected_duration = mean_breakdown_duration
                start_prob = min(backlog / expected_duration, 1.0) * 0.15
                
                if random.random() < start_prob:
                    duration = _sample_breakdown_duration(mean_breakdown_duration)
                    max_allowed = max(1, int(backlog + expected_duration))
                    planned = min(duration, max_allowed)

                    log_func(
                        process="Downtime",
                        event="BreakdownStart",
                        location=unit.location,
                        equipment=unit.equipment,
                        product=None,
                        qty=planned,
                        from_store=None,
                        from_level=None,
                        to_store=None,
                        to_level=None,
                        route_id=None,
                        override_day=log_day,
                        override_time_h=log_time
                    )
                    downtime_consumed += unit.step_hours
                    breakdown_remaining = max(0, planned - 1)
                    yield env.timeout(unit.step_hours)
                    continue
        
        with resource.request() as req:
            yield req
            
            qty_base = unit.candidates[0].rate_tph * unit.step_hours if unit.candidates else 0
            
            # Build list of eligible candidates with best input/output stores selected
            eligible: List[Tuple[ProductionCandidate, Optional[str], Optional[str], float, float]] = []

            for cand in unit.candidates:
                full_qty = cand.rate_tph * unit.step_hours
                needed = full_qty * cand.consumption_pct

                # Determine input store keys (use list if available, else single key)
                in_keys = cand.in_store_keys or ([cand.in_store_key] if cand.in_store_key else [])
                # Determine output store keys (use list if available, else single key)
                out_keys = cand.out_store_keys or ([cand.out_store_key] if cand.out_store_key else [])
                
                # Select best input store (highest inventory)
                selected_in = None
                in_level = 0.0
                if in_keys:
                    result = _select_best_input_store(stores, in_keys, needed)
                    if result:
                        selected_in, in_level = result[0], result[1]

                # Select best output store (most empty space). Allow partial if any space > 0
                selected_out = None
                out_space = 0.0
                if out_keys:
                    result = _select_best_output_store(stores, out_keys, full_qty)
                    if result:
                        selected_out, out_space = result[0], result[1]

                # Eligible if output has ANY space; input may be zero (will result in 0 qty)
                if selected_out is not None:
                    eligible.append((cand, selected_in, selected_out, in_level, out_space))

            if not eligible:
                # Log idle hour - equipment waiting for output space
                log_func(
                    process="Make",
                    event="Idle",
                    location=unit.location,
                    equipment=unit.equipment,
                    product=None,
                    qty=0,
                    from_store=None,
                    from_level=None,
                    to_store=None,
                    to_level=None,
                    route_id=None,
                    override_day=log_day,
                    override_time_h=log_time
                )
                yield env.timeout(unit.step_hours)
                continue

            # Select best candidate:
            # 1. Prefer candidates with positive output space over those with zero space
            # 2. Among those with positive space, pick the one with MOST space
            # 3. If all have zero space, pick the one with most space (least negative)
            #
            # This ensures multi-product equipment (e.g., VRM with CL→GP and GBFS→SG paths)
            # switches to an alternative product when the primary output is full

            candidates_with_space = [(i, c, in_k, out_k, in_l, out_s)
                                     for i, (c, in_k, out_k, in_l, out_s) in enumerate(eligible)
                                     if out_s > 1e-6]

            if candidates_with_space:
                # Pick from candidates that have positive output space
                best_idx, _, _, _, _, best_space = max(candidates_with_space, key=lambda x: x[5])
            else:
                # All candidates have zero or negative space - pick least-full for ProduceBlocked logging
                best_idx = 0
                best_space = -1.0
                for i, (c, in_key, out_key, _in_lvl, out_sp) in enumerate(eligible):
                    if out_sp > best_space:
                        best_space = out_sp
                        best_idx = i

            cand, from_store_key, to_store_key, in_level, out_space_planned = eligible[best_idx]
            full_qty = cand.rate_tph * unit.step_hours
            qty = full_qty

            # Use log_day that was calculated at loop start (not env.now which has advanced)
            production_start_day = log_day

            # 1. Consume (Input Side) - from SINGLE selected store (may be None)
            from_store_bal = None
            partial_reason_input = False
            taken = 0.0
            needed = qty * cand.consumption_pct
            if from_store_key:
                available = stores[from_store_key].level
                taken = min(available, needed)
                if taken > 0:
                    yield stores[from_store_key].get(taken)
                from_store_bal = stores[from_store_key].level if from_store_key else None
                # Scale output if input was short
                if needed > 1e-9 and taken < needed:
                    qty *= (taken / needed)
                    partial_reason_input = True
            else:
                # No input required
                taken = 0.0
                needed = 0.0

            # 2. Process Time
            yield env.timeout(unit.step_hours)

            # 3. Determine available output space at end of hour
            to_store_bal_before = stores[to_store_key].level
            cap = stores[to_store_key].capacity
            space_now = max(0.0, cap - to_store_bal_before)
            allowed = min(qty, space_now)
            partial_reason_output = allowed + 1e-9 < qty

            # 4. Put allowed quantity (if any)
            if allowed > 0:
                yield stores[to_store_key].put(allowed)
            to_store_bal = stores[to_store_key].level

            # 5. Decide event type and qty to log
            if allowed <= 1e-9:
                # Blocked by output capacity (no space at end of hour)
                log_func(
                    process="Make",
                    event="ProduceBlocked",
                    location=unit.location,
                    equipment=unit.equipment,
                    product=cand.product,
                    qty=0.0,
                    from_store=from_store_key,
                    from_level=from_store_bal,
                    to_store=to_store_key,
                    to_level=to_store_bal,
                    route_id=None,
                    override_day=production_start_day,
                    override_time_h=log_time
                )
            else:
                ev = "Produce"
                if partial_reason_input or partial_reason_output or abs(allowed - full_qty) > 1e-6:
                    ev = "ProducePartial"
                log_func(
                    process="Make",
                    event=ev,
                    location=unit.location,
                    equipment=unit.equipment,
                    product=cand.product,
                    qty=allowed,
                    from_store=from_store_key,
                    from_level=from_store_bal,
                    to_store=to_store_key,
                    to_level=to_store_bal,
                    route_id=None,
                    override_day=production_start_day,
                    override_time_h=log_time
                )
