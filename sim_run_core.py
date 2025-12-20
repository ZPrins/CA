# sim_run_core.py
from __future__ import annotations
from typing import Dict, List
import simpy
import os
from collections import defaultdict

from sim_run_types import StoreConfig, ProductionCandidate, MakeUnit, TransportRoute, Demand

# Check if quiet mode is enabled (for multi-run simulations)
QUIET_MODE = os.environ.get('SIM_QUIET_MODE', 'false').lower() == 'true'

def _log_progress(msg):
    """Print progress message only if not in quiet mode."""
    if not QUIET_MODE:
        print(msg, flush=True)

# Logic Modules
from sim_run_core_store import build_stores as _build_stores
from sim_run_core_make import producer as _producer
from sim_run_core_move_train import transporter as _transporter_train
from sim_run_core_move_ship import transporter as _transporter_ship
from sim_run_core_deliver import consumer as _consumer


class SupplyChainSimulation:
    def __init__(self, settings: dict):
        self.settings = settings
        self.env = simpy.Environment()
        self.stores: Dict[str, simpy.Container] = {}
        self.unmet: Dict[str, float] = {}
        self.action_log: List[dict] = []
        self.inventory_snapshots: List[dict] = []
        self.demand_rate_map: Dict[str, float] = {}
        self.port_berths: Dict[str, simpy.Resource] = {}
        self.pending_maintenance_windows: List[dict] = []  # for flushing at end
        
        # 7-step sequence events per hour
        self.step_events: Dict[int, simpy.Event] = {i: self.env.event() for i in range(1, 8)}

    def wait_for_step(self, step_num: int):
        """Wait for a specific step in the current hour's sequence."""
        return self.step_events[step_num]

    def master_clock(self):
        """Orchestrates the 7 steps per hour."""
        while True:
            # At the start of each hour, trigger steps 1-7 in sequence
            for i in range(1, 8):
                # Trigger the current step event
                self.step_events[i].succeed()
                # Replace it IMMEDIATELY with a new untriggered event for the next hour
                # This ensures any process that catches this step and loops back to wait
                # for the same step will wait for the NEXT hour, preventing infinite loops.
                self.step_events[i] = self.env.event()
                
                # We yield a tiny timeout to let all processes waiting on this event to run
                # Using a very small timeout > 0 ensures they actually execute in order
                yield self.env.timeout(1e-9)
            
            # After all 7 steps are triggered, wait for the next hour
            # We also advance the internal time by the processing step (default 1.0)
            # This makes the steps effectively sub-hour markers.
            yield self.env.timeout(1.0 - (7 * 1e-9))

    def log(self, event: str, **details):
        # If override_day is provided, use it only for the day bucket; keep true hourly time_h
        override_day = details.get('override_day', None)
        override_time_h = details.get('override_time_h', None)
        time_h = int(override_time_h) if override_time_h is not None else int(self.env.now)
        time_d = (override_day - 1) if override_day is not None else int(time_h // 24)
        day = time_d + 1
        
        # Log as a tuple with fixed structure for performance
        # order: day, time_h, time_d, process, event, location, equipment, product, qty, time, qty_in, 
        #        from_store, from_level, to_store, to_level, route_id, vessel_id, ship_state
        self.action_log.append((
            day,
            time_h,
            time_d,
            details.get('process'),
            event,
            details.get('location'),
            details.get('equipment'),
            details.get('product'),
            details.get('qty'),
            details.get('time'),
            details.get('qty_in'),
            details.get('from_store'),
            details.get('from_level'),
            details.get('to_store'),
            details.get('to_level'),
            details.get('route_id'),
            details.get('vessel_id'),
            details.get('ship_state')
        ))

    def snapshot(self):
        now = int(round(self.env.now))
        day = int(now // 24)
        for key, cont in self.stores.items():
            parts = key.split("|")
            pc = parts[0] if len(parts) > 0 else ""
            loc = parts[1] if len(parts) > 1 else ""
            eq = parts[2] if len(parts) > 2 else ""
            inp = parts[3] if len(parts) > 3 else ""
            fill = cont.level / cont.capacity if cont.capacity > 0 else 1.0
            
            # Snapshots as tuples: day, time_h, product_class, location, equipment, input, store_key, level, capacity, fill_pct
            self.inventory_snapshots.append((
                day, now, pc, loc, eq, inp, key, cont.level, cont.capacity, fill
            ))

    def choose_output(self, rule: str, candidates: List[ProductionCandidate]):
        best_idx = 0
        best = float('inf')
        for i, cand in enumerate(candidates):
            store = self.stores[cand.out_store_key]
            val = store.level / store.capacity if rule == "min_fill_pct" and store.capacity > 0 else store.level
            if val < best:
                best = val
                best_idx = i
        return best_idx

    def producer(self, resource: simpy.Resource, unit: MakeUnit):
        yield from _producer(
            self.env,
            resource,
            unit,
            self.stores,
            self.log,
            self.choose_output,
            sim=self
        )

    def transporter(self, route: TransportRoute, vessel_id: int = 1):
        mode = (getattr(route, 'mode', 'TRAIN') or 'TRAIN').upper()
        if mode == 'SHIP':
            yield from _transporter_ship(
                self.env,
                route,
                self.stores,
                self.port_berths,
                self.log,
                store_rates=self.settings.get('store_rates', {}),
                require_full=self.settings.get("require_full_payload", True),
                demand_rates=self.demand_rate_map,
                vessel_id=vessel_id,
                sole_supplier_stores=getattr(self, 'sole_supplier_stores', None),
                production_rates=getattr(self, 'production_rate_map', None),
                store_capacities=getattr(self, 'store_capacity_map', None),
                sim=self
            )
        else:  # TRAIN
            yield from _transporter_train(
                self.env,
                route,
                self.stores,
                self.log,
                require_full=self.settings.get("require_full_payload", True),
                debug_full=self.settings.get("debug_full_payload", False),
                sim=self
            )

    def consumer(self, demand: Demand):
        truck_load = float(self.settings.get("demand_truck_load_tons", 25.0) or 25.0)
        step_h = float(self.settings.get("demand_step_hours", 1.0) or 1.0)
        yield from _consumer(
            self.env,
            self.stores[demand.store_key],
            demand.store_key,
            demand.rate_per_hour,
            truck_load,
            step_h,
            self.log,
            self.unmet,
            sim=self
        )

    def run(self, stores_cfg, makes, moves, demands):
        _build_stores(self.env, self.stores, stores_cfg, self.settings, self.log)
        
        # Start the master clock for ordered execution
        self.env.process(self.master_clock())

        make_groups = defaultdict(list)
        for unit in makes:
            key = (unit.location, unit.equipment)
            make_groups[key].append(unit)

        for (loc, eq), units in make_groups.items():
            merged_candidates: List[ProductionCandidate] = []
            if units:
                choice_rule = units[0].choice_rule or self.settings.get("make_output_choice", "min_fill_pct")
                step_hours = units[0].step_hours if getattr(units[0], 'step_hours', None) else float(
                    self.settings.get("step_hours", 1.0))
                maintenance_days = getattr(units[0], 'maintenance_days', None)
                unplanned_downtime_pct = getattr(units[0], 'unplanned_downtime_pct', 0.0) or 0.0
            else:
                choice_rule = self.settings.get("make_output_choice", "min_fill_pct")
                step_hours = float(self.settings.get("step_hours", 1.0))
                maintenance_days = None
                unplanned_downtime_pct = 0.0
            for u in units:
                if isinstance(u.candidates, list):
                    merged_candidates.extend(list(u.candidates))

            merged_unit = MakeUnit(
                location=loc,
                equipment=eq,
                candidates=merged_candidates,
                choice_rule=choice_rule,
                step_hours=step_hours,
                maintenance_days=maintenance_days,
                unplanned_downtime_pct=unplanned_downtime_pct,
            )
            res = simpy.Resource(self.env, capacity=1)
            self.env.process(self.producer(res, merged_unit))

        self.production_rate_map = {}
        for unit in makes:
            if hasattr(unit, 'candidates') and unit.candidates:
                for cand in unit.candidates:
                    out_key = getattr(cand, 'out_store_key', None)
                    rate = float(getattr(cand, 'rate_tph', 0.0) or 0.0)
                    if out_key and rate > 0:
                        current = self.production_rate_map.get(out_key, 0.0)
                        self.production_rate_map[out_key] = current + rate

        self.store_capacity_map = {}
        for sc in stores_cfg:
            key = getattr(sc, 'key', None)
            cap = float(getattr(sc, 'capacity', 0.0) or 0.0)
            if key and cap > 0:
                self.store_capacity_map[key] = cap

        # Build sole_supplier_stores: stores that only have ONE ship route serving them
        # These get a bonus in route selection to ensure they're not starved
        store_route_count = {}
        for mv in moves:
            if (getattr(mv, 'mode', 'TRAIN') or 'TRAIN').upper() == 'SHIP' and mv.itineraries:
                for it in mv.itineraries:
                    for step in it:
                        if step.get('kind') == 'unload':
                            sk = step.get('store_key')
                            if sk:
                                store_route_count[sk] = store_route_count.get(sk, 0) + 1
        
        self.sole_supplier_stores = {sk for sk, count in store_route_count.items() if count == 1}
        
        vessel_counter = 0
        for mv in moves:
            for i in range(mv.n_units):
                vessel_counter += 1
                self.env.process(self.transporter(mv, vessel_id=vessel_counter))

        for d in demands:
            try:
                self.demand_rate_map[str(d.store_key)] = float(getattr(d, 'rate_per_hour', 0.0) or 0.0)
            except Exception:
                self.demand_rate_map[str(d.store_key)] = 0.0
            self.env.process(self.consumer(d))

        # Add production consumption rates to demand map for ship urgency scoring
        # This ensures mills that consume materials (like CL for GP production) are recognized as having demand
        for unit in makes:
            if hasattr(unit, 'candidates') and unit.candidates:
                for cand in unit.candidates:
                    in_keys = []
                    if hasattr(cand, 'in_store_keys') and cand.in_store_keys:
                        in_keys = list(cand.in_store_keys)
                    elif hasattr(cand, 'in_store_key') and cand.in_store_key:
                        in_keys = [cand.in_store_key]
                    
                    consumption_pct = float(getattr(cand, 'consumption_pct', 1.0) or 1.0)
                    # Use rate_tph (tons per hour) - the correct attribute name
                    rate = float(getattr(cand, 'rate_tph', 0.0) or 0.0) * consumption_pct
                    
                    for in_key in in_keys:
                        if in_key and rate > 0:
                            current = self.demand_rate_map.get(str(in_key), 0.0)
                            self.demand_rate_map[str(in_key)] = current + rate

        horizon_days = self.settings.get("horizon_days", 365)

        # Snapshot Day 0
        self.snapshot()

        step_pct = max(1, min(int(self.settings.get("progress_step_pct", 10) or 10), 50))

        # FIX: Store checkpoints as (percent, day) tuples
        checkpoints = []
        for p in range(step_pct, 100, step_pct):
            d = int(round(horizon_days * (p / 100.0)))
            # FIX: Compare d with checkpoints[-1][1] (the day part of the tuple)
            if not checkpoints or d > checkpoints[-1][1]:
                checkpoints.append((p, d))
        cp_idx = 0

        for day in range(1, horizon_days + 1):
            target_time = day * 24
            self.env.run(until=target_time)
            self.snapshot()  # Snapshot at end of day

            # FIX: Compare day with checkpoints[cp_idx][1]
            while cp_idx < len(checkpoints) and day >= checkpoints[cp_idx][1]:
                pct = checkpoints[cp_idx][0]
                _log_progress(f"Progress: {pct}% ({day}/{horizon_days} days)")
                cp_idx += 1

        _log_progress(f"Progress: 100% ({horizon_days}/{horizon_days} days)")
        _log_progress("\n=== Simulation Complete ===")
        _log_progress("Final store levels:")
        for key in sorted(self.stores):
            _log_progress(f"  {key}: {self.stores[key].level:.1f} tons")
        total_unmet = sum(self.unmet.values())
        if total_unmet > 0:
            _log_progress(f"Total unmet demand: {total_unmet:.1f} tons")

        # Log ClosingBalance for each store
        for key, cont in self.stores.items():
            parts = key.split('|')
            product = parts[0] if len(parts) > 0 else None
            location = parts[1] if len(parts) > 1 else None
            self.log(
                process="Store",
                event="ClosingBalance",
                location=location,
                equipment=None,
                product=product,
                qty=cont.level,
                time=0.0,
                from_store=None,
                from_level=None,
                to_store=key,
                to_level=cont.level,
                route_id=None
            )