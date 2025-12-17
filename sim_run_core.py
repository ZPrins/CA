# sim_run_core.py
from __future__ import annotations
from typing import Dict, List
import simpy
from collections import defaultdict

from sim_run_types import StoreConfig, ProductionCandidate, MakeUnit, TransportRoute, Demand

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

    def log(self, event: str, **details):
        self.action_log.append({"event": event, "time_h": self.env.now, **details})

    def snapshot(self):
        for key, cont in self.stores.items():
            parts = key.split("|")
            pc = parts[0] if len(parts) > 0 else ""
            loc = parts[1] if len(parts) > 1 else ""
            eq = parts[2] if len(parts) > 2 else ""
            inp = parts[3] if len(parts) > 3 else ""
            fill = cont.level / cont.capacity if cont.capacity > 0 else 1.0
            self.inventory_snapshots.append({
                "day": int(self.env.now // 24),  # 0-indexed day for snapshots
                "time_h": self.env.now,
                "product_class": pc,
                "location": loc,
                "equipment": eq,
                "input": inp,
                "store_key": key,
                "level": cont.level,
                "capacity": cont.capacity,
                "fill_pct": fill,
            })

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
            self.choose_output
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
                vessel_id=vessel_id
            )
        else:  # TRAIN
            yield from _transporter_train(
                self.env,
                route,
                self.stores,
                self.log,
                require_full=self.settings.get("require_full_payload", True),
                debug_full=self.settings.get("debug_full_payload", False)
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
            self.unmet
        )

    def run(self, stores_cfg, makes, moves, demands):
        _build_stores(self.env, self.stores, stores_cfg, self.settings)

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
            else:
                choice_rule = self.settings.get("make_output_choice", "min_fill_pct")
                step_hours = float(self.settings.get("step_hours", 1.0))
            for u in units:
                if isinstance(u.candidates, list):
                    merged_candidates.extend(list(u.candidates))

            merged_unit = MakeUnit(
                location=loc,
                equipment=eq,
                candidates=merged_candidates,
                choice_rule=choice_rule,
                step_hours=step_hours,
            )
            res = simpy.Resource(self.env, capacity=1)
            self.env.process(self.producer(res, merged_unit))

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
                print(f"Progress: {pct}% ({day}/{horizon_days} days)", flush=True)
                cp_idx += 1

        print(f"Progress: 100% ({horizon_days}/{horizon_days} days)")
        print("\n=== Simulation Complete ===")
        print("Final store levels:")
        for key in sorted(self.stores):
            print(f"  {key}: {self.stores[key].level:.1f} tons")
        total_unmet = sum(self.unmet.values())
        if total_unmet > 0:
            print(f"Total unmet demand: {total_unmet:.1f} tons")