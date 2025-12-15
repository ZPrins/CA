# sim_run_grok_core.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import simpy
import random
from collections import defaultdict

# IMPORT LOGIC MODULES AT TOP LEVEL (Safe now that they don't import 'sim')
from sim_run_grok_core_store import build_stores as _build_stores
from sim_run_grok_core_make import producer as _producer
from sim_run_grok_core_move_train import transporter as _transporter_train
from sim_run_grok_core_move_ship import transporter as _transporter_ship
from sim_run_grok_core_deliver import consumer as _consumer


@dataclass
class StoreConfig:
    key: str
    capacity: float
    opening_low: float
    opening_high: float


@dataclass
class ProductionCandidate:
    product: str
    out_store_key: str
    in_store_key: Optional[str] = None
    rate_tph: float = 0.0
    consumption_pct: float = 1.0


@dataclass
class MakeUnit:
    location: str
    equipment: str
    candidates: List[ProductionCandidate]
    choice_rule: str = "min_fill_pct"
    step_hours: float = 1.0


@dataclass
class TransportRoute:
    product: str
    origin_location: str
    dest_location: str
    origin_stores: List[str]
    dest_stores: List[str]
    n_units: int
    payload_t: float
    load_rate_tph: float
    unload_rate_tph: float
    to_min: float
    back_min: float
    mode: str = "TRAIN"
    route_group: Optional[str] = None
    speed_knots: Optional[float] = None
    hulls_per_vessel: Optional[int] = None
    payload_per_hull_t: Optional[float] = None
    itineraries: Optional[List[List[dict]]] = None
    berth_info: Optional[Dict[str, dict]] = None
    nm_distance: Optional[Dict[tuple, float]] = None


@dataclass
class Demand:
    store_key: str
    rate_per_hour: float


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
                "day": int(self.env.now // 24) + 1,
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

    # ------------------------------------------------------------------
    # PROCESS METHODS - Now just wrappers that delegate with explicit args
    # ------------------------------------------------------------------

    def producer(self, resource: simpy.Resource, unit: MakeUnit):
        yield from _producer(
            self.env,
            resource,
            unit,
            self.stores,
            self.log,
            self.choose_output
        )

    def transporter(self, route: TransportRoute):
        mode = (getattr(route, 'mode', 'TRAIN') or 'TRAIN').upper()

        if mode == 'SHIP':
            # Ship still uses 'sim' in its signature because we haven't refactored it yet.
            # (If you refactored ship too, update this call similarly to train below)
            yield from _transporter_ship(self, route)

        elif mode == 'TRAIN':
            yield from _transporter_train(
                self.env,
                route,
                self.stores,
                self.log,
                require_full=self.settings.get("require_full_payload", True),
                debug_full=self.settings.get("debug_full_payload", False)
            )
        else:
            # Default fallback
            yield from _transporter_train(
                self.env,
                route,
                self.stores,
                self.log,
                require_full=self.settings.get("require_full_payload", True)
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

        for mv in moves:
            for _ in range(mv.n_units):
                self.env.process(self.transporter(mv))

        for d in demands:
            try:
                self.demand_rate_map[str(d.store_key)] = float(getattr(d, 'rate_per_hour', 0.0) or 0.0)
            except Exception:
                self.demand_rate_map[str(d.store_key)] = 0.0
            self.env.process(self.consumer(d))

        horizon_days = self.settings.get("horizon_days", 365)
        # ... (rest of the run method remains identical to your original)
        horizon = horizon_days * 24

        try:
            step_pct = int(self.settings.get("progress_step_pct", 10) or 10)
        except Exception:
            step_pct = 10
        step_pct = max(1, min(step_pct, 50))
        checkpoints = []
        for p in range(step_pct, 100, step_pct):
            d = int(round(horizon_days * (p / 100.0)))
            d = max(1, min(d, int(horizon_days)))
            if not checkpoints or d > checkpoints[-1][1]:
                checkpoints.append((p, d))
        cp_idx = 0

        self.snapshot()
        for day in range(1, horizon_days + 1):
            target_time = day * 24
            self.env.run(until=target_time)
            self.snapshot()
            while cp_idx < len(checkpoints) and day >= checkpoints[cp_idx][1]:
                pct = checkpoints[cp_idx][0]
                print(f"Progress: {pct}% ({day}/{horizon_days} days, t={int(target_time)}h)", flush=True)
                cp_idx += 1

        print(f"Progress: 100% ({horizon_days}/{horizon_days} days, t={int(horizon)}h)")
        print("\n=== Simulation Complete ===")
        print("Final store levels:")
        for key in sorted(self.stores):
            print(f"  {key}: {self.stores[key].level:.1f} tons")
        total_unmet = sum(self.unmet.values())
        if total_unmet > 0:
            print(f"Total unmet demand: {total_unmet:.1f} tons")