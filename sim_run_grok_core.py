# sim_run_grok_core.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import simpy
import random
from collections import defaultdict

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

# UPDATED: Now includes product and lists of stores
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
    mode: str = "TRAIN"  # Transport mode: TRAIN or SHIP (default TRAIN for backward compatibility)
    # SHIP-only optional fields (ignored by TRAIN mover)
    route_group: Optional[str] = None
    speed_knots: Optional[float] = None
    hulls_per_vessel: Optional[int] = None
    payload_per_hull_t: Optional[float] = None
    itineraries: Optional[List[List[dict]]] = None  # list of itineraries; each itinerary is a list of step dicts
    berth_info: Optional[Dict[str, dict]] = None    # location -> {berths, p_occupied, pilot_in_h, pilot_out_h}
    nm_distance: Optional[Dict[tuple, float]] = None  # (loc1, loc2) -> nautical miles

@dataclass
class Demand:
    store_key: str
    rate_per_hour: float  # interpreted as average tons per hour; consumer chunks by truck size

class SupplyChainSimulation:
    def __init__(self, settings: dict):
        self.settings = settings
        self.env = simpy.Environment()
        self.stores: Dict[str, simpy.Container] = {}
        self.unmet: Dict[str, float] = {}
        self.action_log: List[dict] = []
        self.inventory_snapshots: List[dict] = []
        # Map of demand rate per store_key (tons per hour)
        self.demand_rate_map: Dict[str, float] = {}
        # Port berth resources for SHIP operations (location -> simpy.Resource)
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

    def periodic_snapshot_process(self):
        # No longer used - snapshots are taken in run() method
        while True:
            self.snapshot()
            yield self.env.timeout(24)

    def build_stores(self, store_configs: List[StoreConfig]):
        # Delegate to storage module to keep this orchestrator slim
        from sim_run_grok_core_store import build_stores as _build_stores
        _build_stores(self.env, self.stores, store_configs, self.settings)

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
        # Delegate to manufacturing module
        from sim_run_grok_core_make import producer as _producer
        yield from _producer(self, resource, unit)

    def transporter(self, route: TransportRoute):
        # Dispatch based on transport mode.
        mode = (getattr(route, 'mode', 'TRAIN') or 'TRAIN').upper()
        if mode == 'SHIP':
            from sim_run_grok_core_move_ship import transporter as _transporter_ship
            yield from _transporter_ship(self, route)
        elif mode == 'TRAIN':
            from sim_run_grok_core_move_train import transporter as _transporter_train
            yield from _transporter_train(self, route)
        else:
            # Unknown mode: log and default to TRAIN behavior to avoid breaking runs
            try:
                self.log("warn_unknown_transport_mode", route_mode=mode)
            except Exception:
                pass
            from sim_run_grok_core_move_train import transporter as _transporter_train
            yield from _transporter_train(self, route)

    def consumer(self, demand: Demand):
        # Delegate to delivery (TRUCK) module
        from sim_run_grok_core_deliver import consumer as _consumer
        yield from _consumer(self, demand)

    def run(self, stores_cfg, makes, moves, demands):
        self.build_stores(stores_cfg)

        make_groups = defaultdict(list)
        for unit in makes:
            key = (unit.location, unit.equipment)
            make_groups[key].append(unit)

        for (loc, eq), units in make_groups.items():
            # Merge all candidates across units sharing the same physical equipment so the
            # unit can only produce ONE output at a time, chosen by the selection rule.
            merged_candidates: List[ProductionCandidate] = []
            # Derive rule and step from first unit in group (fallback to settings)
            if units:
                choice_rule = units[0].choice_rule or self.settings.get("make_output_choice", "min_fill_pct")
                step_hours = units[0].step_hours if getattr(units[0], 'step_hours', None) else float(self.settings.get("step_hours", 1.0))
            else:
                choice_rule = self.settings.get("make_output_choice", "min_fill_pct")
                step_hours = float(self.settings.get("step_hours", 1.0))
            for u in units:
                if isinstance(u.candidates, list):
                    merged_candidates.extend(list(u.candidates))
            # Build a synthetic merged unit
            merged_unit = MakeUnit(
                location=loc,
                equipment=eq,
                candidates=merged_candidates,
                choice_rule=choice_rule,
                step_hours=step_hours,
            )
            # Single shared resource enforces exclusivity
            res = simpy.Resource(self.env, capacity=1)
            self.env.process(self.producer(res, merged_unit))

        for mv in moves:
            for _ in range(mv.n_units):
                self.env.process(self.transporter(mv))

        for d in demands:
            # Track demand rate per store for plotting/reporting (tons per hour)
            try:
                self.demand_rate_map[str(d.store_key)] = float(getattr(d, 'rate_per_hour', 0.0) or 0.0)
            except Exception:
                self.demand_rate_map[str(d.store_key)] = 0.0
            self.env.process(self.consumer(d))

        horizon_days = self.settings.get("horizon_days", 365)
        horizon = horizon_days * 24

        # Progress checkpoints (percent-based)
        try:
            step_pct = int(self.settings.get("progress_step_pct", 10) or 10)
        except Exception:
            step_pct = 10
        step_pct = max(1, min(step_pct, 50))
        checkpoints: list[tuple[int, int]] = []  # (percent, day_number)
        for p in range(step_pct, 100, step_pct):  # up to but not including 100%
            d = int(round(horizon_days * (p / 100.0)))
            d = max(1, min(d, int(horizon_days)))
            # ensure strictly increasing day targets
            if not checkpoints or d > checkpoints[-1][1]:
                checkpoints.append((p, d))
        cp_idx = 0

        # Run simulation in daily steps, taking snapshots at each day boundary
        self.snapshot()  # Initial snapshot at t=0
        for day in range(1, horizon_days + 1):
            target_time = day * 24
            self.env.run(until=target_time)
            self.snapshot()  # Snapshot at end of each day
            # Emit progress logs when crossing checkpoints
            while cp_idx < len(checkpoints) and day >= checkpoints[cp_idx][1]:
                pct = checkpoints[cp_idx][0]
                # Print to stdout so callers (e.g., Flask) can stream it
                print(f"Progress: {pct}% ({day}/{horizon_days} days, t={int(target_time)}h)", flush=True)
                cp_idx += 1

        # Always print 100% at the end
        print(f"Progress: 100% ({horizon_days}/{horizon_days} days, t={int(horizon)}h)")
        print("\n=== Simulation Complete ===")
        print("Final store levels:")
        for key in sorted(self.stores):
            print(f"  {key}: {self.stores[key].level:.1f} tons")
        total_unmet = sum(self.unmet.values())
        if total_unmet > 0:
            print(f"Total unmet demand: {total_unmet:.1f} tons")