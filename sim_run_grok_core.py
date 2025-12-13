# sim_run_grok_core.py
from __future__ import annotations
from dataclasses import dataclass
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

    def log(self, event: str, **details):
        self.action_log.append({"event": event, "time_h": self.env.now, **details})

    def snapshot(self):
        # Change from daily to hourly snapshots
        if self.env.now % 1 < 0.01:  # Hourly (was % 24)
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

    def build_stores(self, store_configs: List[StoreConfig]):
        if self.settings.get("random_seed") is not None:
            random.seed(self.settings["random_seed"])
        rand_open = self.settings.get("random_opening", True)

        for cfg in store_configs:
            opening = random.uniform(cfg.opening_low, cfg.opening_high) if rand_open else cfg.opening_high
            opening = max(0.0, min(opening, cfg.capacity))
            self.stores[cfg.key] = simpy.Container(self.env, capacity=cfg.capacity, init=opening)

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
        while True:
            with resource.request() as req:
                yield req
                eligible = [
                    c for c in unit.candidates
                    if self.stores[c.out_store_key].level + c.rate_tph * unit.step_hours <= self.stores[c.out_store_key].capacity + 1e-6
                ]
                if not eligible:
                    yield self.env.timeout(unit.step_hours)
                    continue

                cand = eligible[self.choose_output(unit.choice_rule, eligible)]
                qty = cand.rate_tph * unit.step_hours

                if cand.in_store_key:
                    needed = qty * cand.consumption_pct
                    taken = min(self.stores[cand.in_store_key].level, needed)
                    yield self.stores[cand.in_store_key].get(taken)
                    self.log("Consumed", location=unit.location, equipment=unit.equipment, qty=taken, product=cand.product)
                    if needed > 0:
                        qty *= taken / needed

                yield self.stores[cand.out_store_key].put(qty)
                self.log("Produced", location=unit.location, equipment=unit.equipment, qty=qty, product=cand.product)
                self.snapshot()
                yield self.env.timeout(unit.step_hours)

    def transporter(self, route: TransportRoute):
        while True:
            # Find origin store with enough stock
            origin_cont = None
            origin_key = None
            for key in route.origin_stores:
                cont = self.stores[key]
                if cont.level >= route.payload_t:
                    origin_cont = cont
                    origin_key = key
                    break
            if origin_cont is None:
                yield self.env.timeout(1)
                continue

            # Find destination store with enough room
            dest_cont = None
            dest_key = None
            for key in route.dest_stores:
                cont = self.stores[key]
                if cont.capacity - cont.level >= route.payload_t:
                    dest_cont = cont
                    dest_key = key
                    break
            if dest_cont is None:
                yield self.env.timeout(1)
                continue

            # Load
            take = route.payload_t
            load_time = take / max(route.load_rate_tph, 1)
            yield self.env.timeout(load_time)
            yield origin_cont.get(take)
            self.log("Loaded", product=route.product, from_store=origin_key, to_location=route.dest_location, qty=take)

            # Transit
            yield self.env.timeout(route.to_min / 60)

            # Unload
            remaining = take
            while remaining > 0:
                chunk = min(remaining, route.unload_rate_tph)
                unload_time = chunk / max(route.unload_rate_tph, 1)
                yield self.env.timeout(unload_time)
                yield dest_cont.put(chunk)
                remaining -= chunk
                self.log("Unloaded", product=route.product, to_store=dest_key, qty=chunk)

            yield self.env.timeout(route.back_min / 60)

    def consumer(self, demand: Demand):
        store = self.stores[demand.store_key]
        while True:
            take = min(store.level, demand.rate_per_hour)
            if take > 0:
                yield store.get(take)
            unmet = demand.rate_per_hour - take
            if unmet > 0:
                self.unmet[demand.store_key] = self.unmet.get(demand.store_key, 0) + unmet
            yield self.env.timeout(1)

    def run(self, stores_cfg, makes, moves, demands):
        self.build_stores(stores_cfg)

        make_groups = defaultdict(list)
        for unit in makes:
            key = (unit.location, unit.equipment)
            make_groups[key].append(unit)

        for (loc, eq), units in make_groups.items():
            res = simpy.Resource(self.env, capacity=1)
            for unit in units:
                self.env.process(self.producer(res, unit))

        for mv in moves:
            for _ in range(mv.n_units):
                self.env.process(self.transporter(mv))

        for d in demands:
            self.env.process(self.consumer(d))

        horizon = self.settings.get("horizon_days", 365) * 24
        self.env.run(until=horizon)

        print("\n=== Simulation Complete ===")
        print("Final store levels:")
        for key in sorted(self.stores):
            print(f"  {key}: {self.stores[key].level:.1f} tons")
        total_unmet = sum(self.unmet.values())
        if total_unmet > 0:
            print(f"Total unmet demand: {total_unmet:.1f} tons")