# sim_run_core_store.py
from __future__ import annotations
from typing import List
import random
import simpy

# CHANGED: Import from types
from sim_run_types import StoreConfig

def build_stores(env: simpy.Environment, stores_dict: dict, store_configs: List[StoreConfig], settings: dict) -> None:
    if settings.get("random_seed") is not None:
        random.seed(settings["random_seed"])
    rand_open = settings.get("random_opening", True)

    for cfg in store_configs:
        opening = random.uniform(cfg.opening_low, cfg.opening_high) if rand_open else cfg.opening_high
        opening = max(0.0, min(opening, cfg.capacity))
        setattr(cfg, '_actual_opening', opening)
        stores_dict[cfg.key] = simpy.Container(env, capacity=cfg.capacity, init=opening)