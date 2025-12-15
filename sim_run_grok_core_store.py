# sim_run_grok_core_store.py
from __future__ import annotations
from typing import List
import random
import simpy

from sim_run_grok_core import StoreConfig


def build_stores(env: simpy.Environment, stores_dict: dict, store_configs: List[StoreConfig], settings: dict) -> None:
    """
    Initialize simpy.Container objects for each store and put them into stores_dict.
    This function encapsulates storage initialization/opening logic.
    """
    if settings.get("random_seed") is not None:
        random.seed(settings["random_seed"])
    rand_open = settings.get("random_opening", True)

    for cfg in store_configs:
        opening = random.uniform(cfg.opening_low, cfg.opening_high) if rand_open else cfg.opening_high
        opening = max(0.0, min(opening, cfg.capacity))
        stores_dict[cfg.key] = simpy.Container(env, capacity=cfg.capacity, init=opening)
