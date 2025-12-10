"""
SimPy model builder for 'generated_model_inputs.xlsx'.

What this script does
- Loads the generated inputs workbook (created by supply_chain_viz.prepare_inputs_generate)
  with sheets: Network, Settings, Make, Store, Move, Deliver (or Delivery → Deliver)
- Builds a dynamic SimPy model wiring together:
  - Make units: produce material at configured rates, optionally consuming input
  - Store units: inventory buffers with capacity and opening stock
  - Move routes: transporters cycling between origin and destination with load/unload/travel
  - Deliver consumers: demand draw from local store
- Runs the simulation for the configured horizon and prints simple KPIs.

Notes and scope
- This is a pragmatic, readable baseline. It uses time buckets (Hours/Half Days/Days)
  from Settings to progress production/consumption in discrete steps where needed.
- You can refine the logic (batch sizes, stochasticity, maintenance/downtime, multimaterial routing)
  by extending the clearly marked sections below.

Usage examples
  python sim_from_generated.py                      # uses ./generated_model_inputs.xlsx
  python sim_from_generated.py --xlsx Model Inputs.xlsx --product-class GP --days 60

Outputs
- Prints: ending stocks per store, unmet demand per location, simple route throughputs.
- Returns exit code 0 if run completes.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import pandas as pd

try:
    import simpy
except Exception as e:  # pragma: no cover
    raise RuntimeError("simpy isn't installed. Install with 'pip install simpy'.") from e

# Reuse helpers and constants from supply_chain_viz
import supply_chain_viz as viz


# ------------------------------ Data models ---------------------------------

Key = str  # composite key helpers use strings


def key_triplet(loc: str, eq: str, inp: str) -> Key:
    return f"{str(loc).strip().upper()}|{str(eq).strip().upper()}|{str(inp).strip().upper()}"


def key_pair(loc: str, inp: str) -> Key:
    return f"{str(loc).strip().upper()}|{str(inp).strip().upper()}"


@dataclass
class Settings:
    runs: int = 1
    horizon_days: float = 30
    time_bucket: str = "Hours"  # Hours | Half Days | Days

    def step_hours(self) -> float:
        tb = str(self.time_bucket).strip().lower()
        if tb in ("hour", "hours", "hr", "hrs"):
            return 1.0
        if tb in ("half days", "half-day", "halfday", "12h"):
            return 12.0
        # default days
        return 24.0

    def horizon_hours(self) -> float:
        return float(self.horizon_days) * 24.0


@dataclass
class RouteStats:
    trips_completed: int = 0
    tons_moved: float = 0.0


# ------------------------------ Load workbook -------------------------------


def load_generated_inputs(xlsx_path: Path) -> dict[str, pd.DataFrame]:
    if not xlsx_path.exists():
        raise FileNotFoundError(xlsx_path)

    # Network via the robust normalizer
    net_df = viz.read_table(xlsx_path)

    # Other sheets best-effort
    def read_sheet(name: str) -> pd.DataFrame:
        df = viz._read_sheet_df(xlsx_path, name)  # type: ignore[attr-defined]
        return df if df is not None else pd.DataFrame()

    settings = read_sheet("Settings")
    make = read_sheet("Make")
    store = read_sheet("Store")
    move = read_sheet("Move")
    deliver = read_sheet("Deliver")
    if deliver.empty:
        # Some books use "Delivery"
        deliver = read_sheet("Delivery")

    return {
        "Network": net_df,
        "Settings": settings,
        "Make": make,
        "Store": store,
        "Move": move,
        "Deliver": deliver,
    }


# ------------------------------ Build model ---------------------------------


def parse_settings(df: pd.DataFrame) -> Settings:
    s = Settings()
    if df is None or df.empty:
        return s
    # Normalize
    cols = {str(c).strip().lower(): c for c in df.columns}
    if "setting" in cols and "value" in cols:
        for _, r in df.iterrows():
            name = str(r[cols["setting"]]).strip().lower()
            val = r[cols["value"]]
            if name.startswith("number of simulation runs"):
                try:
                    s.runs = int(val)
                except Exception:
                    pass
            elif name.startswith("modeling horizon"):
                try:
                    s.horizon_days = float(val)
                except Exception:
                    pass
            elif name.startswith("time buckets"):
                s.time_bucket = str(val)
    return s


@dataclass
class Components:
    env: simpy.Environment
    # Inventory buffers keyed by (PC|Location|Equipment|Input)
    stores: Dict[Key, simpy.Container]
    # Make units keyed by node id
    make_units: Dict[Key, simpy.Resource]
    # Stats
    unmet_demand: Dict[Key, float]
    route_stats: Dict[Key, RouteStats]


def build_simpy_from_generated(
    xlsx_path: Path,
    product_class: Optional[str] = None,
) -> tuple[Components, dict[str, Any]]:
    """Build a SimPy environment from the generated inputs workbook.

    Returns (components, metadata)
    """
    sheets = load_generated_inputs(xlsx_path)
    net = sheets["Network"].copy()
    if product_class:
        net = net[net["product_class"].str.upper() == str(product_class).upper()].copy()
        if net.empty:
            raise ValueError(f"No rows found for product class '{product_class}'.")

    settings = parse_settings(sheets["Settings"])  # defaults if missing

    # Initialize export bundle
    export: dict[str, Any] = {
        "settings": {},
        "stores": [],
        "makes": [],
        "moves": [],
        "deliveries": [],
    }

    # Seed export settings early and randomness config (from sim_config if available)
    use_random_opening = True
    random_seed: Optional[int] = None
    try:
        import sim_config as _cfg_mod  # type: ignore
        try:
            _cfg = _cfg_mod.Config() if hasattr(_cfg_mod, "Config") else getattr(_cfg_mod, "config", None)
        except Exception:
            _cfg = getattr(_cfg_mod, "config", None)
        if _cfg is not None:
            use_random_opening = bool(getattr(_cfg, "use_random_opening", True))
            random_seed = getattr(_cfg, "random_seed", None)
    except Exception:
        # If sim_config isn't available (e.g., CLI mode), default values above apply
        pass

    try:
        # Read optional make output choice rule and snapshot interval from sim_config
        make_output_choice = "min_fill_pct"
        snapshot_hours = 24.0
        try:
            if _cfg is not None:
                make_output_choice = str(getattr(_cfg, "make_output_choice", make_output_choice))
                snapshot_hours = float(getattr(_cfg, "snapshot_hours", snapshot_hours))
        except Exception:
            pass
        export["settings"] = {
            "horizon_days": float(getattr(settings, "horizon_days", 30)),
            "time_bucket": str(getattr(settings, "time_bucket", "Hours")),
            "step_hours": float(settings.step_hours()),
            "random_opening": bool(use_random_opening),
            "random_seed": random_seed,
            "make_output_choice": make_output_choice,
            "snapshot_hours": snapshot_hours,
        }
    except Exception:
        pass

    # Normalize ancillary sheets: enforce the expected columns if present
    make_df = sheets["Make"].copy()
    store_df = sheets["Store"].copy()
    move_df = sheets["Move"].copy()
    deliver_df = sheets["Deliver"].copy()

    # Build key lookups
    # Make sheet
    if not make_df.empty:
        # Normalize to a unified 'Output' column (backward compatible):
        # Prefer 'Output'; else if only 'Product' present, use it as Output; else fall back to 'Input'.
        cols_lower = {str(c).strip().lower(): c for c in make_df.columns}
        if "output" in cols_lower:
            out_col = cols_lower["output"]
            make_df = make_df.rename(columns={out_col: "Output"})
        elif "product" in cols_lower:
            prod_col = cols_lower["product"]
            make_df = make_df.rename(columns={prod_col: "Output"})
        elif "input" in cols_lower:
            in_col = cols_lower["input"]
            make_df = make_df.rename(columns={in_col: "Output"})
        # Standardize key columns
        make_df = make_df.rename(columns={
            "Location": "Location", "Equipment Name": "Equipment Name",
        })
        make_df["KEY"] = viz._normalize_key_triplet(  # type: ignore[attr-defined]
            make_df.fillna(""), "Location", "Equipment Name", "Output"
        )
    # Store sheet
    if not store_df.empty:
        store_df = store_df.rename(columns={
            "Location": "Location", "Equipment Name": "Equipment Name", "Input": "Input",
        })
        store_df["KEY"] = viz._normalize_key_triplet(  # type: ignore[attr-defined]
            store_df.fillna(""), "Location", "Equipment Name", "Input"
        )
    # Move sheet
    if not move_df.empty:
        # Unify potential name mismatches (support both legacy 'Product Class' and new 'Product')
        move_df = move_df.rename(columns={
            "Product": "Product",
            "Product Class": "Product Class",
            "Location": "Location",
            "Equipment Name": "Equipment Name",
            "Next Location": "Next Location",
        })
        key_first = "Product" if "Product" in move_df.columns else "Product Class"
        move_df["KEY4"] = viz._normalize_key_quad(  # type: ignore[attr-defined]
            move_df.fillna(""), key_first, "Location", "Equipment Name", "Next Location"
        )
    # Deliver sheet
    if not deliver_df.empty:
        deliver_df = deliver_df.rename(columns={
            "Location": "Location", "Input": "Input"
        })
        deliver_df["KEY"] = viz._normalize_key_pair(  # type: ignore[attr-defined]
            deliver_df.fillna(""), "Location", "Input"
        )

    # Routing: map source node → list of (dst_node, output)
    routes: Dict[Key, list[Tuple[Key, str]]] = {}
    nodes_by_id: Dict[Key, dict] = {}

    for _, r in net.iterrows():
        pc_row = str(r.get("product_class", "")).strip()
        src = viz.node_id(r["location"], r["equipment_name"], pc_row) if r["equipment_name"] else None
        if src:
            nodes_by_id.setdefault(src, {
                "product_class": pc_row,
                "location": r["location"],
                "equipment": r["equipment_name"],
                "process": r["process"],
                "input": r["input"],
                "output": r["output"],
            })
        next_loc = str(r["next_location"]).strip()
        next_eq = str(r["next_equipment"]).strip()
        if src and next_loc and next_eq:
            dst = viz.node_id(next_loc, next_eq, pc_row)
            routes.setdefault(src, []).append((dst, str(r["output"]).strip()))
            nodes_by_id.setdefault(dst, {
                "product_class": pc_row,
                "location": next_loc,
                "equipment": next_eq,
                "process": r["next_process"],
                "input": r["output"],
                "output": None,
            })

    # Build environment and components
    env = simpy.Environment()
    stores: Dict[Key, simpy.Container] = {}
    make_units: Dict[Key, simpy.Resource] = {}
    unmet_demand: Dict[Key, float] = {}
    route_stats: Dict[Key, RouteStats] = {}
    warnings: list[dict[str, Any]] = []
    # Detailed action log
    action_log: list[dict[str, Any]] = []
    # Inventory snapshots (periodic)
    inventory_snapshots: list[dict[str, Any]] = []


    def log_action(event: str, t: float, details: dict[str, Any]) -> None:
        try:
            entry = {"event": event, "time_h": float(t)}
            entry.update(details)
            action_log.append(entry)
        except Exception:
            # Never fail the sim due to logging
            pass

    def _to_float(val: Any, default: float = 0.0) -> float:
        try:
            if val is None:
                return float(default)
            if isinstance(val, (int, float)):
                return float(val)
            s = str(val).strip()
            if s == "" or s.lower() in {"nan", "none"}:
                return float(default)
            s = s.replace(",", "")
            return float(s)
        except Exception:
            return float(default)

    def _warn(kind: str, message: str, context: dict[str, Any]) -> None:
        warnings.append({"type": kind, "message": message, **context})

    # Helper to form store key using node info (PC|Loc|Eq|Input)
    def store_key(node_id_: Key, input_name: str) -> Key:
        nd = nodes_by_id[node_id_]
        return f"{nd['product_class'].upper()}|{str(nd['location']).strip().upper()}|{str(nd['equipment']).strip().upper()}|{str(input_name).strip().upper()}"

    # RNG for openings
    try:
        import random as _rnd_open
    except Exception:
        _rnd_open = None
    if random_seed is not None and _rnd_open is not None:
        try:
            _rnd_open.seed(int(random_seed))
        except Exception:
            pass

    # Create Store containers from the Store sheet (authoritative), not from Network nodes
    # Strategy:
    #  - Iterate each row in Store sheet (Location, Equipment Name, Input)
    #  - Determine applicable product classes (PCs) by inspecting nodes present in Network at that (Location, Equipment Name)
    #    If none found, fall back to all PCs observed anywhere in the Network. Respect the product_class filter.
    #  - For each applicable PC, create a store key PC|LOC|EQ|INPUT and instantiate a SimPy Container using capacity/opening from Store sheet.
    #  - Emit a diagnostic if no Network node matched the store row.
    if not store_df.empty:
        # Precompute product classes by (LOC, EQ) from nodes discovered in Network parsing
        pc_by_loc_eq: dict[tuple[str, str], set[str]] = {}
        all_pcs: set[str] = set()
        for nid, nd in nodes_by_id.items():
            pc_v = str(nd.get("product_class", "")).strip().upper()
            loc_v = str(nd.get("location", "")).strip().upper()
            eq_v = str(nd.get("equipment", "")).strip().upper()
            if pc_v:
                all_pcs.add(pc_v)
            key_le = (loc_v, eq_v)
            pc_by_loc_eq.setdefault(key_le, set()).add(pc_v)
        for _, srow in store_df.iterrows():
            loc = str(srow.get("Location", "")).strip()
            eq = str(srow.get("Equipment Name", "")).strip()
            inp = str(srow.get("Input", "")).strip()
            if not loc or not eq or not inp:
                continue
            loc_u, eq_u, inp_u = loc.upper(), eq.upper(), inp.upper()
            # Candidate PCs from network at this (loc, eq)
            pcs_here = pc_by_loc_eq.get((loc_u, eq_u), set())
            matched_network = len(pcs_here) > 0
            # Respect external product_class filter if provided
            if product_class:
                pcs_here = {str(product_class).upper()} if (not pcs_here or str(product_class).upper() in pcs_here) else {str(product_class).upper()}
            # If none matched, fall back to all PCs observed in network (or default to empty → skip)
            if not pcs_here:
                pcs_here = set(all_pcs)
            if not pcs_here:
                # If the workbook has no PCs at all, skip but warn
                _warn("store_without_network_node", "Store row had no matching nodes and no PCs detected in Network; skipped", {"location": loc, "equipment": eq, "input": inp})
                continue
            # Capacity and opening bounds from Store sheet
            capacity = _to_float(srow.get("Silo Max Capacity", 1e12), 1e12)
            hi = _to_float(srow.get("Silo Opening Stock (High)", 0.0), 0.0)
            lo = _to_float(srow.get("Silo Opening Stock (Low)", 0.0), 0.0)
            for pc in sorted(pcs_here):
                # If a product_class filter was supplied and this pc doesn't match, skip
                if product_class and pc != str(product_class).upper():
                    continue
                ctx = {"product_class": pc, "location": loc, "equipment": eq, "input": inp}
                # Sample opening if configured
                opening = 0.0
                if use_random_opening:
                    if lo is None and hi is None:
                        opening = 0.0
                    else:
                        lo_v = float(0.0 if lo is None else lo)
                        hi_v = float(0.0 if hi is None else hi)
                        if hi_v < lo_v:
                            _warn("store_opening_bounds_swapped", "Opening High < Low; swapped for sampling", {**ctx, "low": lo_v, "high": hi_v})
                            lo_v, hi_v = hi_v, lo_v
                        if _rnd_open is not None:
                            opening = float(_rnd_open.uniform(lo_v, hi_v))
                        else:
                            opening = (lo_v + hi_v) / 2.0
                else:
                    if lo is not None or hi is not None:
                        lo_v = float(0.0 if lo is None else lo)
                        hi_v = float(0.0 if hi is None else hi)
                        opening = (lo_v + hi_v) / 2.0
                    else:
                        opening = 0.0
                # Reconcile capacity and opening
                if capacity is None:
                    capacity = 1e12
                capacity = max(float(capacity), 0.0)
                opening = max(float(opening), 0.0)
                if capacity <= 0.0 and opening > 0.0:
                    _warn("store_capacity_adjusted", "Capacity was <= 0 while opening > 0; set capacity = opening", {**ctx, "orig_capacity": capacity, "orig_opening": opening})
                    capacity = opening
                if opening > capacity:
                    _warn("store_opening_clamped", "Opening stock exceeded capacity; clamped to capacity", {**ctx, "orig_capacity": capacity, "orig_opening": opening})
                    opening = capacity
                # Create the store key and container
                k = f"{pc}|{loc_u}|{eq_u}|{inp_u}"
                if k in stores:
                    # Avoid duplicate creation; skip or update to max capacity if needed
                    continue
                stores[k] = simpy.Container(env, init=opening, capacity=capacity)
                # Export
                try:
                    export["stores"].append({
                        "store_key": k,
                        "product_class": pc,
                        "location": loc,
                        "equipment": eq,
                        "input": inp,
                        "capacity": float(capacity if capacity is not None else 0.0),
                        "opening_low": float(0.0 if lo is None else lo),
                        "opening_high": float(0.0 if hi is None else hi),
                        "opening": float(opening),
                    })
                except Exception:
                    pass
            if not matched_network:
                _warn("store_without_network_node", "Store row had no matching (Location, Equipment) node in Network; created from Store sheet anyway", {"location": loc, "equipment": eq, "input": inp})
    else:
        # No Store sheet: keep previous behavior (no stores created at this stage)
        pass

    # Create Make unit resources and producer processes
    # Selection rule from config for multi-output makes
    _make_choice_rule = "min_fill_pct"
    try:
        if _cfg is not None:
            _make_choice_rule = str(getattr(_cfg, "make_output_choice", _make_choice_rule))
    except Exception:
        pass

    for nid, nd in nodes_by_id.items():
        if str(nd.get("process", "")).strip().lower() != "make":
            continue
        inp = str(nd.get("input", "")).strip()
        outp_default = str(nd.get("output", "")).strip()
        # All destination candidates for this Make
        dests = routes.get(nid, [])
        if not dests:
            continue
        candidates: list[dict[str, str]] = []
        for dst_id, edge_out in dests:
            try:
                nd_dst = nodes_by_id.get(dst_id)
                if not nd_dst or str(nd_dst.get("process", "")).strip().lower() != "store":
                    continue
                out_mat = str(edge_out or outp_default).strip()
                if not out_mat:
                    continue
                out_key = store_key(dst_id, out_mat)
                candidates.append({"product": out_mat, "out_store_key": out_key})
            except Exception:
                continue
        # Deduplicate candidates by out_store_key
        seen = set()
        uniq_candidates = []
        for c in candidates:
            if c["out_store_key"] in seen:
                continue
            seen.add(c["out_store_key"])
            uniq_candidates.append(c)
        candidates = uniq_candidates
        # Supplement candidates from Store sheet using Make.Output (prefer local stores at same location)
        try:
            out_pref = str(nd.get("output", "")).strip()
            pc_nd = str(nd.get("product_class", "")).strip().upper()
            loc_nd = str(nd.get("location", "")).strip().upper()
            if out_pref:
                for skey in list(stores.keys()):
                    try:
                        pc_k, loc_k, eq_k, inp_k = skey.split("|")
                    except Exception:
                        continue
                    if pc_k != pc_nd:
                        continue
                    if loc_k != loc_nd:
                        continue
                    if inp_k != out_pref.upper():
                        continue
                    if skey not in seen:
                        candidates.append({"product": out_pref, "out_store_key": skey})
                        seen.add(skey)
        except Exception:
            pass
        if not candidates:
            _warn("make_no_store_candidates", "Make unit has no valid destination Store candidates; idling", {"node": nid, "location": nd.get("location", ""), "equipment": nd.get("equipment", "")})
            continue

        # Look up Make parameters
        mean_rate = 0.0  # tons per hour
        cons_pct = 100.0  # % of output mass drawn as input
        if not make_df.empty:
            # Use explicit columns rather than precomputed KEY; prefer Output match, else Input match
            out_for_key = None
            try:
                out_for_key = str(nd.get("output", "")).strip() or None
            except Exception:
                out_for_key = None
            if out_for_key:
                try:
                    match = make_df[(make_df.get("Location", "").astype(str).str.strip().str.upper() == str(nd["location"]).strip().upper()) &
                                    (make_df.get("Equipment Name", "").astype(str).str.strip().str.upper() == str(nd["equipment"]).strip().upper()) &
                                    (make_df.get("Output", "").astype(str).str.strip().str.upper() == str(out_for_key).strip().upper())]
                except Exception:
                    match = pd.DataFrame()
            else:
                match = pd.DataFrame()
            if match.empty:
                try:
                    match = make_df[(make_df.get("Location", "").astype(str).str.strip().str.upper() == str(nd["location"]).strip().upper()) &
                                    (make_df.get("Equipment Name", "").astype(str).str.strip().str.upper() == str(nd["equipment"]).strip().upper()) &
                                    (make_df.get("Input", "").astype(str).str.strip().str.upper() == str(inp).strip().upper())]
                except Exception:
                    match = pd.DataFrame()
            if not match.empty:
                m = match.iloc[0]
                try:
                    mean_rate = float(m.get("Mean Production Rate (Tons/hr)", 0) or 0)
                except Exception:
                    pass
                try:
                    cons_pct = float(m.get("Consumption %", cons_pct) or cons_pct)
                except Exception:
                    pass
        # Bounds and warnings
        if mean_rate < 0:
            _warn("make_rate_negative", "Mean Production Rate was negative; clamped to 0", {"node": nid, "value": mean_rate})
            mean_rate = 0.0
        if cons_pct < 0 or cons_pct > 1:
            _warn("consumption_percent_out_of_range", "Consumption % outside [0,1]; clamped (expects 0.88 for 88%)", {"node": nid, "value": cons_pct})
            cons_pct = max(0.0, min(1.0, cons_pct))
        # Resource representing the unit (capacity 1 by default)
        res = simpy.Resource(env, capacity=1)
        make_units[nid] = res

        def _choose_candidate(rule: str, cands: list[dict[str, str]]):
            # Return index of chosen candidate based on store fill/level
            best_i = None
            best_metric = None
            for i, c in enumerate(cands):
                skey = c["out_store_key"]
                cont = stores.get(skey)
                if cont is None:
                    continue
                level = float(cont.level)
                cap = float(cont.capacity) if getattr(cont, "capacity", None) is not None else None
                if rule == "min_level" or cap in (None, 0.0):
                    metric = level
                else:
                    try:
                        metric = level / max(cap, 1e-9)
                    except Exception:
                        metric = 1.0
                if best_i is None or metric < best_metric:
                    best_i = i
                    best_metric = metric
            return best_i if best_i is not None else 0

        def producer(env: simpy.Environment, unit: simpy.Resource, cands: list[dict[str, str]], in_store_key_: Optional[Key], rate_tph: float, cons_pct_: float, step_h: float, rule: str, nd_meta: dict):
            # Single-output-at-a-time production; choose destination each step by rule
            while True:
                # Choose destination
                idx = _choose_candidate(rule, cands)
                target = cands[idx]
                out_store_key_ = target["out_store_key"]
                out_product_ = target["product"]

                # Calculate max producible this step
                qty_out = rate_tph * step_h if rate_tph > 0 else 0.0
                # Respect destination capacity first (avoid consuming input we can't store)
                dst_cont = stores.get(out_store_key_)
                if dst_cont is not None and qty_out > 0:
                    room = dst_cont.capacity - dst_cont.level if dst_cont.capacity is not None else qty_out
                    qty_out = max(0.0, min(qty_out, room))
                else:
                    qty_out = 0.0

                # Input withdrawal proportional to output (cons_pct_ is 0..1, e.g., 0.88 for 88%)
                qty_in = qty_out * cons_pct_ if in_store_key_ else 0.0
                if in_store_key_ and qty_out > 0:
                    src_cont = stores.get(in_store_key_)
                    if src_cont is not None:
                        avail = src_cont.level
                        take = min(avail, qty_in)
                        if take > 0:
                            # Withdraw from input store
                            yield src_cont.get(take)
                            # Log consumption event
                            try:
                                inp_token = in_store_key_.split("|")[3] if isinstance(in_store_key_, str) and "|" in in_store_key_ else ""
                                log_action(
                                    "Consumed",
                                    env.now,
                                    {
                                        "product_class": str(nd_meta.get("product_class", "")),
                                        "location": str(nd_meta.get("location", "")),
                                        "equipment": str(nd_meta.get("equipment", "")),
                                        "process": "Make",
                                        "product": str(inp_token),
                                        "qty_t": float(take),
                                        "units": "TON",
                                    },
                                )
                            except Exception:
                                pass
                        if qty_in > 0 and take < qty_in:
                            # Not enough input; scale output proportionally
                            scale = take / qty_in if qty_in > 0 else 0.0
                            if scale < 1.0:
                                try:
                                    _warn("make_scaled_for_input", "Scaled production due to insufficient input this step", {
                                        "requested_out_t": float(rate_tph * step_h),
                                        "planned_out_t": float(qty_in / max(cons_pct_, 1e-12)),
                                        "actual_out_t": float(qty_out * scale),
                                        "input_needed_t": float(qty_in),
                                        "input_taken_t": float(take),
                                    })
                                except Exception:
                                    pass
                            qty_out *= scale

                # Put output
                if dst_cont is not None and qty_out > 0:
                    yield dst_cont.put(qty_out)
                    # Log production event
                    try:
                        log_action(
                            "Produced",
                            env.now,
                            {
                                "product_class": str(nd_meta.get("product_class", "")),
                                "location": str(nd_meta.get("location", "")),
                                "equipment": str(nd_meta.get("equipment", "")),
                                "process": "Make",
                                "product": str(out_product_),
                                "qty_t": float(qty_out),
                                "units": "TON",
                            },
                        )
                    except Exception:
                        pass
                yield env.timeout(step_h)

        # Bind input store if there is a store at same PC+Location and input==inp
        in_store_key = None
        for nid2, nd2 in nodes_by_id.items():
            if nd2.get("product_class") != nd.get("product_class"):
                continue
            if str(nd2.get("location")).strip().upper() != str(nd.get("location")).strip().upper():
                continue
            if str(nd2.get("process", "")).strip().lower() == "store" and str(nd2.get("input", "")).strip().upper() == inp.upper():
                in_store_key = store_key(nid2, inp)
                break

        # If an input is required but the input store is missing, warn and skip spawning this Make unit
        try:
            if str(inp).strip() and in_store_key is None:
                _warn("make_input_store_missing", "Make unit requires input store but none found at location; idling", {
                    "node": nid,
                    "location": nd.get("location", ""),
                    "equipment": nd.get("equipment", ""),
                    "input": inp,
                })
                continue
        except Exception:
            pass

        # Export Make unit
        try:
            export["makes"].append({
                "node": nid,
                "product_class": str(nd.get("product_class", "")),
                "location": str(nd.get("location", "")),
                "equipment": str(nd.get("equipment", "")),
                "input": inp,
                "candidates": candidates,
                "in_store_key": in_store_key,
                "mean_rate_tph": float(mean_rate),
                "consumption_pct": float(cons_pct),
                "step_hours": float(settings.step_hours()),
                "choice_rule": _make_choice_rule,
            })
        except Exception:
            pass

        # Bind snapshot of this node's identity to avoid late-binding issues in closure
        nd_meta = {
            "product_class": nd.get("product_class", ""),
            "location": nd.get("location", ""),
            "equipment": nd.get("equipment", ""),
        }
        env.process(producer(env, res, candidates, in_store_key, mean_rate, cons_pct, settings.step_hours(), _make_choice_rule, nd_meta))

    # Create Move processes: per (PC, Location, Equipment Name, Next Location)
    # Each route spawns #Equipment parallel transporters cycling
    if not move_df.empty:
        # Normalize PC filter if provided
        if product_class:
            move_df = move_df[move_df.get("Product", "").astype(str).str.upper() == str(product_class).upper()].copy()
        for _, row in move_df.iterrows():
            # Material now comes from 'Product' column
            product = str(row.get("Product", "")).strip()
            src_loc = str(row.get("Location", "")).strip()
            src_eq = str(row.get("Equipment Name", "")).strip()
            dst_loc = str(row.get("Next Location", "")).strip()
            if not product or not src_loc or not dst_loc:
                continue
            # Determine the product-class token to prefix store keys. We assume it equals the product token (e.g., GP, SG, CL).
            pc = product.upper()
            # Helper: resolve a store key from current stores by criteria
            def _pick_store_key(candidate_keys: list[str]) -> str | None:
                if not candidate_keys:
                    return None
                # Prefer equipment names ending with '_STORE'
                best = None
                for k in candidate_keys:
                    try:
                        _pc, _loc, _eq, _inp = k.split("|")
                    except Exception:
                        _pc = _loc = _eq = _inp = ""
                    if _eq.endswith("_STORE"):
                        return k
                    if best is None:
                        best = k
                return best
            # Origin: match exact (pc, src_loc, src_eq, product) if present, else any store at src_loc with input==product
            src_loc_u, src_eq_u, product_u = src_loc.upper(), src_eq.upper(), product.upper()
            exact_origin = f"{pc}|{src_loc_u}|{src_eq_u}|{product_u}"
            origin_key = exact_origin if exact_origin in stores else None
            if origin_key is None:
                cand = [k for k in stores.keys() if k.startswith(pc+"|") and k.split("|")[1]==src_loc_u and k.split("|")[3]==product_u]
                origin_key = _pick_store_key(cand)
            if origin_key is None:
                _warn("move_origin_not_found", "Move origin store not found for lane", {"product": product, "location": src_loc, "equipment": src_eq})
                continue
            # Destination: any store at dst_loc with input==product
            dst_loc_u = dst_loc.upper()
            cand_d = [k for k in stores.keys() if k.startswith(pc+"|") and k.split("|")[1]==dst_loc_u and k.split("|")[3]==product_u]
            dest_key = _pick_store_key(cand_d)
            if dest_key is None:
                _warn("move_dest_not_found", "Move destination store not found for lane", {"product": product, "next_location": dst_loc})
                continue

            # Parameters
            try:
                n_units = int(row.get("#Equipment (99-unlimited)", 1) or 1)
            except Exception:
                n_units = 1
            try:
                parcels = float(row.get("#Parcels", 1) or 1)
            except Exception:
                parcels = 1.0
            try:
                cap_per_parcel = float(row.get("Capacity Per Parcel", 0) or 0)
            except Exception:
                cap_per_parcel = 0.0
            try:
                load_rate = float(row.get("Load Rate (Ton/hr)", 0) or 0)
            except Exception:
                load_rate = 0.0
            try:
                travel_to_min = float(row.get("Travel to Time (Min)", 0) or 0)
            except Exception:
                travel_to_min = 0.0
            try:
                unload_rate = float(row.get("Unload Rate (Ton/Hr)", 0) or 0)
            except Exception:
                unload_rate = 0.0
            try:
                travel_back_min = float(row.get("Travel back Time (Min)", 0) or 0)
            except Exception:
                travel_back_min = 0.0

            # Clamp negatives and warn
            ctx = {"pc": pc, "src_loc": src_loc, "src_eq": src_eq, "dst_loc": dst_loc, "material": product}
            if n_units < 0:
                _warn("move_units_negative", "#Equipment was negative; clamped to 0", {**ctx, "value": n_units})
                n_units = 0
            for name, val in [("#Parcels", parcels), ("Capacity Per Parcel", cap_per_parcel), ("Load Rate (Ton/hr)", load_rate), ("Travel to Time (Min)", travel_to_min), ("Unload Rate (Ton/Hr)", unload_rate), ("Travel back Time (Min)", travel_back_min)]:
                if val < 0:
                    _warn("move_param_negative", f"{name} was negative; clamped to 0", {**ctx, "value": val})
            parcels = max(0.0, parcels)
            cap_per_parcel = max(0.0, cap_per_parcel)
            load_rate = max(0.0, load_rate)
            travel_to_min = max(0.0, travel_to_min)
            unload_rate = max(0.0, unload_rate)
            travel_back_min = max(0.0, travel_back_min)

            payload = parcels * cap_per_parcel  # tons
            step_h = settings.step_hours()

            rkey = f"{pc.upper()}|{src_loc.upper()}|{src_eq.upper()}→{dst_loc.upper()}|{product.upper()}"
            route_stats.setdefault(rkey, RouteStats())

            def transporter(env: simpy.Environment, origin: simpy.Container, dest: simpy.Container, payload_t: float, load_rate_tph: float, unload_rate_tph: float, to_min: float, back_min: float, stats: RouteStats):
                def _parse_store_key(k: str):
                    try:
                        pc_v, loc_v, eq_v, inp_v = str(k).split("|")
                        return pc_v, loc_v, eq_v, inp_v
                    except Exception:
                        return "", "", "", ""
                while True:
                    # Load
                    if load_rate_tph <= 0 or payload_t <= 0:
                        # Nothing to do
                        yield env.timeout((to_min + unload_rate_tph + back_min) / 60.0 if to_min or back_min else step_h)
                        continue
                    # Load at rate, constrained by origin stock
                    load_time_h = payload_t / max(load_rate_tph, 1e-9)
                    take = min(origin.level, payload_t)
                    if take > 0:
                        yield env.timeout(load_time_h)
                        yield origin.get(take)
                        # Log load event
                        try:
                            _pc_o, _loc_o, _eq_o, _inp = _parse_store_key(origin_key)
                            _pc_d, _loc_d, _eq_d, _ = _parse_store_key(dest_key)
                            log_action(
                                "Loaded",
                                env.now,
                                {
                                    "product_class": _pc_o,
                                    "product": _inp,
                                    "qty_t": float(take),
                                    "units": "TON",
                                    "src_location": _loc_o,
                                    "src_equipment": _eq_o,
                                    "dst_location": _loc_d,
                                    "dst_equipment": _eq_d,
                                },
                            )
                        except Exception:
                            pass
                    else:
                        # Wait a step for stock to appear
                        yield env.timeout(step_h)
                        continue
                    # Travel to
                    yield env.timeout(to_min / 60.0)
                    # Unload at rate
                    unload_time_h = take / max(unload_rate_tph, 1e-9)
                    # Respect destination capacity
                    room = dest.capacity - dest.level if dest.capacity is not None else take
                    put_amt = min(take, max(room, 0))
                    if put_amt > 0:
                        yield env.timeout(unload_time_h)
                        yield dest.put(put_amt)
                        stats.trips_completed += 1
                        stats.tons_moved += put_amt
                        # Log unload event
                        try:
                            _pc_o, _loc_o, _eq_o, _inp = _parse_store_key(origin_key)
                            _pc_d, _loc_d, _eq_d, _ = _parse_store_key(dest_key)
                            log_action(
                                "Unloaded",
                                env.now,
                                {
                                    "product_class": _pc_o,
                                    "product": _inp,
                                    "qty_t": float(put_amt),
                                    "units": "TON",
                                    "src_location": _loc_o,
                                    "src_equipment": _eq_o,
                                    "dst_location": _loc_d,
                                    "dst_equipment": _eq_d,
                                },
                            )
                        except Exception:
                            pass
                    else:
                        # No room: return without unloading and try later
                        yield env.timeout(step_h)
                    # Travel back
                    yield env.timeout(back_min / 60.0)

            origin = stores.get(origin_key)
            dest = stores.get(dest_key)
            if origin is None or dest is None:
                continue
            # Export move lane
            try:
                export["moves"].append({
                    "route_key": rkey,
                    "origin_key": origin_key,
                    "dest_key": dest_key,
                    "n_units": int(n_units),
                    "parcels": float(parcels),
                    "cap_per_parcel": float(cap_per_parcel),
                    "payload_t": float(payload),
                    "load_rate_tph": float(load_rate),
                    "unload_rate_tph": float(unload_rate),
                    "to_min": float(travel_to_min),
                    "back_min": float(travel_back_min),
                    "step_hours": float(step_h),
                })
            except Exception:
                pass

            for i in range(max(1, n_units)):
                env.process(transporter(env, origin, dest, payload, load_rate, unload_rate, travel_to_min, travel_back_min, route_stats[rkey]))

    # Deliver processes: consume per location and input (Annual Demand in tons/year)
    if not deliver_df.empty:
        # Identify columns
        loc_col = None
        inp_col = None
        annual_col = None
        for c in deliver_df.columns:
            lc = str(c).strip().lower()
            if lc == "location":
                loc_col = c
            elif lc == "input":
                inp_col = c
            elif lc.startswith("annual"):
                annual_col = c
        if loc_col and annual_col:
            step_h = settings.step_hours()
            for _, r in deliver_df.iterrows():
                loc = str(r.get(loc_col, "")).strip()
                inp_val = str(r.get(inp_col, "")).strip()
                # Skip if no location
                if not loc:
                    continue
                # If Input missing, try to infer any store input at this location (same PC preference)
                if not inp_val:
                    # prefer inputs present in stores at loc
                    loc_u = loc.upper()
                    cand_inps = [k.split("|")[3] for k in stores.keys() if k.split("|")[1] == loc_u]
                    inp_val = cand_inps[0] if cand_inps else ""
                if not inp_val:
                    _warn("deliver_input_missing", "Deliver row missing Input and could not infer from stores", {"location": loc})
                    continue
                try:
                    annual = float(r.get(annual_col, 0) or 0)
                except Exception:
                    annual = 0.0
                rate_per_hour = annual / 8760.0
                rate_per_step = rate_per_hour * step_h
                if rate_per_step <= 0:
                    continue
                # Resolve a store container at this location with input==inp_val
                pc_candidates = set(k.split("|")[0] for k in stores.keys())
                skey = None
                # Prefer store keys at loc with input match and *_STORE equipment
                loc_u = loc.upper()
                inp_u = inp_val.upper()
                cand_keys = [k for k in stores.keys() if k.split("|")[1]==loc_u and k.split("|")[3]==inp_u]
                def _prefer_store(keys: list[str]) -> str | None:
                    if not keys:
                        return None
                    for k in keys:
                        try:
                            if k.split("|")[2].endswith("_STORE"):
                                return k
                        except Exception:
                            pass
                    return keys[0]
                skey = _prefer_store(cand_keys)
                if skey is None:
                    _warn("deliver_store_not_found", "No store found for Deliver row", {"location": loc, "input": inp_val})
                    continue
                cont = stores.get(skey)
                if cont is None:
                    continue
                # Accumulator key uses pc|location|input
                pc_val = skey.split("|")[0]
                ukey = f"{pc_val}|{loc_u}|{inp_u}"
                unmet_demand.setdefault(ukey, 0.0)
                def consumer(env: simpy.Environment, cont: simpy.Container, rate: float, acc_key: str):
                    while True:
                        take = min(cont.level, rate)
                        if take > 0:
                            yield cont.get(take)
                            try:
                                pc_v, loc_v, inp_v = acc_key.split("|")
                            except Exception:
                                pc_v, loc_v, inp_v = "", "", ""
                            try:
                                log_action(
                                    "Delivered",
                                    env.now,
                                    {
                                        "product_class": pc_v,
                                        "location": loc_v,
                                        "product": inp_v,
                                        "qty_t": float(take),
                                        "units": "TON",
                                        "process": "Deliver",
                                    },
                                )
                            except Exception:
                                pass
                        short = rate - take
                        if short > 0:
                            unmet_demand[acc_key] = unmet_demand.get(acc_key, 0.0) + short
                            try:
                                pc_v, loc_v, inp_v = acc_key.split("|")
                            except Exception:
                                pc_v, loc_v, inp_v = "", "", ""
                            try:
                                log_action(
                                    "Unmet",
                                    env.now,
                                    {
                                        "product_class": pc_v,
                                        "location": loc_v,
                                        "product": inp_v,
                                        "qty_t": float(short),
                                        "units": "TON",
                                        "process": "Deliver",
                                    },
                                )
                            except Exception:
                                pass
                        yield env.timeout(step_h)
                # Export delivery
                try:
                    export["deliveries"].append({
                        "store_key": skey,
                        "acc_key": ukey,
                        "rate_per_step": float(rate_per_step),
                        "step_hours": float(step_h),
                    })
                except Exception:
                    pass
                env.process(consumer(env, cont, rate_per_step, ukey))

    # Periodic inventory snapshot process
    try:
        snapshot_hours = float(export.get("settings", {}).get("snapshot_hours", 24.0))
    except Exception:
        snapshot_hours = 24.0

    def _snapshot_once(t_now: float):
        for k, cont in stores.items():
            try:
                pc_v, loc_v, eq_v, inp_v = k.split("|")
            except Exception:
                pc_v = loc_v = eq_v = inp_v = ""
            level = float(getattr(cont, "level", 0.0))
            cap_attr = getattr(cont, "capacity", None)
            try:
                cap_f = float(cap_attr) if cap_attr is not None else float("nan")
            except Exception:
                cap_f = float("nan")
            try:
                fill = level / cap_f if cap_f and cap_f > 0 else float("nan")
            except Exception:
                fill = float("nan")
            day_idx = int(t_now // 24.0) + 1
            inventory_snapshots.append({
                "day": day_idx,
                "time_h": float(t_now),
                "product_class": pc_v,
                "location": loc_v,
                "equipment": eq_v,
                "input": inp_v,
                "store_key": k,
                "level": level,
                "capacity": cap_f,
                "fill_pct": fill,
            })

    def snapshotter(env: simpy.Environment):
        # Initial snapshot at t=0
        _snapshot_once(env.now)
        while True:
            yield env.timeout(snapshot_hours)
            _snapshot_once(env.now)

    try:
        env.process(snapshotter(env))
    except Exception:
        pass

    components = Components(env=env, stores=stores, make_units=make_units, unmet_demand=unmet_demand, route_stats=route_stats)
    meta = {"settings": settings, "warnings": warnings, "action_log": action_log, "export": export, "inventory_snapshots": inventory_snapshots}
    return components, meta


# ------------------------------ CLI runner ----------------------------------


def run_and_report(xlsx: Path, product_class: Optional[str], days: Optional[float]) -> None:
    comps, meta = build_simpy_from_generated(xlsx, product_class=product_class)
    settings: Settings = meta["settings"]
    horizon_h = (days * 24.0) if days is not None else settings.horizon_hours()

    comps.env.run(until=horizon_h)

    # Report
    print("Simulation complete.")
    print(f"Horizon: {horizon_h:.2f} hours | Time bucket: {settings.time_bucket}")
    print("\nEnding stocks:")
    by_loc: Dict[str, float] = {}
    for skey, cont in sorted(comps.stores.items()):
        pc, loc, eq, inp = skey.split("|")
        print(f"  [{pc}] {loc:<16s} {eq:<16s} {inp:<6s} -> {cont.level:.2f} / {cont.capacity if cont.capacity != float('inf') else '∞'}")
        by_loc[f"{loc}|{inp}"] = by_loc.get(f"{loc}|{inp}", 0.0) + float(cont.level)

    if comps.unmet_demand:
        print("\nUnmet demand:")
        for k, v in comps.unmet_demand.items():
            pc, loc, inp = k.split("|")
            print(f"  [{pc}] {loc:<16s} {inp:<6s} -> {v:.2f} tons")

    if comps.route_stats:
        print("\nRoute stats:")
        for r, st in comps.route_stats.items():
            print(f"  {r}: trips={st.trips_completed}, tons_moved={st.tons_moved:.2f}")


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build and run a SimPy model from generated_model_inputs.xlsx")
    p.add_argument("--xlsx", type=Path, default=Path("generated_model_inputs.xlsx"), help="Path to generated workbook")
    p.add_argument("--product-class", type=str, default=None, help="Filter to a specific product class (e.g., GP)")
    p.add_argument("--days", type=float, default=None, help="Override horizon days (otherwise read from Settings)")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    run_and_report(args.xlsx, args.product_class, args.days)


if __name__ == "__main__":
    main()
