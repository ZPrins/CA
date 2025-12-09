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

    # Normalize ancillary sheets: enforce the expected columns if present
    make_df = sheets["Make"].copy()
    store_df = sheets["Store"].copy()
    move_df = sheets["Move"].copy()
    deliver_df = sheets["Deliver"].copy()

    # Build key lookups
    # Make sheet
    if not make_df.empty:
        make_df = make_df.rename(columns={
            "Location": "Location", "Equipment Name": "Equipment Name", "Input": "Input",
        })
        make_df["KEY"] = viz._normalize_key_triplet(  # type: ignore[attr-defined]
            make_df.fillna(""), "Location", "Equipment Name", "Input"
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
        # Unify potential name mismatches
        move_df = move_df.rename(columns={
            "Product Class": "Product Class",
            "Location": "Location",
            "Equipment Name": "Equipment Name",
            "Next Location": "Next Location",
        })
        move_df["KEY4"] = viz._normalize_key_quad(  # type: ignore[attr-defined]
            move_df.fillna(""), "Product Class", "Location", "Equipment Name", "Next Location"
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

    # Create Store containers for all nodes whose primary process is Store
    for nid, nd in nodes_by_id.items():
        proc = str(nd.get("process", "")).strip().lower()
        if proc == "store":
            inp = str(nd.get("input", "")).strip()
            k = store_key(nid, inp)
            # Defaults
            capacity = 1e12
            opening = 0.0
            # Lookup Store sheet by (Location, Equipment Name, Input)
            if not store_df.empty:
                # Find matching row (normalized)
                norm_key = viz._normalize_key_triplet(pd.DataFrame([{  # type: ignore[attr-defined]
                    "Location": nd["location"],
                    "Equipment Name": nd["equipment"],
                    "Input": inp,
                }]), "Location", "Equipment Name", "Input").iloc[0]
                match = store_df[store_df["KEY"] == norm_key]
                if not match.empty:
                    m = match.iloc[0]
                    capacity = _to_float(m.get("Silo Max Capacity", capacity), capacity)
                    hi = _to_float(m.get("Silo Opening Stock (High)", opening), 0)
                    lo = _to_float(m.get("Silo Opening Stock (Low)", opening), 0)
                    opening = (hi + lo) / 2.0 if (hi or lo) else opening
            # Sanitize non-negatives
            if capacity is None:
                capacity = 1e12
            capacity = max(float(capacity), 0.0)
            opening = max(float(opening), 0.0)
            # Reconcile opening vs capacity (forgiving with warning)
            ctx = {
                "product_class": str(nd.get("product_class", "")),
                "location": str(nd.get("location", "")),
                "equipment": str(nd.get("equipment", "")),
                "input": inp,
            }
            if capacity <= 0.0 and opening > 0.0:
                _warn("store_capacity_adjusted", "Capacity was <= 0 while opening > 0; set capacity = opening", {**ctx, "orig_capacity": capacity, "orig_opening": opening})
                capacity = opening
            if opening > capacity:
                _warn("store_opening_clamped", "Opening stock exceeded capacity; clamped to capacity", {**ctx, "orig_capacity": capacity, "orig_opening": opening})
                opening = capacity
            stores[k] = simpy.Container(env, init=opening, capacity=capacity)

    # Create Make unit resources and producer processes
    for nid, nd in nodes_by_id.items():
        if str(nd.get("process", "")).strip().lower() != "make":
            continue
        inp = str(nd.get("input", "")).strip()
        outp = str(nd.get("output", "")).strip()
        # Destination store: first route that changes equipment/location or any
        dests = routes.get(nid, [])
        if not dests:
            continue
        dst_id, _edge_out = dests[0]
        # Store keys for input and output
        out_store_key = store_key(dst_id, outp)
        in_store_key = None
        # Try to find a store feeding this Make (same location) for the input
        for s_id, _ in routes.items():
            # Just in case, not strictly necessary now
            pass
        # Look up Make parameters
        mean_rate = 0.0  # tons per hour
        cons_pct = 100.0  # % of output mass drawn as input
        if not make_df.empty:
            norm_key = viz._normalize_key_triplet(pd.DataFrame([{  # type: ignore[attr-defined]
                "Location": nd["location"],
                "Equipment Name": nd["equipment"],
                "Input": inp,
            }]), "Location", "Equipment Name", "Input").iloc[0]
            match = make_df[make_df["KEY"] == norm_key]
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
        if cons_pct < 0 or cons_pct > 100:
            _warn("consumption_percent_out_of_range", "Consumption % outside [0,100]; clamped", {"node": nid, "value": cons_pct})
            cons_pct = max(0.0, min(100.0, cons_pct))
        # Resource representing the unit (capacity 1 by default)
        res = simpy.Resource(env, capacity=1)
        make_units[nid] = res

        def producer(env: simpy.Environment, unit: simpy.Resource, out_store_key_: Key, in_store_key_: Optional[Key], rate_tph: float, cons_pct_: float, step_h: float):
            # Discrete-time production; respects input availability if in_store_key_ provided
            while True:
                qty_out = rate_tph * step_h if rate_tph > 0 else 0.0
                qty_in = qty_out * (cons_pct_ / 100.0) if in_store_key_ else 0.0
                # Optional input withdrawal
                if in_store_key_:
                    src_cont = stores.get(in_store_key_)
                    if src_cont is not None:
                        avail = src_cont.level
                        take = min(avail, qty_in)
                        if take > 0:
                            yield src_cont.get(take)
                        # If not enough input, scale output down proportionally
                        if qty_in > 0 and take < qty_in:
                            scale = take / qty_in if qty_in > 0 else 0.0
                            qty_out *= scale
                # Put output
                dst_cont = stores.get(out_store_key_)
                if dst_cont is not None and qty_out > 0:
                    room = dst_cont.capacity - dst_cont.level if dst_cont.capacity is not None else qty_out
                    put_amt = min(qty_out, room)
                    if put_amt > 0:
                        yield dst_cont.put(put_amt)
                yield env.timeout(step_h)

        # Bind in_store if there is a store at same location with that input
        in_store_key = None
        # Find a store node at same PC+Location with equipment name ending with '_STORE' and input==inp
        for nid2, nd2 in nodes_by_id.items():
            if nd2.get("product_class") != nd.get("product_class"):
                continue
            if str(nd2.get("location")).strip().upper() != str(nd.get("location")).strip().upper():
                continue
            if str(nd2.get("process", "")).strip().lower() == "store" and str(nd2.get("input", "")).strip().upper() == inp.upper():
                in_store_key = store_key(nid2, inp)
                break

        env.process(producer(env, res, out_store_key, in_store_key, mean_rate, cons_pct, settings.step_hours()))

    # Create Move processes: per (PC, Location, Equipment Name, Next Location)
    # Each route spawns #Equipment parallel transporters cycling
    if not move_df.empty:
        # Normalize PC filter if provided
        if product_class:
            move_df = move_df[move_df.get("Product Class", "").astype(str).str.upper() == str(product_class).upper()].copy()
        for _, row in move_df.iterrows():
            pc = str(row.get("Product Class", "")).strip()
            src_loc = str(row.get("Location", "")).strip()
            src_eq = str(row.get("Equipment Name", "")).strip()
            dst_loc = str(row.get("Next Location", "")).strip()
            # Find source store node id for this material: look up in network for Store at src with matching eq
            # We assume the material is the node's input
            # Resolve material from Network: find the row where (pc, loc, eq) is a Store and has output matching the next chain
            material = None
            src_store_node_id = None
            dst_store_node_id = None
            # Heuristic: prefer Store nodes whose equipment matches *_STORE and product class
            for nid, nd in nodes_by_id.items():
                if nd.get("product_class", "").upper() != pc.upper():
                    continue
                if str(nd.get("location", "")).strip().upper() == src_loc.upper() and str(nd.get("equipment", "")).strip().upper() == src_eq.upper():
                    # In Network sample, move edges originate from a Store node
                    if str(nd.get("process", "")).strip().lower() == "store":
                        material = str(nd.get("input", "")).strip()
                        src_store_node_id = nid
                        # Find a destination store at dst_loc holding the same material
                        for nid2, nd2 in nodes_by_id.items():
                            if nd2.get("product_class", "").upper() != pc.upper():
                                continue
                            if str(nd2.get("location", "")).strip().upper() == dst_loc.upper() and str(nd2.get("process", "")).strip().lower() == "store" and str(nd2.get("input", "")).strip().upper() == material.upper():
                                dst_store_node_id = nid2
                                break
                        break
            if src_store_node_id is None or dst_store_node_id is None or material is None:
                continue

            origin_key = store_key(src_store_node_id, material)
            dest_key = store_key(dst_store_node_id, material)

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
            ctx = {"pc": pc, "src_loc": src_loc, "src_eq": src_eq, "dst_loc": dst_loc, "material": material}
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

            rkey = f"{pc.upper()}|{src_loc.upper()}|{src_eq.upper()}→{dst_loc.upper()}|{material.upper()}"
            route_stats.setdefault(rkey, RouteStats())

            def transporter(env: simpy.Environment, origin: simpy.Container, dest: simpy.Container, payload_t: float, load_rate_tph: float, unload_rate_tph: float, to_min: float, back_min: float, stats: RouteStats):
                while True:
                    # Load
                    load_needed = payload_t
                    if load_rate_tph <= 0 or payload_t <= 0:
                        # Nothing to do
                        yield env.timeout((to_min + unload_rate_tph + back_min) / 60.0 if to_min or back_min else step_h)
                        continue
                    # Load at rate, constrained by origin stock and dest capacity at unload stage
                    load_time_h = payload_t / max(load_rate_tph, 1e-9)
                    # Withdraw from origin at the end of loading period to keep it simple
                    take = min(origin.level, payload_t)
                    if take > 0:
                        yield env.timeout(load_time_h)
                        yield origin.get(take)
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
                    else:
                        # No room: return without unloading and try later
                        yield env.timeout(step_h)
                    # Travel back
                    yield env.timeout(back_min / 60.0)

            origin = stores.get(origin_key)
            dest = stores.get(dest_key)
            if origin is None or dest is None:
                continue
            for i in range(max(1, n_units)):
                env.process(transporter(env, origin, dest, payload, load_rate, unload_rate, travel_to_min, travel_back_min, route_stats[rkey]))

    # Deliver processes: consume per location and input
    if not deliver_df.empty:
        # Aggregate demand per (PC, Location, Input)
        # We assume 'Demand per Location' is per day; scale to step size
        agg: Dict[Tuple[str, str, str], float] = {}
        loc_col = None
        dem_col = None
        for c in deliver_df.columns:
            lc = str(c).strip().lower()
            if lc == "location":
                loc_col = c
            if lc.startswith("demand"):
                dem_col = c
        if loc_col and dem_col:
            for _, r in deliver_df.iterrows():
                loc = str(r[loc_col]).strip()
                # Try to deduce input from nearby Store node for this PC
                # Fallback: use 'Input' column if present
                inp_val = str(r.get("Input", "")).strip()
                if not inp_val:
                    # find any store at this location and product class
                    for nid, nd in nodes_by_id.items():
                        if product_class and nd.get("product_class", "").upper() != str(product_class).upper():
                            continue
                        if str(nd.get("location", "")).strip().upper() == loc.upper() and str(nd.get("process", "")).strip().lower() == "store":
                            inp_val = str(nd.get("input", "")).strip()
                            break
                pc_val = str(product_class or next(iter({nd.get('product_class', '') for nd in nodes_by_id.values()}), '')).strip()
                key = (pc_val.upper(), loc.upper(), inp_val.upper())
                try:
                    agg[key] = agg.get(key, 0.0) + float(r[dem_col] or 0)
                except Exception:
                    pass

            step_h = settings.step_hours()
            step_days = step_h / 24.0
            for (pc_val, loc, inp_val), demand_per_day in agg.items():
                # Need a store at this location and input
                # Find store node id
                sid = None
                for nid, nd in nodes_by_id.items():
                    if nd.get("product_class", "").upper() != pc_val:
                        continue
                    if str(nd.get("location", "")).strip().upper() == loc and str(nd.get("process", "")).strip().lower() == "store" and str(nd.get("input", "")).strip().upper() == inp_val:
                        sid = nid
                        break
                if sid is None:
                    continue
                skey = store_key(sid, inp_val)
                cont = stores.get(skey)
                if cont is None:
                    continue
                rate_per_step = demand_per_day * step_days
                ukey = f"{pc_val}|{loc}|{inp_val}"
                unmet_demand.setdefault(ukey, 0.0)

                def consumer(env: simpy.Environment, cont: simpy.Container, rate: float, acc_key: str):
                    while True:
                        take = min(cont.level, rate)
                        if take > 0:
                            yield cont.get(take)
                        short = rate - take
                        if short > 0:
                            unmet_demand[acc_key] = unmet_demand.get(acc_key, 0.0) + short
                        yield env.timeout(step_h)

                env.process(consumer(env, cont, rate_per_step, ukey))

    components = Components(env=env, stores=stores, make_units=make_units, unmet_demand=unmet_demand, route_stats=route_stats)
    meta = {"settings": settings, "warnings": warnings}
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
