# data_loader.py
import pandas as pd
from io import BytesIO
from sim_run_grok_core import StoreConfig, ProductionCandidate, MakeUnit, TransportRoute, Demand
from sim_run_grok_config import config

EXCEL_FILE = str(config.excel_file)

def load_data():
    def read_sheet(sheet_name: str):
        try:
            return pd.read_excel(EXCEL_FILE, sheet_name=sheet_name)
        except PermissionError as e:
            # Clear guidance when the workbook is locked by Excel or another process
            raise RuntimeError(
                f"Unable to read '{EXCEL_FILE}' (sheet '{sheet_name}') because the file is locked. "
                "Please close the workbook in Excel (or any other app) and run the simulation again."
            ) from e
        except OSError as e:
            # Other OS errors: present a concise message
            raise RuntimeError(
                f"Unable to read '{EXCEL_FILE}' (sheet '{sheet_name}'): {e}"
            ) from e

    store_df = read_sheet("Store")
    make_df = read_sheet("Make")
    # Prefer explicit modes: Move_TRAIN and Move_SHIP (be tolerant to naming variants)
    def _try_read_any(names: list[str]):
        for nm in names:
            try:
                df = read_sheet(nm)
                if df is not None:
                    return df
            except Exception:
                pass
        return None

    move_train_df = _try_read_any([
        "Move_TRAIN", "MOVE_TRAIN", "Move-TRAIN", "Move Train", "Train", "TRAINS", "MOVE TRAIN"
    ])
    move_ship_df = _try_read_any([
        "Move_SHIP", "MOVE_SHIP", "Move-SHIP", "Move Ship", "Ship", "SHIPS", "MOVE SHIP"
    ])
    ship_berths_df = _try_read_any(["SHIP_BERTHS", "Ship_Berths", "Ship Berths"])
    ship_routes_df = _try_read_any(["SHIP_ROUTES", "Ship_Routes", "Ship Routes"])
    ship_dist_df = _try_read_any(["SHIP_DISTANCES", "Ship_Distances", "Ship Distances", "SHIP_DISTANCE"]) 
    # Legacy Move sheet (kept for backward compatibility, but not required)
    move_df_legacy = _try_read_any(["Move", "MOVE"])
    deliver_df = read_sheet("Deliver")
    network_df = read_sheet("Network")

    settings = {
        "horizon_days": 365,
        "random_opening": True,
        "random_seed": None,
    }

    def _num(x, default=0.0):
        try:
            return default if pd.isna(x) else float(x)
        except Exception:
            return default

    # Stores
    store_lookup = {}
    stores = []
    for _, row in store_df.iterrows():
        loc = row["Location"]
        eq = row["Equipment Name"]
        prod = row["Input"]
        if pd.isna(prod):
            continue
        key = f"{prod}|{loc}|{eq}|{prod}"
        store_lookup[(loc, eq, prod)] = key
        cap = 0.0 if pd.isna(row["Silo Max Capacity"]) else float(row["Silo Max Capacity"]) 
        open_low = 0.0 if pd.isna(row["Silo Opening Stock (Low)"]) else float(row["Silo Opening Stock (Low)"])
        open_high = 0.0 if pd.isna(row["Silo Opening Stock (High)"]) else float(row["Silo Opening Stock (High)"])
        # Clamp openings between 0 and capacity
        open_low = max(0.0, min(open_low, cap))
        open_high = max(0.0, min(open_high, cap))
        stores.append(StoreConfig(
            key=key,
            capacity=cap,
            opening_low=open_low,
            opening_high=open_high
        ))

    # Makes - with proper multi-store output
    makes = []
    for _, make_row in make_df.iterrows():
        loc = make_row["Location"]
        eq = make_row["Equipment Name"]
        inp = make_row["Input"] if pd.notna(make_row["Input"]) else None
        outp = make_row["Output"]

        in_key = store_lookup.get((loc, eq, inp)) if inp else None

        # Find output stores from Network
        output_candidates = []
        make_rows = network_df[
            (network_df["Process"] == "Make") &
            (network_df["Location"] == loc) &
            (network_df["Equipment Name"] == eq) &
            (network_df["Output"] == outp)
        ]
        for _, nr in make_rows.iterrows():
            next_eq = nr["Next Equipment"]
            next_loc = nr["Next Location"]
            key = store_lookup.get((next_loc, next_eq, outp))
            if key:
                output_candidates.append(key)

        if not output_candidates:
            for (sl, se, sp) in store_lookup:
                if sl == loc and sp == outp:
                    output_candidates.append(store_lookup[(sl, se, sp)])
            output_candidates = list(set(output_candidates))

        def _num(x, default=0.0):
            try:
                return default if pd.isna(x) else float(x)
            except Exception:
                return default
        candidates = []
        for out_key in output_candidates:
            candidates.append(ProductionCandidate(
                product=outp,
                out_store_key=out_key,
                in_store_key=in_key,
                rate_tph=_num(make_row.get("Mean Production Rate (Tons/hr)"), 0.0),
                consumption_pct=_num(make_row.get("Consumption %", 1.0), 1.0)
            ))

        makes.append(MakeUnit(
            location=loc,
            equipment=eq,
            candidates=candidates
        ))

    # Build per-store load/unload rate lookup from Store sheet
    store_rates = {}
    for _, srow in store_df.iterrows():
        try:
            loc = srow["Location"]
            eq = srow["Equipment Name"]
            inp = srow.get("Input")
            key = store_lookup.get((loc, eq, inp))
            if key:
                load_r = _num(srow.get("Load Rate (ton/hr)"), None)
                unload_r = _num(srow.get("Unload Rate (ton/hr)"), None)
                store_rates[key] = (load_r, unload_r)
        except Exception:
            pass
    # Expose store rates for movers (e.g., SHIP) via settings
    settings["store_rates"] = store_rates

    # Moves - prefer Move_TRAIN if available, else fall back to legacy Move
    moves = []

    # Diagnostics counters
    diag = {
        'TRAIN': {
            'rows': 0, 'built': 0,
            'sk_missing_required': 0, 'sk_no_origin': 0, 'sk_no_dest': 0,
            'used_explicit_store': 0, 'fallback_no_explicit': 0, 'fallback_invalid_explicit': 0,
        },
        'SHIP': {
            'rows': 0, 'built': 0,
            'sk_missing_required': 0, 'sk_no_origin': 0, 'sk_no_dest': 0,
            'used_explicit_store': 0, 'fallback_no_explicit': 0, 'fallback_invalid_explicit': 0,
        }
    }

    def _first_non_na(row_obj, names: list[str], default=None):
        for nm in names:
            try:
                val = row_obj.get(nm)
            except Exception:
                val = None
            if val is not None and not pd.isna(val):
                try:
                    return float(val)
                except Exception:
                    try:
                        return float(str(val).strip())
                    except Exception:
                        pass
        return default

    def _pick_keys_by_explicit(origin_loc, product, explicit_store):
        # If explicit store provided, try to match equipment name first (case-insensitive)
        keys = []
        exp = str(explicit_store).strip().upper() if explicit_store else ""
        for (loc, eq, prod), store_key in store_lookup.items():
            if loc == origin_loc and prod == product:
                if exp and str(eq).strip().upper() != exp:
                    continue
                keys.append(store_key)
        return keys

    if move_train_df is not None and not getattr(move_train_df, 'empty', True):
        for _, row in move_train_df.iterrows():
            diag['TRAIN']['rows'] += 1
            product = str(row.get("Product", "")).strip()
            origin_loc = str(row.get("Origin Location", "")).strip()
            dest_loc = str(row.get("Destination Location", "")).strip()
            if not product or not origin_loc or not dest_loc:
                diag['TRAIN']['sk_missing_required'] += 1
                continue
            origin_store_name = str(row.get("Origin Store", "")).strip()
            dest_store_name = str(row.get("Destination Store", "")).strip()
            origin_stores = _pick_keys_by_explicit(origin_loc, product, origin_store_name)
            dest_stores = _pick_keys_by_explicit(dest_loc, product, dest_store_name)
            if origin_store_name:
                if origin_stores:
                    diag['TRAIN']['used_explicit_store'] += 1
                else:
                    diag['TRAIN']['fallback_invalid_explicit'] += 1
            else:
                diag['TRAIN']['fallback_no_explicit'] += 1
            if dest_store_name:
                if dest_stores:
                    diag['TRAIN']['used_explicit_store'] += 1
                else:
                    diag['TRAIN']['fallback_invalid_explicit'] += 1
            else:
                diag['TRAIN']['fallback_no_explicit'] += 1
            if not origin_stores:
                origin_stores = [store_lookup[(loc, eq, prod)] for (loc, eq, prod) in store_lookup if loc == origin_loc and prod == product]
            if not dest_stores:
                dest_stores = [store_lookup[(loc, eq, prod)] for (loc, eq, prod) in store_lookup if loc == dest_loc and prod == product]
            if not origin_stores:
                diag['TRAIN']['sk_no_origin'] += 1
                continue
            if not dest_stores:
                diag['TRAIN']['sk_no_dest'] += 1
                continue
            # Compute payload and times
            try:
                n_units = int(_first_non_na(row, ["# Trains", "# Train", "# Units", "Units"]) or 0)
            except Exception:
                n_units = 0
            carriages = _first_non_na(row, ["# Carraiges", "# Carriages"], 0) or 0
            carr_cap = _first_non_na(row, ["# Carraige Capacity (ton)", "# Carriage Capacity (ton)", "Carriage Capacity (ton)"], 0) or 0
            payload = float(carriages * carr_cap)
            distance = _first_non_na(row, ["Distance", "Distance (km)", "Dist (km)"], 0.0) or 0.0
            v_loaded = _first_non_na(row, ["Avg Speed - Loaded (km/hr)", "Avg Speed Loaded (km/hr)", "Avg Speed - Loaded (km/h)"], 0.0) or 0.0
            v_empty = _first_non_na(row, ["Avg Speed - Empty (km/hr)", "Avg Speed Empty (km/hr)", "Avg Speed - Empty (km/h)"], 0.0) or 0.0
            to_min = (distance / v_loaded * 60.0) if v_loaded > 0 else 0.0
            back_min = (distance / v_empty * 60.0) if v_empty > 0 else 0.0
            # Rates
            def _get_rate(row_obj, candidates, default):
                v = _first_non_na(row_obj, candidates, None)
                return float(v) if v is not None and v > 0 else float(default)
            load_rate = _get_rate(row, ["Load Rate (ton/hr)", "Load Rate (Ton/hr)", "Load Rate"], 500.0)
            unload_rate = _get_rate(row, ["Unload Rate (ton/hr)", "Unload Rate (Ton/hr)", "Unload Rate"], 400.0)
            moves.append(TransportRoute(
                product=product,
                origin_location=origin_loc,
                dest_location=dest_loc,
                origin_stores=origin_stores,
                dest_stores=dest_stores,
                n_units=n_units if n_units > 0 else 0,
                payload_t=max(0.0, payload),
                load_rate_tph=max(0.0, float(load_rate)),
                unload_rate_tph=max(0.0, float(unload_rate)),
                to_min=max(0.0, to_min),
                back_min=max(0.0, back_min),
                mode="TRAIN",
            ))
            diag['TRAIN']['built'] += 1
    # SHIP multi-sheet ingestion: prefer SHIP_ROUTES + SHIP_DISTANCES + Move_SHIP
    if ship_routes_df is not None and not getattr(ship_routes_df, 'empty', True):
        # Build berth info
        berth_info = {}
        if ship_berths_df is not None and not getattr(ship_berths_df, 'empty', True):
            for _, brow in ship_berths_df.iterrows():
                try:
                    loc = str(brow.get("Location", "")).strip()
                    if not loc:
                        continue
                    berths = int(_num(brow.get("# Berths"), 0))
                    pocc = float(_num(brow.get("Probability Berth Occupied %"), 0.0))
                    if pocc > 1.0:
                        pocc = pocc / 100.0
                    pilot_in = float(_num(brow.get("Pilot In (Hours)"), 0.0))
                    pilot_out = float(_num(brow.get("Pilot Out (Hours)"), 0.0))
                    berth_info[loc] = {"berths": berths, "p_occupied": pocc, "pilot_in_h": pilot_in, "pilot_out_h": pilot_out}
                except Exception:
                    pass
        # Build distance map (nM)
        nm_distance = {}
        if ship_dist_df is not None and not getattr(ship_dist_df, 'empty', True):
            for _, drow in ship_dist_df.iterrows():
                try:
                    a = str(drow.get("Location 1", "")).strip()
                    b = str(drow.get("Location 2", "")).strip()
                    if not a or not b:
                        continue
                    dist = float(_num(drow.get("Distance (nM)"), 0.0))
                    nm_distance[(a, b)] = dist
                    nm_distance[(b, a)] = dist
                except Exception:
                    pass
        # Build route groups from Move_SHIP
        route_groups = {}
        if move_ship_df is not None and not getattr(move_ship_df, 'empty', True):
            for _, mrow in move_ship_df.iterrows():
                try:
                    origin_loc = str(mrow.get("Origin Location", "")).strip()
                    rg = str(mrow.get("Route Group", "")).strip()
                    if not rg:
                        continue
                    vessels = int(_num(mrow.get("# Vessels"), 0))
                    speed_knots = float(_num(mrow.get("Route avg Speed (knots)"), 0.0))
                    hulls = int(_num(mrow.get("#Hulls"), 0))
                    payload_per_hull = float(_num(mrow.get("Payload per Hull"), 0.0))
                    ent = route_groups.setdefault(rg, {"n_vessels": 0, "speed_knots": speed_knots, "hulls": hulls, "payload_per_hull": payload_per_hull, "origins": set()})
                    ent["n_vessels"] += vessels
                    if speed_knots > 0:
                        ent["speed_knots"] = speed_knots
                    if hulls > 0:
                        ent["hulls"] = hulls
                    if payload_per_hull > 0:
                        ent["payload_per_hull"] = payload_per_hull
                    if origin_loc:
                        ent["origins"].add(origin_loc)
                except Exception:
                    pass
        # Parse SHIP_ROUTES wide format into itineraries per route group
        itineraries_by_group = {}
        try:
            # Expect first column named 'Field'
            if 'Field' in ship_routes_df.columns:
                field_col = 'Field'
            else:
                field_col = ship_routes_df.columns[0]
            # Normalize rows: map field name -> row index
            rows = {str(ship_routes_df.iloc[i][field_col]).strip(): i for i in range(len(ship_routes_df))}
            # Identify all route columns (skip first col)
            route_cols = [c for c in ship_routes_df.columns if c != field_col]
            for col in route_cols:
                # Read basic fields
                def gv(field_name):
                    idx = rows.get(field_name)
                    if idx is None:
                        return None
                    try:
                        v = ship_routes_df.at[idx, col]
                        return None if pd.isna(v) else str(v).strip()
                    except Exception:
                        return None
                rg = gv('Route Group')
                route_id = gv('Route ID')
                origin = gv('Origin Location')
                return_loc = gv('Return Location')
                if not rg or not origin or not return_loc:
                    continue
                steps = []
                # Start marker for origin tracking
                steps.append({"kind": "start", "location": origin, "route_id": route_id})
                # Helper to append load/unload lines at current location
                def append_load_unload(prefix: str, loc_for_step: str):
                    # prefix is 'Product' for origin block, or can still be 'Product' in sheet
                    # Look for numbered rows Product 1 Load, Product 2 Load, ... and corresponding Stores
                    k = 1
                    while True:
                        load_label = f"Product {k} Load"
                        store_label = f"Product {k} Store"
                        unload_label = f"Product {k} Unload"
                        unload_store_label = f"Product {k} Unload Store"
                        # try to read all four safely
                        prod_load = gv(load_label)
                        prod_store = gv(store_label)
                        prod_unload = gv(unload_label)
                        prod_unload_store = gv(unload_store_label)
                        if prod_load is None and prod_unload is None:
                            # Stop if both are missing for this k; advance k until a gap of 2 encountered
                            # But to avoid infinite loops, break after k grows too large
                            if k > 10:
                                break
                        if prod_load:
                            # resolve store key at this loc
                            store_key = None
                            for (l, eq, p), sk in store_lookup.items():
                                if str(l).strip() == loc_for_step and str(p).strip() == prod_load and str(eq).strip() == (prod_store or '').strip():
                                    store_key = sk
                                    break
                            steps.append({"kind": "load", "location": loc_for_step, "product": prod_load, "store_eq": prod_store, "store_key": store_key, "route_id": route_id})
                        if prod_unload:
                            store_key = None
                            for (l, eq, p), sk in store_lookup.items():
                                if str(l).strip() == loc_for_step and str(p).strip() == prod_unload and str(eq).strip() == (prod_unload_store or '').strip():
                                    store_key = sk
                                    break
                            steps.append({"kind": "unload", "location": loc_for_step, "product": prod_unload, "store_eq": prod_unload_store, "store_key": store_key, "route_id": route_id})
                        k += 1
                # Origin loads (any Product i Load / Store near top of the column)
                append_load_unload('Product', origin)
                # Destinations: iterate Destination k Location rows in increasing k until missing
                dest_idx = 1
                current = origin
                while True:
                    dest_label = f"Destination {dest_idx} Location"
                    dest = gv(dest_label)
                    if not dest:
                        break
                    steps.append({"kind": "sail", "from": current, "to": dest, "route_id": route_id})
                    current = dest
                    # At this destination, check unload/load lines (mid-journey loads allowed)
                    append_load_unload('Product', current)
                    dest_idx += 1
                # Return leg
                steps.append({"kind": "sail", "from": current, "to": return_loc, "route_id": route_id})
                itineraries_by_group.setdefault(rg, []).append(steps)
        except Exception:
            pass
        # Build SHIP TransportRoute entries (one per Route Group)
        for rg, meta in route_groups.items():
            its = itineraries_by_group.get(rg, [])
            n_units = int(meta.get("n_vessels", 0) or 0)
            hulls = int(meta.get("hulls", 0) or 0)
            payload_per_hull = float(meta.get("payload_per_hull", 0.0) or 0.0)
            payload_t = float(hulls * payload_per_hull)
            speed_knots = float(meta.get("speed_knots", 0.0) or 0.0)
            # Choose a representative origin for compatibility fields
            origin_locs = list(meta.get("origins", []) or [])
            origin_loc = origin_locs[0] if origin_locs else (its[0][0]["location"] if its and its[0] and its[0][0].get("kind") == "start" else "")
            # Emit route (TRAIN fields are placeholders for SHIP)
            moves.append(TransportRoute(
                product="",
                origin_location=origin_loc,
                dest_location=origin_loc,
                origin_stores=[],
                dest_stores=[],
                n_units=n_units,
                payload_t=payload_t,
                load_rate_tph=500.0,
                unload_rate_tph=400.0,
                to_min=0.0,
                back_min=0.0,
                mode="SHIP",
                route_group=rg,
                speed_knots=speed_knots,
                hulls_per_vessel=hulls,
                payload_per_hull_t=payload_per_hull,
                itineraries=its,
                berth_info=berth_info,
                nm_distance=nm_distance,
            ))
            diag['SHIP']['built'] += 1
    elif move_ship_df is not None and not getattr(move_ship_df, 'empty', True):
        # Fallback simple SHIP ingestion from Move_SHIP only (legacy mirror of TRAIN)
        for _, row in move_ship_df.iterrows():
            diag['SHIP']['rows'] += 1
            # Minimal parsing: treat as direct OD akin to TRAIN if columns exist
            origin_loc = str(row.get("Origin Location", "")).strip()
            dest_loc = str(row.get("Destination Location", "")).strip()
            if not origin_loc or not dest_loc:
                diag['SHIP']['sk_missing_required'] += 1
                continue
            # Without explicit product, we cannot resolve stores sensibly; leave empty
            try:
                n_units = int(_first_non_na(row, ["# Ships", "# Ship", "# Vessels", "# Units", "Units"]) or 0)
            except Exception:
                n_units = 0
            hulls = int(_num(row.get("#Hulls"), 0))
            payload_per_hull = float(_num(row.get("Payload per Hull"), 0.0))
            payload = float(hulls * payload_per_hull)
            moves.append(TransportRoute(
                product="",
                origin_location=origin_loc,
                dest_location=dest_loc,
                origin_stores=[],
                dest_stores=[],
                n_units=n_units,
                payload_t=max(0.0, float(payload)),
                load_rate_tph=500.0,
                unload_rate_tph=400.0,
                to_min=0.0,
                back_min=0.0,
                mode="SHIP",
            ))
            diag['SHIP']['built'] += 1
    elif move_df_legacy is not None and not getattr(move_df_legacy, 'empty', True):
        for _, row in move_df_legacy.iterrows():
            product = row.get("Product")
            origin_loc = row.get("Location")
            dest_loc = row.get("Next Location")

            origin_stores = [store_lookup[(loc, eq, prod)] for (loc, eq, prod) in store_lookup if loc == origin_loc and prod == product]
            dest_stores = [store_lookup[(loc, eq, prod)] for (loc, eq, prod) in store_lookup if loc == dest_loc and prod == product]

            if not origin_stores or not dest_stores:
                print(f"Warning: Incomplete route {product} {origin_loc} → {dest_loc}")
                continue

            parcels = 0.0 if pd.isna(row.get("#Parcels")) else float(row.get("#Parcels"))
            cap_per_parcel = 0.0 if pd.isna(row.get("Capacity Per Parcel")) else float(row.get("Capacity Per Parcel"))
            payload = parcels * cap_per_parcel
            n_units = 99 if pd.isna(row.get("#Equipment \n(99-unlimited)")) or row.get("#Equipment \n(99-unlimited)") >= 99 else int(row.get("#Equipment \n(99-unlimited)"))

            moves.append(TransportRoute(
                product=product,
                origin_location=origin_loc,
                dest_location=dest_loc,
                origin_stores=origin_stores,
                dest_stores=dest_stores,
                n_units=n_units,
                payload_t=float(payload),
                load_rate_tph=_num(row.get("Load Rate (Ton/hr)"), 500.0),
                unload_rate_tph=_num(row.get("Unload Rate (Ton/Hr)"), 400.0),
                to_min=_num(row.get("Travel to Time (Min)"), 0.0),
                back_min=_num(row.get("Travel back Time (Min)"), 0.0)
            ))

    # Demands — use annual demand from Deliver sheet; compute per-hour using config, round to 2 decimals
    demands = []
    hours_per_year = 365 * 24.0
    step_h = float(getattr(config, 'demand_step_hours', 1.0))

    # Counters for reporting
    used_explicit_store = 0
    fallback_no_explicit = 0
    fallback_invalid_explicit = 0

    # Build fast index of available stores by (loc, prod) and by (loc, eq, prod)
    stores_by_loc_prod = {}
    for (l, e, p), key in store_lookup.items():
        stores_by_loc_prod.setdefault((str(l), str(p)), []).append((e, key))

    for _, row in deliver_df.iterrows():
        loc = row.get("Location")
        prod = row.get("Input")
        if pd.isna(loc) or pd.isna(prod):
            continue
        loc_s = str(loc).strip()
        prod_s = str(prod).strip()
        candidates = stores_by_loc_prod.get((loc_s, prod_s), [])
        if not candidates:
            continue

        # Prefer an explicit Demand Store match when provided
        demand_store_name = None
        if "Demand Store" in deliver_df.columns:
            val = row.get("Demand Store")
            if pd.notna(val):
                s = str(val).strip()
                demand_store_name = s if s != "" else None

        chosen_key = None
        if demand_store_name:
            ds_upper = demand_store_name.upper()
            for eq, key in candidates:
                if str(eq).strip().upper() == ds_upper:
                    chosen_key = key
                    used_explicit_store += 1
                    break
            if chosen_key is None:
                # Hard error: explicit Demand Store not found for this (Location, Input)
                available = ", ".join(sorted(str(eq) for eq, _ in candidates)) or "<none>"
                raise ValueError(
                    f"Deliver row specifies Demand Store='{demand_store_name}' not found at Location='{loc_s}', Input='{prod_s}'. "
                    f"Available stores for this pair: {available}"
                )
        if chosen_key is None:
            # Deterministic fallback only when no explicit Demand Store was provided
            candidates_sorted = sorted(candidates, key=lambda t: str(t[0]).upper())
            chosen_key = candidates_sorted[0][1]
            if not demand_store_name:
                fallback_no_explicit += 1

        # Prefer explicit annual demand columns; fallback to legacy hourly column
        annual = None
        for col in [
            "Annual Demand (Tons)",
            "Annual Demand",
            "Annual Demand (Ton)",
            "Annual_Demand_Tons",
        ]:
            if col in deliver_df.columns and pd.notna(row.get(col, None)):
                try:
                    annual = float(row[col])
                    break
                except Exception:
                    pass
        if annual is None:
            # Legacy hourly fallback for backward compatibility
            hourly_raw = row.get("Demand per Location/Hour", 0.0)
            try:
                hourly = float(hourly_raw or 0.0)
            except Exception:
                hourly = 0.0
            annual = hourly * hours_per_year
        # Compute per-hour rate from annual; consumer will scale by step length
        rate_per_hour = round(float(annual) / hours_per_year, 2)
        demands.append(Demand(store_key=chosen_key, rate_per_hour=rate_per_hour))

    # Summary logs
    try:
        # Move diagnostics
        tr = diag.get('TRAIN', {})
        sh = diag.get('SHIP', {})
        print(
            "Move parsing: "
            f"TRAIN rows={tr.get('rows',0)}, built={tr.get('built',0)}, "
            f"sk_missing_required={tr.get('sk_missing_required',0)}, sk_no_origin={tr.get('sk_no_origin',0)}, sk_no_dest={tr.get('sk_no_dest',0)}; "
            f"SHIP rows={sh.get('rows',0)}, built={sh.get('built',0)}, "
            f"sk_missing_required={sh.get('sk_missing_required',0)}, sk_no_origin={sh.get('sk_no_origin',0)}, sk_no_dest={sh.get('sk_no_dest',0)}",
            flush=True,
        )
    except Exception:
        pass

    try:
        print(f"Deliver mapping: used_explicit_store={used_explicit_store}, fallback_no_explicit={fallback_no_explicit}, fallback_invalid_explicit={fallback_invalid_explicit}")
    except Exception:
        pass

    return settings, stores, makes, moves, demands