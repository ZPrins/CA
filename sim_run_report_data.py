import pandas as pd


def build_report_frames(sim, makes=None):
    """
    Build all report dataframes once from simulation data.
    Returns a dict with precomputed DataFrames for CSV and plot generation.
    """
    makes = makes or []
    result = {}
    
    # 1. Inventory DataFrame
    if sim.inventory_snapshots:
        df_inv = pd.DataFrame(sim.inventory_snapshots)
        df_inv["time_h"] = pd.to_numeric(df_inv["time_h"], errors="coerce")
        df_inv = df_inv.dropna(subset=["time_h"])
        df_inv["day"] = pd.to_numeric(df_inv["day"], errors="coerce").fillna(0).astype(int)
        df_inv = df_inv.sort_values(["store_key", "day"])
        df_inv = df_inv.groupby(["store_key", "day"]).last().reset_index()
        df_inv["level"] = df_inv["level"].round(0).astype(int)
        
        try:
            rate_map = getattr(sim, 'demand_rate_map', {}) or {}
        except:
            rate_map = {}
        df_inv["demand_per_day"] = df_inv["store_key"].map(lambda k: float(rate_map.get(str(k), 0.0)) * 24.0).round(1)
        result["df_inv"] = df_inv
    else:
        result["df_inv"] = pd.DataFrame()
    
    # 2. Action Log DataFrame
    if sim.action_log:
        df_log = pd.DataFrame(sim.action_log)
        df_log["time_h"] = pd.to_numeric(df_log["time_h"], errors="coerce")
        df_log["time_d"] = pd.to_numeric(df_log.get("time_d", df_log["time_h"] // 24), errors="coerce").fillna(0).astype(int)
        df_log["day"] = df_log["time_d"].astype(int) + 1
        if "qty_t" not in df_log.columns and "qty" in df_log.columns:
            df_log["qty_t"] = pd.to_numeric(df_log["qty"], errors="coerce").fillna(0.0).astype(float)
        result["df_log"] = df_log
    else:
        result["df_log"] = pd.DataFrame()
    
    # 3. Flows aggregation using new double-entry format
    flows = {}
    if not result["df_log"].empty:
        df_log = result["df_log"]
        
        def aggregate_flow(mask, equipment_type, direction):
            subset = df_log[mask].copy()
            if subset.empty:
                return
            # Use store_key for the new format
            if "store_key" not in subset.columns:
                return
            if "qty_t" not in subset.columns and "qty" in subset.columns:
                subset["qty_t"] = pd.to_numeric(subset["qty"], errors="coerce").fillna(0.0).astype(float)
            subset = subset.dropna(subset=["store_key"])
            if subset.empty:
                return
            grouped = subset.groupby(["store_key", "day"])["qty_t"].sum().reset_index()
            for _, row in grouped.iterrows():
                sk = row["store_key"]
                d = int(row["day"])
                q = abs(row["qty_t"])  # Use absolute value since ConsumeMAT is negative
                if sk not in flows:
                    flows[sk] = {}
                if d not in flows[sk]:
                    flows[sk][d] = {}
                key = f"{equipment_type}_{direction}"
                flows[sk][d][key] = flows[sk][d].get(key, 0) + q
        
        # Train flows - use ConsumeMAT/ReplenishMAT for material tracking
        aggregate_flow((df_log["event"] == "ReplenishMAT") & (df_log["equipment"] == "Train"), "Train", "in")
        aggregate_flow((df_log["event"] == "ConsumeMAT") & (df_log["equipment"] == "Train"), "Train", "out")
        
        # Ship flows
        aggregate_flow((df_log["event"] == "ReplenishMAT") & (df_log["equipment"] == "Ship"), "Ship", "in")
        aggregate_flow((df_log["event"] == "ConsumeMAT") & (df_log["equipment"] == "Ship"), "Ship", "out")
        
        # Production flows - use ReplenishMAT (output) and ConsumeMAT (input)
        aggregate_flow((df_log["event"] == "ReplenishMAT") & (df_log["process"] == "Make"), "Production", "in")
        aggregate_flow((df_log["event"] == "ConsumeMAT") & (df_log["process"] == "Make"), "Consumption", "out")

    result["flows"] = flows
    
    # 4. Downtime aggregation
    downtime_by_equipment = {}
    equipment_to_stores = {}
    
    for make_unit in makes:
        loc = getattr(make_unit, 'location', '')
        equip = getattr(make_unit, 'equipment', '')
        if loc and equip:
            unit_key = f"{loc}|{equip}"
            if unit_key not in equipment_to_stores:
                equipment_to_stores[unit_key] = set()
            candidates = getattr(make_unit, 'candidates', []) or []
            for cand in candidates:
                out_key = getattr(cand, 'out_store_key', None)
                if out_key:
                    equipment_to_stores[unit_key].add(out_key)
                out_keys = getattr(cand, 'out_store_keys', []) or []
                for ok in out_keys:
                    if ok:
                        equipment_to_stores[unit_key].add(ok)
    
    if not result["df_log"].empty:
        df_log = result["df_log"]
        # Use ReplenishMAT events for production-to-store mapping
        production_events = df_log[(df_log["process"] == "Make") & (df_log["event"] == "ReplenishMAT")]
        if not production_events.empty:
            for _, row in production_events.iterrows():
                loc = row.get("location", "")
                equip = row.get("equipment", "")
                store_key = row.get("store_key", "")
                if loc and equip and store_key:
                    unit_key = f"{loc}|{equip}"
                    if unit_key not in equipment_to_stores:
                        equipment_to_stores[unit_key] = set()
                    equipment_to_stores[unit_key].add(store_key)
        
        downtime_events = df_log[df_log["process"] == "Downtime"]
        if not downtime_events.empty:
            for _, row in downtime_events.iterrows():
                loc = row.get("location", "Unknown")
                equip = row.get("equipment", "Unknown")
                d = int(row["day"])
                event_type = row.get("event", "Unknown")
                hours = float(row.get("qty", 0) or 0)
                unit_key = f"{loc}|{equip}"
                if unit_key not in downtime_by_equipment:
                    downtime_by_equipment[unit_key] = {}
                if d not in downtime_by_equipment[unit_key]:
                    downtime_by_equipment[unit_key][d] = {"Maintenance": 0, "Breakdown": 0}

                # Map event types to aggregation keys
                key = event_type
                if event_type in ("Maintenance", "MaintenanceStart"):
                    key = "Maintenance"
                elif event_type in ("Breakdown", "BreakdownStart"):
                    key = "Breakdown"

                if key in downtime_by_equipment[unit_key][d]:
                    # For per-hour events (Maintenance, Breakdown), count 1 hour each
                    # For consolidated events (MaintenanceStart, BreakdownStart), use qty as total hours
                    if event_type in ("MaintenanceStart", "BreakdownStart"):
                        downtime_by_equipment[unit_key][d][key] += hours
                    else:
                        downtime_by_equipment[unit_key][d][key] += 1  # per-hour events count as 1 hour each


    result["downtime_by_equipment"] = downtime_by_equipment
    result["equipment_to_stores"] = equipment_to_stores
    
    store_to_equipment = {}
    for unit_key, stores in equipment_to_stores.items():
        for store_key in stores:
            if store_key not in store_to_equipment:
                store_to_equipment[store_key] = set()
            store_to_equipment[store_key].add(unit_key)
    result["store_to_equipment"] = store_to_equipment
    
    # 5. Store levels and unmet demand
    ending = []
    for key, cont in sim.stores.items():
        ending.append({"Store": key, "Level": round(cont.level, 1), "Capacity": cont.capacity})
    result["df_store_levels"] = pd.DataFrame(ending)
    
    unmet_rows = [{"Key": k, "Unmet": round(float(v), 2)} for k, v in sim.unmet.items()]
    result["df_unmet"] = pd.DataFrame(unmet_rows)
    
    return result
