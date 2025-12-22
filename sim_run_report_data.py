import pandas as pd
import duckdb


def build_report_frames(sim, makes=None):
    """
    Build all report dataframes once from simulation data.
    Returns a dict with precomputed DataFrames for CSV and plot generation.
    Uses duckdb for high-performance aggregations.
    """
    makes = makes or []
    result = {}
    
    # 1. Inventory DataFrame
    if sim.inventory_snapshots:
        # Use tuples directly for fast DataFrame creation
        cols_inv = ["day", "time_h", "product_class", "location", "equipment", "input", "store_key", "level", "capacity", "fill_pct"]
        df_snapshots = pd.DataFrame.from_records(sim.inventory_snapshots, columns=cols_inv)
        
        # Use duckdb to process snapshots quickly
        df_inv = duckdb.query("""
            SELECT 
                store_key, 
                CAST(day AS INTEGER) as day,
                LAST(level) as level,
                LAST(capacity) as capacity,
                LAST(time_h) as time_h
            FROM df_snapshots
            WHERE time_h IS NOT NULL
            GROUP BY store_key, day
            ORDER BY store_key, day
        """).df()
        
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
        cols_log = [
            "day", "time_h", "time_d", "process", "event", "location", "equipment", "product", 
            "qty", "time", "unmet_demand", "qty_out", "from_store", "from_level", "from_fill_pct",
            "qty_in", "to_store", "to_level", "to_fill_pct", "route_id", "vessel_id", "ship_state"
        ]
        df_log = pd.DataFrame.from_records(sim.action_log, columns=cols_log)
        
        # Ensure qty_t is available for flows
        df_log["qty_t"] = pd.to_numeric(df_log["qty"], errors='coerce').fillna(0.0)
        # Ensure time_t is available for durations
        df_log["time_t"] = pd.to_numeric(df_log["time"], errors='coerce').fillna(0.0)
        
        # Collapse consecutive identical Ship log entries (e.g., waiting lines)
        df_log = _collapse_ship_log(df_log)
        
        # Collapse Truck log entries by time_d, location, product
        df_log = _collapse_truck_log(df_log)
        
        result["df_log"] = df_log
    else:
        result["df_log"] = pd.DataFrame()
    
    # 3. Flows aggregation
    flows = {}
    if not result["df_log"].empty:
        df_log = result["df_log"]
        
        # Use DuckDB to aggregate all flows at once
        flow_agg = duckdb.query("""
            SELECT 'Train' as equip, 'in' as dir, to_store as sk, day, SUM(qty_t) as q FROM df_log WHERE event = 'Unload' AND equipment = 'Train' AND to_store IS NOT NULL GROUP BY 1,2,3,4
            UNION ALL
            SELECT 'Ship' as equip, 'in' as dir, to_store as sk, day, SUM(qty_t) as q FROM df_log WHERE event IN ('Unload', 'ShipUnload') AND equipment = 'Ship' AND to_store IS NOT NULL GROUP BY 1,2,3,4
            UNION ALL
            SELECT 'Train' as equip, 'out' as dir, from_store as sk, day, SUM(qty_t) as q FROM df_log WHERE event = 'Load' AND equipment = 'Train' AND from_store IS NOT NULL GROUP BY 1,2,3,4
            UNION ALL
            SELECT 'Ship' as equip, 'out' as dir, from_store as sk, day, SUM(qty_t) as q FROM df_log WHERE event IN ('Load', 'ShipLoad') AND equipment = 'Ship' AND from_store IS NOT NULL GROUP BY 1,2,3,4
            UNION ALL
            SELECT 'Production' as equip, 'in' as dir, to_store as sk, day, SUM(qty_t) as q FROM df_log WHERE event IN ('Produce', 'ProducePartial') AND to_store IS NOT NULL GROUP BY 1,2,3,4
            UNION ALL
            SELECT 'Consumption' as equip, 'out' as dir, from_store as sk, day, SUM(qty_t) as q FROM df_log WHERE event IN ('Produce', 'ProducePartial') AND from_store IS NOT NULL GROUP BY 1,2,3,4
        """).df()
        
        for _, row in flow_agg.iterrows():
            sk, d, q, equip, direction = row['sk'], int(row['day']), row['q'], row['equip'], row['dir']
            if sk not in flows: flows[sk] = {}
            if d not in flows[sk]: flows[sk][d] = {}
            flows[sk][d][f"{equip}_{direction}"] = q

    result["flows"] = flows
    
    # 4. Downtime aggregation
    downtime_by_equipment = {}
    equipment_to_stores = {}
    
    for make_unit in makes:
        loc, equip = getattr(make_unit, 'location', ''), getattr(make_unit, 'equipment', '')
        if loc and equip:
            unit_key = f"{loc}|{equip}"
            if unit_key not in equipment_to_stores: equipment_to_stores[unit_key] = set()
            for cand in (getattr(make_unit, 'candidates', []) or []):
                for ok in ([getattr(cand, 'out_store_key', None)] + (getattr(cand, 'out_store_keys', []) or [])):
                    if ok: equipment_to_stores[unit_key].add(ok)
    
    if not result["df_log"].empty:
        df_log = result["df_log"]
        # Fast equipment_to_stores update from log
        prod_map = duckdb.query("""
            SELECT DISTINCT location, equipment, to_store 
            FROM df_log 
            WHERE process = 'Make' AND event IN ('Produce', 'ProducePartial') 
              AND location IS NOT NULL AND equipment IS NOT NULL AND to_store IS NOT NULL
        """).df()
        for _, row in prod_map.iterrows():
            uk = f"{row['location']}|{row['equipment']}"
            if uk not in equipment_to_stores: equipment_to_stores[uk] = set()
            equipment_to_stores[uk].add(row['to_store'])

        # Aggregated downtime
        dt_agg = duckdb.query("""
            SELECT 
                location, equipment, day,
                SUM(CASE WHEN event IN ('Maintenance', 'MaintenanceStart') THEN COALESCE(time_t, 0.0) ELSE 0 END) as Maintenance,
                SUM(CASE WHEN event IN ('Breakdown', 'BreakdownStart') THEN COALESCE(time_t, 0.0) ELSE 0 END) as Breakdown,
                SUM(CASE WHEN event = 'ResourceWait' THEN COALESCE(time_t, 0.0) ELSE 0 END) as ResourceWait,
                SUM(CASE WHEN event = 'ProduceBlocked' THEN COALESCE(time_t, 0.0) ELSE 0 END) as Blocked,
                SUM(CASE WHEN event = 'Idle' THEN COALESCE(time_t, 0.0) ELSE 0 END) as Idle,
                SUM(CASE WHEN event = 'Wait for Berth' THEN COALESCE(time_t, 0.0) ELSE 0 END) as WaitBerth
            FROM df_log
            WHERE process IN ('Downtime', 'Make', 'Move')
            GROUP BY 1, 2, 3
        """).df()
        
        for uk, group in dt_agg.groupby(['location', 'equipment']):
            unit_key = f"{uk[0]}|{uk[1]}"
            downtime_by_equipment[unit_key] = group.set_index('day')[['Maintenance', 'Breakdown', 'ResourceWait', 'Blocked', 'Idle', 'WaitBerth']].to_dict('index')

    result["downtime_by_equipment"] = downtime_by_equipment
    result["equipment_to_stores"] = equipment_to_stores
    
    store_to_equipment = {}
    for unit_key, stores in equipment_to_stores.items():
        for store_key in stores:
            if store_key not in store_to_equipment: store_to_equipment[store_key] = set()
            store_to_equipment[store_key].add(unit_key)
    result["store_to_equipment"] = store_to_equipment
    
    # 5. Store levels and unmet demand
    result["df_store_levels"] = pd.DataFrame([{"Store": k, "Level": round(c.level, 1), "Capacity": c.capacity} for k, c in sim.stores.items()])
    result["df_unmet"] = pd.DataFrame([{"Key": k, "Unmet": round(float(v), 2)} for k, v in sim.unmet.items()])
    
    return result


def _collapse_ship_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses consecutive log entries for the same vessel and event where all key parameters are identical.
    Sums the time and quantity for these entries.
    """
    if df.empty:
        return df

    # Define columns that must be identical to collapse
    # We include qty to ensure we don't collapse different load/unload amounts if they happen consecutively
    group_cols = [
        "process", "event", "location", "equipment", "product", 
        "from_store", "to_store", "route_id", "vessel_id", "ship_state"
    ]
    
    # We primarily target 'Ship' equipment as requested
    is_ship = df['equipment'] == 'Ship'
    if not is_ship.any():
        return df
        
    df_ship = df[is_ship].copy()
    df_others = df[~is_ship].copy()
    
    # Sort ship log by vessel and time to find consecutive entries for EACH vessel
    df_ship = df_ship.sort_values(by=["vessel_id", "time_h"])
    
    # Identify consecutive groups for ships
    # For comparison, normalize numeric columns and handle NA
    # We use a custom comparison that ignores 'qty' for wait/idle events
    temp_ship = df_ship[group_cols + (['qty'] if 'qty' in df_ship.columns else [])].copy()
    for col in ["route_id", "vessel_id", "qty"]:
        if col in temp_ship.columns:
            temp_ship[col] = pd.to_numeric(temp_ship[col], errors='coerce')
    
    temp_ship = temp_ship.fillna("___NULL___")
    
    # Define which columns to compare. 
    # For some events, we might want to ignore some columns.
    # But for simplicity, let's stick to group_cols and ensure we group by vessel.
    
    # IMPORTANT: ship_changed should be True if CURRENT row is different from PREVIOUS row (of SAME vessel)
    # We use shift(1) on the group
    prev_temp_ship = df_ship.groupby("vessel_id")[group_cols + (['qty'] if 'qty' in df_ship.columns else [])].shift(1)
    
    # Normalize prev_temp_ship too
    for col in ["route_id", "vessel_id", "qty"]:
        if col in prev_temp_ship.columns:
            prev_temp_ship[col] = pd.to_numeric(prev_temp_ship[col], errors='coerce')
    prev_temp_ship = prev_temp_ship.fillna("___NULL___")

    ship_changed = (temp_ship != prev_temp_ship).any(axis=1)
    ship_group_id = ship_changed.cumsum()
    
    # Define aggregation: sum numeric values, keep 'first' for everything else
    agg_dict = {
        'day': 'first',
        'time_h': 'first',
        'time_d': 'first',
        'process': 'first',
        'event': 'first',
        'location': 'first',
        'equipment': 'first',
        'product': 'first',
        'qty': 'sum',
        'time': 'sum',
        'unmet_demand': 'sum',
        'qty_out': 'sum',
        'from_store': 'first',
        'from_level': 'last',
        'from_fill_pct': 'last',
        'qty_in': 'sum',
        'to_store': 'first',
        'to_level': 'last',
        'to_fill_pct': 'last',
        'route_id': 'first',
        'vessel_id': 'first',
        'ship_state': 'first',
        'qty_t': 'sum',
        'time_t': 'sum'
    }
    
    collapsed_ship = df_ship.groupby(ship_group_id).agg(agg_dict).reset_index(drop=True)
    
    # Re-combine and sort by time_h to preserve chronological order across all equipment
    df_result = pd.concat([collapsed_ship, df_others]).sort_values(by=['time_h', 'vessel_id']).reset_index(drop=True)
    return df_result


def _collapse_truck_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses Truck log entries for each unique time_d, location, product, adding up the qty.
    """
    if df.empty:
        return df

    is_truck = df['equipment'] == 'Truck'
    if not is_truck.any():
        return df
        
    df_truck = df[is_truck].copy()
    df_others = df[~is_truck].copy()

    # Define aggregation: sum numeric values, keep 'first' for everything else
    # Note: time_h will be the 'first' time_h in the day for that group
    agg_dict = {
        'day': 'first',
        'time_h': 'first',
        'time_d': 'first',
        'process': 'first',
        'event': 'first',
        'location': 'first',
        'equipment': 'first',
        'product': 'first',
        'qty': 'sum',
        'time': 'sum',
        'unmet_demand': 'sum',
        'qty_out': 'sum',
        'from_store': 'first',
        'from_level': 'last', # Take the final level after all deliveries
        'from_fill_pct': 'last',
        'qty_in': 'sum',
        'to_store': 'first',
        'to_level': 'last',
        'to_fill_pct': 'last',
        'route_id': 'first',
        'vessel_id': 'first',
        'ship_state': 'first',
        'qty_t': 'sum',
        'time_t': 'sum'
    }

    # Group by time_d, location, product
    collapsed_truck = df_truck.groupby(['time_d', 'location', 'product'], as_index=False).agg(agg_dict)
    
    # Re-combine and sort by time_h
    df_result = pd.concat([collapsed_truck, df_others]).sort_values(by=['time_h']).reset_index(drop=True)
    return df_result
