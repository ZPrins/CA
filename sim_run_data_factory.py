# sim_run_data_factory.py
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from sim_run_types import StoreConfig, ProductionCandidate, MakeUnit, TransportRoute, Demand


def build_store_configs(df_store: pd.DataFrame) -> List[StoreConfig]:
    stores: List[StoreConfig] = []
    if df_store.empty: return stores
    for _, row in df_store.iterrows():
        try:
            stores.append(StoreConfig(
                key=row['Store_Key'],
                capacity=float(row['Capacity_T']),
                opening_low=float(row['Opening_Low_T']),
                opening_high=float(row['Opening_High_T']),
            ))
        except Exception as e:
            print(f"Warning: Could not create StoreConfig for row {row.get('Store_Key')}. Error: {e}")
    return stores


def _parse_maintenance_days(val) -> List[int]:
    """Parse maintenance days from comma-separated string with ranges like '1-5,6,9-12'"""
    if pd.isna(val) or val is None or str(val).strip() == '':
        return []
    try:
        result = []
        parts = str(val).split(',')
        for part in parts:
            part = part.strip()
            if '-' in part:
                range_parts = part.split('-')
                if len(range_parts) == 2:
                    start = int(range_parts[0].strip())
                    end = int(range_parts[1].strip())
                    result.extend(range(start, end + 1))
            elif part.isdigit():
                result.append(int(part))
        return sorted(set(result))
    except:
        return []

def build_make_units(df_make: pd.DataFrame) -> List[MakeUnit]:
    make_groups: Dict[Tuple[str, str], List[ProductionCandidate]] = defaultdict(list)
    unit_params: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(dict)
    if df_make.empty: return []
    for _, row in df_make.iterrows():
        location = row['Location']
        equipment = row['Equipment']
        unit_key = (location, equipment)
        unit_params[unit_key]['choice_rule'] = row.get('Choice_Rule', 'min_fill_pct')
        unit_params[unit_key]['step_hours'] = float(row.get('Step_Hours', 1.0))
        
        maint_days = _parse_maintenance_days(row.get('Maintenance_Days', ''))
        if maint_days:
            unit_params[unit_key]['maintenance_days'] = maint_days
        
        downtime_pct = row.get('Unplanned_Downtime_Pct', 0.0)
        if pd.notna(downtime_pct):
            unit_params[unit_key]['unplanned_downtime_pct'] = float(downtime_pct)
        
        # Get lists of input/output stores if available
        in_store_keys = row.get('Input_Store_Keys') if 'Input_Store_Keys' in row.index else None
        out_store_keys = row.get('Output_Store_Keys') if 'Output_Store_Keys' in row.index else None
        
        # Ensure they are lists (not None or other types)
        if in_store_keys is not None and not isinstance(in_store_keys, list):
            in_store_keys = [in_store_keys] if in_store_keys else None
        if out_store_keys is not None and not isinstance(out_store_keys, list):
            out_store_keys = [out_store_keys] if out_store_keys else None
        
        candidate = ProductionCandidate(
            product=row['Product_Class'],
            out_store_key=row['Output_Store_Key'],
            in_store_key=row['Input_Store_Key'] if row['Input_Store_Key'] else None,
            rate_tph=float(row['Rate_TPH']),
            consumption_pct=float(row['Consumption_Pct']),
            in_store_keys=in_store_keys if in_store_keys else None,
            out_store_keys=out_store_keys if out_store_keys else None
        )
        make_groups[unit_key].append(candidate)
    make_units: List[MakeUnit] = []
    for unit_key, candidates in make_groups.items():
        loc, eq = unit_key
        params = unit_params[unit_key]
        make_units.append(MakeUnit(
            location=loc,
            equipment=eq,
            candidates=candidates,
            choice_rule=params.get('choice_rule', 'min_fill_pct'),
            step_hours=params.get('step_hours', 1.0),
            maintenance_days=params.get('maintenance_days'),
            unplanned_downtime_pct=params.get('unplanned_downtime_pct', 0.0)
        ))
    return make_units


def build_transport_routes(clean_data: Dict[str, pd.DataFrame]) -> List[TransportRoute]:
    routes: List[TransportRoute] = []
    df_train = clean_data.get('Move_TRAIN', pd.DataFrame())
    df_ship_config = clean_data.get('Move_SHIP', pd.DataFrame())

    # --- TRAINS ---
    if not df_train.empty:
        for _, row in df_train.iterrows():
            try:
                orig_keys = [k.strip() for k in str(row['Store_Keys_Origin']).split(',') if k.strip()]
                dest_keys = [k.strip() for k in str(row['Store_Keys_Dest']).split(',') if k.strip()]
                if not orig_keys or not dest_keys:
                    print(
                        f"  [WARN] Skipping TRAIN route: {row.get('Origin_Location')} -> {row.get('Dest_Location')} ({row.get('Product_Class')}). No stores found.")
                    continue
                routes.append(TransportRoute(
                    product=row['Product_Class'],
                    origin_location=row['Origin_Location'],
                    dest_location=row['Dest_Location'],
                    origin_stores=orig_keys,
                    dest_stores=dest_keys,
                    n_units=int(row['N_Units']),
                    payload_t=float(row['Payload_T']),
                    load_rate_tph=float(row['Load_Rate_TPH']),
                    unload_rate_tph=float(row['Unload_Rate_TPH']),
                    to_min=float(row['To_Min']),
                    back_min=float(row['Back_Min']),
                    mode="TRAIN"
                ))
            except Exception as e:
                print(f"Warning: Could not create Train Route. Error: {e}")

    # --- SHIPS ---
    if not df_ship_config.empty:
        berth_info = _process_berth_data(clean_data.get('SHIP_BERTHS', pd.DataFrame()))
        ship_routes_data = _process_ship_routes_data(clean_data.get('SHIP_ROUTES', pd.DataFrame()))
        nm_distances = _process_distance_data(clean_data.get('SHIP_DISTANCES', pd.DataFrame()))

        for _, row in df_ship_config.iterrows():
            try:
                route_group = row['Route_Group']
                product = row.get('Product_Class')
                itineraries = ship_routes_data.get(route_group, [])
                if not itineraries:
                    print(f"  [WARN] Skipping SHIP route group '{route_group}'. No itineraries found.")
                    continue

                # Infer Product
                if not product or str(product) == 'nan':
                    for it in itineraries:
                        for step in it:
                            p = step.get('product')  # Helper now saves as 'product'
                            if step.get('kind') == 'load' and p:
                                product = p
                                break
                        if product: break
                if not product: product = "Unknown"

                all_locations = set()
                for it in itineraries:
                    for step in it:
                        loc = step.get('location') or step.get('from')
                        if loc: all_locations.add(loc)

                origin = next(iter(all_locations)) if all_locations else "Unknown"
                dest = origin

                origin_loc = row.get('Origin_Location', origin) or origin
                routes.append(TransportRoute(
                    product=str(product),
                    origin_location=origin_loc,
                    dest_location=dest,
                    origin_stores=[],
                    dest_stores=[],
                    n_units=int(row.get('N_Units', 0) or 0),
                    payload_t=float(row.get('Payload_T', 0.0) or 0.0),
                    load_rate_tph=float(row.get('Load_Rate_TPH', 0.0) or 0.0),
                    unload_rate_tph=float(row.get('Unload_Rate_TPH', 0.0) or 0.0),
                    to_min=float(row.get('To_Min', 0.0) or 0.0),
                    back_min=float(row.get('Back_Min', 0.0) or 0.0),
                    mode="SHIP",
                    route_group=route_group,
                    speed_knots=float(row.get('Speed_Knots', 0.0) or 0.0),
                    holds_per_vessel=int(row.get('Holds_Per_Vessel', 0) or 0),
                    payload_per_hold_t=float(row.get('Payload_Per_Hold_T', 0.0) or 0.0),
                    max_wait_product_h=float(row.get('Max_Wait_Product_H', 0.0) or 0.0),
                    itineraries=itineraries,
                    berth_info=berth_info,
                    nm_distance=nm_distances
                ))
            except Exception as e:
                print(f"Warning: Could not create Ship Route. Error: {e}")
    return routes


def build_demands(df_deliver: pd.DataFrame) -> List[Demand]:
    demands: List[Demand] = []
    if df_deliver.empty: return demands
    for _, row in df_deliver.iterrows():
        try:
            demands.append(Demand(store_key=row['Store_Key'], rate_per_hour=float(row['Rate_Per_Hour'])))
        except Exception as e:
            print(f"Warning: Could not create Demand for row {row.get('Store_Key')}. Error: {e}")
    return demands


# --- Helpers ---
def _process_ship_routes_data(df_routes: pd.DataFrame) -> Dict[str, List[List[Dict[str, Any]]]]:
    route_groups: Dict[str, List[List[Dict[str, Any]]]] = defaultdict(list)
    if df_routes.empty: return {}
    for route_group, group_df in df_routes.groupby('Route_Group'):
        group_df = group_df.sort_values(by=['Route_ID', 'Sequence_ID'])
        for route_id, route_df in group_df.groupby('Route_ID'):
            itinerary: List[Dict[str, Any]] = []
            current_location = None
            for _, row in route_df.iterrows():
                if str(row['Kind']).lower() == 'start':
                    current_location = str(row['Location'])
                    itinerary.append({'kind': 'start', 'location': current_location, 'route_id': route_id})
                    break
            if not current_location: continue
            for _, row in route_df.iterrows():
                kind = str(row['Kind']).lower()
                location = str(row['Location']) if str(row['Location']) != 'nan' else current_location
                if kind == 'start': continue
                if kind == 'sail':
                    itinerary.append({'kind': 'sail', 'from': current_location, 'to': location})
                    current_location = location
                elif kind in ('load', 'unload'):
                    # FIX: SAVE AS 'product' (lowercase) to match core logic
                    itinerary.append({
                        'kind': kind,
                        'location': location,
                        'store_key': str(row.get('Store_Key')),
                        'product': str(row.get('Product_Class'))
                    })
            if itinerary:
                route_groups[route_group].append(itinerary)
    return route_groups


def _process_berth_data(df_berths: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    berth_info: Dict[str, Dict[str, Any]] = {}
    if df_berths.empty: return berth_info
    for _, row in df_berths.iterrows():
        location = str(row['Location'])
        if not location or location == 'nan': continue
        info = {
            'berths': int(row.get('N_Berths', 1) or 1),
            'pilot_in_h': float(row.get('Pilot_In_H', 0.0) or 0.0),
            'pilot_out_h': float(row.get('Pilot_Out_H', 0.0) or 0.0),
            'p_occupied': float(row.get('P_Occupied', 0.0) or 0.0),
            'store_key': str(row.get('Store_Key'))
        }
        berth_info[location] = info
    return berth_info


def _process_distance_data(df_distances: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    nm_distances: Dict[Tuple[str, str], float] = {}
    if df_distances.empty: return nm_distances
    cols = ['From_Loc', 'To_Loc', 'Distance_NM']
    if len(df_distances.columns) >= 3:
        current_cols = df_distances.columns.tolist()
        if len(current_cols) >= 3 and current_cols[0] not in cols:
            df_distances.columns = cols + current_cols[3:]
        for _, row in df_distances.iterrows():
            try:
                loc_a, loc_b = str(row['From_Loc']), str(row['To_Loc'])
                dist = float(row['Distance_NM'])
                nm_distances[(loc_a, loc_b)] = dist
                nm_distances[(loc_b, loc_a)] = dist
            except Exception:
                pass
    return nm_distances