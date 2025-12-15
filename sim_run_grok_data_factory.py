# sim_run_grok_data_factory.py
# Layer 3: Factory (Takes clean DataFrames and produces simulation objects)

import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from sim_run_grok_core import StoreConfig, ProductionCandidate, MakeUnit, TransportRoute, Demand


def build_store_configs(df_store: pd.DataFrame) -> List[StoreConfig]:
    """Converts the clean 'Store' DataFrame into a list of StoreConfig objects."""
    stores: List[StoreConfig] = []
    if df_store.empty:
        return stores

    for _, row in df_store.iterrows():
        try:
            stores.append(StoreConfig(
                key=row['Store_Key'],
                capacity=float(row['Capacity_T']),
                opening_low=float(row['Opening_Low_T']),
                opening_high=float(row['Opening_High_T']),
            ))
        except Exception as e:
            print(f"Warning: Could not create StoreConfig for row {row['Store_Key']}. Error: {e}")

    return stores


def build_make_units(df_make: pd.DataFrame) -> List[MakeUnit]:
    """Converts the clean 'Make' DataFrame into a list of MakeUnit objects."""

    # Group candidates by location and equipment (the unit)
    make_groups: Dict[Tuple[str, str], List[ProductionCandidate]] = defaultdict(list)
    unit_params: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(dict)

    if df_make.empty:
        return []

    for _, row in df_make.iterrows():
        location = row['Location']
        equipment = row['Equipment']
        unit_key = (location, equipment)

        # Collect unit-level parameters (last one wins if duplicated)
        unit_params[unit_key]['choice_rule'] = row['Choice_Rule']
        unit_params[unit_key]['step_hours'] = float(row['Step_Hours']) if float(row['Step_Hours']) > 0 else 1.0

        # Create the candidate
        candidate = ProductionCandidate(
            product=row['Product_Class'],
            out_store_key=row['Output_Store_Key'],
            in_store_key=row['Input_Store_Key'] if row['Input_Store_Key'] else None,
            rate_tph=float(row['Rate_TPH']),
            consumption_pct=float(row['Consumption_Pct'])
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
            step_hours=params.get('step_hours', 1.0)
        ))

    return make_units


def build_transport_routes(clean_data: Dict[str, pd.DataFrame]) -> List[TransportRoute]:
    """
    Converts clean Move DataFrames and related Ship configuration into
    a list of TransportRoute objects.
    """
    routes: List[TransportRoute] = []
    df_train = clean_data.get('Move_TRAIN', pd.DataFrame())
    df_ship_config = clean_data.get('Move_SHIP', pd.DataFrame())

    # --- Load Train Routes ---
    if not df_train.empty:
        for _, row in df_train.iterrows():
            try:
                # Split comma-separated store keys
                orig_keys = [k.strip() for k in str(row['Store_Keys_Origin']).split(',') if k.strip()]
                dest_keys = [k.strip() for k in str(row['Store_Keys_Dest']).split(',') if k.strip()]

                if not orig_keys or not dest_keys:
                    print(f"Warning: Train Route missing origin or destination stores: {row['Product_Class']}")
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
                print(f"Warning: Could not create Train Route. Error: {e}. Row: {row}")

    # --- Load Ship Routes ---
    if not df_ship_config.empty:
        berth_info = _process_berth_data(clean_data.get('SHIP_BERTHS', pd.DataFrame()))
        ship_routes_data = _process_ship_routes_data(clean_data.get('SHIP_ROUTES', pd.DataFrame()))
        nm_distances = _process_distance_data(clean_data.get('SHIP_DISTANCES', pd.DataFrame()))

        for _, row in df_ship_config.iterrows():
            try:
                route_group = row['Route_Group']
                product = row['Product_Class']

                # Check for required data
                itineraries = ship_routes_data.get(route_group, [])
                if not itineraries:
                    print(f"Warning: Ship Route Group '{route_group}' has no defined itineraries in SHIP_ROUTES.")
                    continue

                # Extract origin and destination from itineraries for TransportRoute structure
                all_locations = set()
                for it in itineraries:
                    for step in it:
                        loc = step.get('location') or step.get('from')
                        if loc: all_locations.add(loc)

                # Default origin/dest for the structure, used for logging/debugging
                origin = next(iter(all_locations)) if all_locations else "Unknown"
                dest = origin

                routes.append(TransportRoute(
                    product=product,
                    origin_location=origin,
                    dest_location=dest,
                    origin_stores=[],  # Handled dynamically by itinerary
                    dest_stores=[],  # Handled dynamically by itinerary
                    n_units=int(row.get('N_Units', 0) or 0),
                    payload_t=float(row.get('Payload_T', 0.0) or 0.0),
                    load_rate_tph=float(row.get('Load_Rate_TPH', 0.0) or 0.0),
                    unload_rate_tph=float(row.get('Unload_Rate_TPH', 0.0) or 0.0),
                    to_min=float(row.get('To_Min', 0.0) or 0.0),  # Used for initial transit only
                    back_min=float(row.get('Back_Min', 0.0) or 0.0),  # Used for initial transit only
                    mode="SHIP",
                    route_group=route_group,
                    speed_knots=float(row.get('Speed_Knots', 0.0) or 0.0),
                    itineraries=itineraries,
                    berth_info=berth_info,
                    nm_distance=nm_distances
                ))
            except Exception as e:
                print(f"Warning: Could not create Ship Route. Error: {e}. Row: {row}")

    return routes


def build_demands(df_deliver: pd.DataFrame) -> List[Demand]:
    """Converts the clean 'Deliver' DataFrame into a list of Demand objects."""
    demands: List[Demand] = []
    if df_deliver.empty:
        return demands

    for _, row in df_deliver.iterrows():
        try:
            demands.append(Demand(
                store_key=row['Store_Key'],
                rate_per_hour=float(row['Rate_Per_Hour'])
            ))
        except Exception as e:
            print(f"Warning: Could not create Demand for row {row.get('Store_Key')}. Error: {e}")

    return demands


# --- Internal Helper Functions (Moved from original loader) ---

def _process_ship_routes_data(df_routes: pd.DataFrame) -> Dict[str, List[List[Dict[str, Any]]]]:
    route_groups: Dict[str, List[List[Dict[str, Any]]]] = defaultdict(list)

    if df_routes.empty:
        return {}

    for route_group, group_df in df_routes.groupby('Route_Group'):
        # Sort by Route_ID then Sequence_ID
        group_df = group_df.sort_values(by=['Route_ID', 'Sequence_ID'])

        for route_id, route_df in group_df.groupby('Route_ID'):
            itinerary: List[Dict[str, Any]] = []
            current_location = None

            # Find the starting location
            for _, row in route_df.iterrows():
                if str(row['Kind']).lower() == 'start':
                    current_location = str(row['Location'])
                    itinerary.append({
                        'kind': 'start',
                        'location': current_location,
                        'route_id': route_id,
                    })
                    break

            if not current_location:
                print(f"Warning: Route Group {route_group}, ID {route_id} missing 'start' step.")
                continue

            for _, row in route_df.iterrows():
                kind = str(row['Kind']).lower()
                location = str(row['Location']) if str(row['Location']) != 'nan' else current_location

                if kind == 'start':
                    continue  # Already handled

                if kind == 'sail':
                    # Sail step
                    itinerary.append({
                        'kind': 'sail',
                        'from': current_location,
                        'to': location
                    })
                    current_location = location
                elif kind in ('load', 'unload'):
                    # Load/Unload step
                    itinerary.append({
                        'kind': kind,
                        'location': location,
                        'store_key': str(row['Store_Key']),
                        'product': str(row['Product_Class'])
                    })

            if itinerary:
                route_groups[route_group].append(itinerary)

    return route_groups


def _process_berth_data(df_berths: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    berth_info: Dict[str, Dict[str, Any]] = {}
    if df_berths.empty:
        return berth_info

    for _, row in df_berths.iterrows():
        location = str(row['Location'])
        if not location or location == 'nan':
            continue

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
    if df_distances.empty:
        return nm_distances

    # Assumes df_distances has columns: From_Loc, To_Loc, Distance_NM

    cols = ['From_Loc', 'To_Loc', 'Distance_NM']
    if len(df_distances.columns) >= 3:
        # Renames columns if they are not explicitly named (e.g., loaded without header)
        current_cols = df_distances.columns.tolist()
        if len(current_cols) >= 3 and current_cols[0] not in cols:
            df_distances.columns = cols + current_cols[3:]

        for _, row in df_distances.iterrows():
            try:
                loc_a = str(row['From_Loc'])
                loc_b = str(row['To_Loc'])
                dist = float(row['Distance_NM'])

                # Store bi-directionally
                nm_distances[(loc_a, loc_b)] = dist
                nm_distances[(loc_b, loc_a)] = dist
            except Exception as e:
                # print(f"Warning: Could not process distance row. Error: {e}")
                pass

    return nm_distances