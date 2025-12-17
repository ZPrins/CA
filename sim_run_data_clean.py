# sim_run_data_clean.py
# Layer 2: Cleaning, Mapping, and Normalization

import pandas as pd
from typing import Dict, Any, Tuple
from sim_run_utils import clean_df_cols_str, nan_to_none


def _rename_cols(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """Helper to rename columns if they exist."""
    return df.rename(columns=mapping)


def _normalize_ship_routes_wide_to_long(df: pd.DataFrame,
                                        store_lookup: Dict[Tuple[str, str, str], str]) -> pd.DataFrame:
    # (Same as before - keeping existing logic)
    first_col = df.columns[0]
    is_wide = (str(first_col).strip().lower() == 'field') or \
              (df[first_col].astype(str).str.contains('Route Group').any())

    if not is_wide: return df

    print("  [INFO] Normalizing Wide 'SHIP_ROUTES' format...")
    all_cols = [c for c in df.columns if c != first_col]
    route_cols = []
    for c in all_cols:
        route_map = dict(zip(df[first_col].astype(str).str.strip(), df[c].astype(str).str.strip()))
        rg = route_map.get('Route Group')
        if rg and rg.lower() not in ('nan', 'none', ''):
            route_cols.append(c)
    long_rows = []

    for r_col in route_cols:
        route_map = dict(zip(df[first_col].astype(str).str.strip(), df[r_col].astype(str).str.strip()))
        rg = route_map.get('Route Group')
        rid = route_map.get('Route ID')
        origin = route_map.get('Origin Location')
        return_loc = route_map.get('Return Location')

        if not rg or rg.lower() == 'nan': continue

        seq = 1
        current_loc = origin

        long_rows.append({'Route_Group': rg, 'Route_ID': rid, 'Sequence_ID': seq, 'Kind': 'start', 'Location': origin,
                          'Store_Key': None, 'Product_Class': None})
        seq += 1

        for _, row in df.iterrows():
            field_name = str(row[first_col]).strip()
            val = str(row[r_col]).strip()
            if val.lower() in ('nan', 'none', ''): continue

            if 'Load' in field_name and 'Unload' not in field_name and 'Store' not in field_name:
                store_field = field_name.replace("Load", "Store")
                store_name = route_map.get(store_field)
                key = None
                if store_name:
                    key = store_lookup.get((current_loc, store_name.upper(), val))
                    if not key: key = store_lookup.get((current_loc, store_name, val))

                long_rows.append(
                    {'Route_Group': rg, 'Route_ID': rid, 'Sequence_ID': seq, 'Kind': 'load', 'Location': current_loc,
                     'Store_Key': key, 'Product_Class': val})
                seq += 1
            elif 'Destination' in field_name and 'Location' in field_name:
                dest_loc = val
                long_rows.append(
                    {'Route_Group': rg, 'Route_ID': rid, 'Sequence_ID': seq, 'Kind': 'sail', 'Location': dest_loc,
                     'Store_Key': None, 'Product_Class': None})
                seq += 1
                current_loc = dest_loc
            elif 'Unload' in field_name and 'Store' not in field_name:
                # Try multiple store field name patterns
                store_name = None
                for pattern in [field_name + " Store", 
                                field_name.replace(" Unload", " Store"),
                                field_name.replace("Unload", "Store")]:
                    store_name = route_map.get(pattern)
                    if store_name and store_name.lower() not in ('nan', 'none', ''):
                        break
                
                key = None
                if store_name:
                    key = store_lookup.get((current_loc, store_name.upper(), val))
                    if not key: key = store_lookup.get((current_loc, store_name, val))

                long_rows.append(
                    {'Route_Group': rg, 'Route_ID': rid, 'Sequence_ID': seq, 'Kind': 'unload', 'Location': current_loc,
                     'Store_Key': key, 'Product_Class': val})
                seq += 1

        if return_loc and return_loc.lower() != 'nan':
            long_rows.append(
                {'Route_Group': rg, 'Route_ID': rid, 'Sequence_ID': seq, 'Kind': 'sail', 'Location': return_loc,
                 'Store_Key': None, 'Product_Class': None})

    return pd.DataFrame(long_rows)


def clean_all_data(raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    clean_data = {}

    # --- 1. STORE ---
    df_store = raw_data.get('Store', pd.DataFrame())
    loc_prod_map = {}
    store_lookup_map = {}

    if not df_store.empty:
        df_store = _rename_cols(df_store, {
            "Equipment Name": "Equipment",
            "Input": "Product_Class",
            "Silo Max Capacity": "Capacity_T",
            "Silo Opening Stock (Low)": "Opening_Low_T",
            "Silo Opening Stock (High)": "Opening_High_T"
        })

        df_store = clean_df_cols_str(df_store, ['Location', 'Equipment', 'Product_Class'])
        df_store['Equipment'] = df_store['Equipment'].str.upper()

        def make_key(row):
            try:
                if not row['Product_Class'] or not row['Location']: return None
                p, l, e = str(row['Product_Class']), str(row['Location']), str(row['Equipment'])
                return f"{p}|{l}|{e}|{p}"
            except:
                return None

        df_store['Store_Key'] = df_store.apply(make_key, axis=1)

        for col in ['Capacity_T', 'Opening_Low_T', 'Opening_High_T']:
            if col in df_store.columns:
                df_store[col] = pd.to_numeric(df_store[col], errors='coerce').fillna(0.0)

        clean_data['Store'] = df_store.dropna(subset=['Store_Key'])

        for _, row in clean_data['Store'].iterrows():
            k = row['Store_Key']
            l, e, p = row['Location'], row['Equipment'], row['Product_Class']
            store_lookup_map[(l, e, p)] = k
            if (l, p) not in loc_prod_map: loc_prod_map[(l, p)] = []
            loc_prod_map[(l, p)].append(k)
    else:
        clean_data['Store'] = pd.DataFrame()

    # --- 2. MAKE ---
    df_make = raw_data.get('Make', pd.DataFrame())
    if not df_make.empty:
        df_make = _rename_cols(df_make, {
            "Equipment Name": "Equipment",
            "Input": "Input_Product",
            "Output": "Product_Class",
            "Mean Production Rate (Tons/hr)": "Rate_TPH",
            "Consumption %": "Consumption_Pct",
            "Planned Maintenance Dates (Days of year)": "Maintenance_Days",
            "Unplanned downtime %": "Unplanned_Downtime_Pct"
        })
        if 'Choice_Rule' not in df_make.columns: df_make['Choice_Rule'] = 'min_fill_pct'

        df_make = clean_df_cols_str(df_make, ['Location', 'Equipment', 'Product_Class', 'Input_Product', 'Choice_Rule'])
        df_make['Equipment'] = df_make['Equipment'].str.upper()

        def resolve_make_keys(row):
            # Find ALL input stores for this location and input product
            inp_keys = []
            if row['Input_Product']:
                # First try exact match (Location, Equipment, Product)
                exact_key = store_lookup_map.get((row['Location'], row['Equipment'], row['Input_Product']))
                if exact_key:
                    inp_keys.append(exact_key)
                # Also add all other stores at this location for the same product
                all_at_loc = loc_prod_map.get((row['Location'], row['Input_Product']), [])
                for k in all_at_loc:
                    if k not in inp_keys:
                        inp_keys.append(k)
            
            # Find ALL output stores for this location and output product
            out_keys = loc_prod_map.get((row['Location'], row['Product_Class']), [])
            
            # For backward compatibility, also set single keys
            inp_key = inp_keys[0] if inp_keys else None
            out_key = out_keys[0] if out_keys else None
            
            return pd.Series([inp_key, out_key, inp_keys, out_keys])

        df_make[['Input_Store_Key', 'Output_Store_Key', 'Input_Store_Keys', 'Output_Store_Keys']] = df_make.apply(resolve_make_keys, axis=1)

        for col in ['Rate_TPH', 'Consumption_Pct', 'Step_Hours']:
            if col in df_make.columns:
                df_make[col] = pd.to_numeric(df_make[col], errors='coerce').fillna(0.0)

        if 'Consumption_Pct' not in df_make.columns: df_make['Consumption_Pct'] = 1.0
        if 'Step_Hours' not in df_make.columns: df_make['Step_Hours'] = 1.0

        clean_data['Make'] = df_make.dropna(subset=['Output_Store_Key'])

    # --- 3. DELIVER ---
    df_deliver = raw_data.get('Deliver', pd.DataFrame())
    if not df_deliver.empty:
        def calc_rate(row):
            try:
                if 'Annual Demand (Tons)' in row and pd.notna(row['Annual Demand (Tons)']):
                    return float(row['Annual Demand (Tons)']) / (365 * 24)
                if 'Demand per Location/Hour' in row:
                    return float(row['Demand per Location/Hour'])
                return 0.0
            except:
                return 0.0

        df_deliver['Rate_Per_Hour'] = df_deliver.apply(calc_rate, axis=1)
        df_deliver = _rename_cols(df_deliver, {"Input": "Product_Class"})

        def resolve_deliver_key(row):
            candidates = loc_prod_map.get((row['Location'], row['Product_Class']), [])
            return candidates[0] if candidates else None

        df_deliver['Store_Key'] = df_deliver.apply(resolve_deliver_key, axis=1)
        clean_data['Deliver'] = df_deliver.dropna(subset=['Store_Key'])

    # --- 4. MOVE (TRAIN) ---
    df_train = raw_data.get('Move_TRAIN', pd.DataFrame())
    if not df_train.empty:
        if 'Product Class' in df_train.columns and 'Product' in df_train.columns:
            df_train = df_train.drop(columns=['Product Class'])

        df_train = df_train.rename(columns={
            "Product Class": "Product_Class",
            "Product": "Product_Class",
            "Origin Location": "Origin_Location",
            "Destination Location": "Dest_Location",
            "Origin Store": "Origin_Store",
            "Destination Store": "Dest_Store",
            "# Trains": "N_Units",
            "Load Rate (ton/hr)": "Load_Rate_TPH",
            "Unload Rate (ton/hr)": "Unload_Rate_TPH",
            "Load Rate (tons/hr)": "Load_Rate_TPH",
            "Unload Rate (tons/hr)": "Unload_Rate_TPH",
            "Travel to Time (Min)": "To_Min",
            "Travel back Time (Min)": "Back_Min",
            "Avg Speed - Loaded (km/hr)": "Speed_Loaded",
            "Avg Speed - Empty (km/hr)": "Speed_Empty",
            "Distance": "Distance_Km"
        })

        df_train = df_train.loc[:, ~df_train.columns.duplicated()]
        df_train = clean_df_cols_str(df_train, ['Product_Class', 'Origin_Location', 'Dest_Location', 'Origin_Store', 'Dest_Store'])
        # Ensure equipment/store names are uppercase to match Store sheet processing
        if 'Origin_Store' in df_train.columns: df_train['Origin_Store'] = df_train['Origin_Store'].astype(str).str.upper()
        if 'Dest_Store' in df_train.columns: df_train['Dest_Store'] = df_train['Dest_Store'].astype(str).str.upper()

        def calc_payload_train(row):
            try:
                carr = 0
                for c_key in ['# Carraiges', '# Carriages', '# Wagons', '#Carraiges', '#Carriages']:
                    if c_key in row and pd.notna(row[c_key]):
                        carr = float(row[c_key])
                        break

                cap = 0
                for k_key in ['Carraige Capacity (ton)', 'Carriage Capacity (ton)', '# Carraige Capacity (ton)',
                              'Carriage Capacity']:
                    if k_key in row and pd.notna(row[k_key]):
                        cap = float(row[k_key])
                        break
                return carr * cap
            except:
                return 0.0

        df_train['Payload_T'] = df_train.apply(calc_payload_train, axis=1)

        def calc_times(row):
            to_min, back_min = row.get('To_Min', 0.0), row.get('Back_Min', 0.0)
            dist = float(row.get('Distance_Km', 0.0) or 0.0)
            if dist > 0:
                s_load = float(row.get('Speed_Loaded', 60) or 60)
                s_empty = float(row.get('Speed_Empty', 60) or 60)
                if (not to_min or to_min == 0) and s_load > 0: to_min = (dist / s_load) * 60
                if (not back_min or back_min == 0) and s_empty > 0: back_min = (dist / s_empty) * 60
            return pd.Series([to_min, back_min])

        df_train[['To_Min', 'Back_Min']] = df_train.apply(calc_times, axis=1)

        def resolve_route_keys(row):
            # Prefer explicit store/equipment selections from Move_TRAIN if provided
            origs_list = []
            dests_list = []

            try:
                orig_store = (row.get('Origin_Store') or '').strip() if 'Origin_Store' in row else ''
            except Exception:
                orig_store = ''
            try:
                dest_store = (row.get('Dest_Store') or '').strip() if 'Dest_Store' in row else ''
            except Exception:
                dest_store = ''

            # Resolve explicit origin store to store key if possible
            if orig_store:
                k = store_lookup_map.get((row['Origin_Location'], orig_store, row['Product_Class']))
                if k:
                    origs_list = [k]
            # Fallback: all location+product stores (preserve Store sheet order)
            if not origs_list:
                origs_list = loc_prod_map.get((row['Origin_Location'], row['Product_Class']), [])

            # Resolve explicit destination store to store key if possible
            if dest_store:
                k2 = store_lookup_map.get((row['Dest_Location'], dest_store, row['Product_Class']))
                if k2:
                    dests_list = [k2]
            # Fallback: all location+product stores (preserve Store sheet order)
            if not dests_list:
                dests_list = loc_prod_map.get((row['Dest_Location'], row['Product_Class']), [])

            return pd.Series([",".join(origs_list), ",".join(dests_list)])

        df_train[['Store_Keys_Origin', 'Store_Keys_Dest']] = df_train.apply(resolve_route_keys, axis=1)

        for col in ['N_Units', 'Load_Rate_TPH', 'Unload_Rate_TPH', 'To_Min', 'Back_Min']:
            if col in df_train.columns:
                df_train[col] = pd.to_numeric(df_train[col], errors='coerce').fillna(0.0)

        clean_data['Move_TRAIN'] = df_train.dropna(subset=['Product_Class'])

    # --- 5. SHIP ---
    # FIX: Rename "# Vessels" to "N_Units"
    df_ship = raw_data.get('Move_SHIP', pd.DataFrame())
    df_ship = _rename_cols(df_ship, {
        "Route Group": "Route_Group",
        "# Vessels": "N_Units",
        "Vessels": "N_Units",
        "No. Vessels": "N_Units",
        "Origin Location": "Origin_Location",
        "Route avg Speed (knots)": "Speed_Knots",
        "#Holds": "Holds_Per_Vessel",
        "Payload per Hold": "Payload_Per_Hold_T"
    })

    if not df_ship.empty:
        def calc_payload_ship(row):
            try:
                holds = float(row.get('Holds_Per_Vessel', 0) or 0)
                per_hold = float(row.get('Payload_Per_Hold_T', 0) or 0)
                return holds * per_hold
            except:
                return 0.0

        df_ship['Payload_T'] = df_ship.apply(calc_payload_ship, axis=1)
        
        for col in ['N_Units', 'Speed_Knots', 'Holds_Per_Vessel', 'Payload_Per_Hold_T']:
            if col in df_ship.columns:
                df_ship[col] = pd.to_numeric(df_ship[col], errors='coerce').fillna(0.0)

    clean_data['Move_SHIP'] = clean_df_cols_str(df_ship, ['Product_Class', 'Route_Group'])

    raw_routes = raw_data.get('SHIP_ROUTES', pd.DataFrame())
    if not raw_routes.empty:
        clean_data['SHIP_ROUTES'] = _normalize_ship_routes_wide_to_long(raw_routes, store_lookup_map)
    else:
        clean_data['SHIP_ROUTES'] = pd.DataFrame()

    df_berths = raw_data.get('SHIP_BERTHS', pd.DataFrame())
    df_berths = _rename_cols(df_berths, {
        "# Berths": "N_Berths",
        "Probability Berth Occupied %": "P_Occupied",
        "Pilot In (Hours)": "Pilot_In_H",
        "Pilot Out (Hours)": "Pilot_Out_H"
    })
    clean_data['SHIP_BERTHS'] = clean_df_cols_str(df_berths, ['Location', 'Store_Key'])
    clean_data['SHIP_DISTANCES'] = clean_df_cols_str(raw_data.get('SHIP_DISTANCES', pd.DataFrame()), [0, 1])

    return clean_data