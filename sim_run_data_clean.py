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
    """Optimized wide-to-long normalization for SHIP_ROUTES."""
    first_col = df.columns[0]
    
    # Check if it's actually in wide format by looking at the first column values
    first_col_vals = df[first_col].astype(str).str.strip().str.lower()
    is_wide = (str(first_col).strip().lower() == 'field') or (first_col_vals == 'route group').any()

    if not is_wide: return df

    print("  [INFO] Normalizing Wide 'SHIP_ROUTES' format...")
    
    # Clean the field names to ensure reliable lookups
    df = df.copy()
    df[first_col] = df[first_col].astype(str).str.strip()
    
    # Identify which columns represent routes (exclude the 'Field' column)
    all_cols = [c for c in df.columns if c != first_col]
    
    # Find the row indices for key fields
    field_series = df[first_col]
    idx_rg = field_series[field_series == 'Route Group'].index
    if idx_rg.empty: return df # Should not happen if is_wide
    
    idx_rid = field_series[field_series == 'Route ID'].index
    idx_origin = field_series[field_series == 'Origin Location'].index
    idx_return = field_series[field_series == 'Return Location'].index

    long_rows = []
    
    for r_col in all_cols:
        # Quick check if this column has a valid Route Group
        rg = df.at[idx_rg[0], r_col]
        if pd.isna(rg) or str(rg).strip().lower() in ('nan', 'none', ''):
            continue
            
        rg = str(rg).strip()
        rid = str(df.at[idx_rid[0], r_col]).strip() if not idx_rid.empty else None
        origin = str(df.at[idx_origin[0], r_col]).strip() if not idx_origin.empty else None
        return_loc = str(df.at[idx_return[0], r_col]).strip() if not idx_return.empty else None

        if not origin or origin.lower() == 'nan': continue

        seq = 1
        current_loc = origin

        long_rows.append({
            'Route_Group': rg, 'Route_ID': rid, 'Sequence_ID': seq, 'Kind': 'start', 
            'Location': origin, 'Store_Key': None, 'Product_Class': None
        })
        seq += 1

        # Process loads and unloads. 
        # We iterate over rows once for this route column.
        for idx, row in df.iterrows():
            field_name = row[first_col]
            val = str(row[r_col]).strip()
            if val.lower() in ('nan', 'none', ''): continue

            if 'Load' in field_name and 'Unload' not in field_name and 'Store' not in field_name:
                # Find associated Store field
                store_field = field_name.replace("Load", "Store")
                # Look up store name in the same column for the store field row
                store_row = df[df[first_col] == store_field]
                if not store_row.empty:
                    store_name = str(store_row.iloc[0][r_col]).strip()
                    key = store_lookup.get((current_loc, store_name.upper(), val)) or \
                          store_lookup.get((current_loc, store_name, val))
                    
                    long_rows.append({
                        'Route_Group': rg, 'Route_ID': rid, 'Sequence_ID': seq, 'Kind': 'load', 
                        'Location': current_loc, 'Store_Key': key, 'Product_Class': val
                    })
                    seq += 1
            
            elif 'Destination' in field_name and 'Location' in field_name:
                dest_loc = val
                if dest_loc != current_loc:
                    long_rows.append({
                        'Route_Group': rg, 'Route_ID': rid, 'Sequence_ID': seq, 'Kind': 'sail', 
                        'Location': dest_loc, 'Store_Key': None, 'Product_Class': None
                    })
                    seq += 1
                    current_loc = dest_loc
            
            elif 'Unload' in field_name and 'Store' not in field_name:
                # Find associated store name
                store_name = None
                for pattern in [field_name + " Store", 
                                field_name.replace(" Unload", " Store"),
                                field_name.replace("Unload", "Store")]:
                    store_row = df[df[first_col] == pattern]
                    if not store_row.empty:
                        sn = str(store_row.iloc[0][r_col]).strip()
                        if sn.lower() not in ('nan', 'none', ''):
                            store_name = sn
                            break
                
                if store_name:
                    key = store_lookup.get((current_loc, store_name.upper(), val)) or \
                          store_lookup.get((current_loc, store_name, val))
                    long_rows.append({
                        'Route_Group': rg, 'Route_ID': rid, 'Sequence_ID': seq, 'Kind': 'unload', 
                        'Location': current_loc, 'Store_Key': key, 'Product_Class': val
                    })
                    seq += 1

        if return_loc and return_loc.lower() != 'nan' and return_loc != current_loc:
            long_rows.append({
                'Route_Group': rg, 'Route_ID': rid, 'Sequence_ID': seq, 'Kind': 'sail', 
                'Location': return_loc, 'Store_Key': None, 'Product_Class': None
            })

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
            "Silo Opening Stock (High)": "Opening_High_T",
            "Load Rate (TPH)": "Load_Rate_TPH",
            "Load Rate (ton/hr)": "Load_Rate_TPH",
            "Unload Rate (TPH)": "Unload_Rate_TPH",
            "Unload Rate (ton/hr)": "Unload_Rate_TPH"
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

        for col in ['Capacity_T', 'Opening_Low_T', 'Opening_High_T', 'Load_Rate_TPH', 'Unload_Rate_TPH']:
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

    # --- 1.5 NETWORK (Pre-process for Make/Move/Conveyor) ---
    df_net = raw_data.get('Network', pd.DataFrame())
    network_make_input_map = {} # (Location, Equipment, Product) -> List of Store Keys
    network_make_output_map = {} # (Location, Equipment, Product) -> List of Store Keys
    
    # NEW: Maps for Conveyor Discovery
    conveyor_origins = {} # (Location, Equipment, Product) -> List of Store Keys
    conveyor_destinations = {} # (Location, Equipment, Product) -> List of Store Keys

    if not df_net.empty:
        # Standardize columns
        df_net = df_net.rename(columns={
            'Equipment Name': 'Equipment',
            'Next Equipment': 'Next_Equipment'
        })
        df_net = clean_df_cols_str(df_net, ['Location', 'Equipment', 'Process', 'Input', 'Output', 'Next_Equipment'])
        
        # Build Input Map: Find Stores where Next_Equipment == a Make Unit OR Conveyor
        for _, row in df_net.iterrows():
            loc = row['Location']
            eq = row['Equipment']
            proc = row['Process']
            prod = row['Output'] # For stores, Output is the product it holds
            next_eq = row['Next_Equipment']
            
            if proc == 'Store' and next_eq:
                store_key = store_lookup_map.get((loc, eq, prod))
                if store_key:
                    # Generic input map for Make units
                    key = (loc, next_eq, prod)
                    if key not in network_make_input_map: network_make_input_map[key] = []
                    if store_key not in network_make_input_map[key]:
                        network_make_input_map[key].append(store_key)
                    
                    # Specific origin map for Conveyors
                    # We check if next_eq is a conveyor in the next pass, but we can store it here
                    if key not in conveyor_origins: conveyor_origins[key] = []
                    if store_key not in conveyor_origins[key]:
                        conveyor_origins[key].append(store_key)

        # Build Output Map: Find Units (Make/Conveyor) and their Next_Equipment (Store)
        for _, row in df_net.iterrows():
            loc = row['Location']
            eq = row['Equipment']
            proc = row['Process']
            prod = row['Output']
            next_eq = row['Next_Equipment']
            
            if next_eq:
                store_key = store_lookup_map.get((loc, next_eq, prod))
                if store_key:
                    if proc == 'Make':
                        key = (loc, eq, prod)
                        if key not in network_make_output_map: network_make_output_map[key] = []
                        if store_key not in network_make_output_map[key]:
                            network_make_output_map[key].append(store_key)
                    
                    if proc == 'Move' or eq.upper() == 'CONVEYOR':
                        key = (loc, eq, prod)
                        if key not in conveyor_destinations: conveyor_destinations[key] = []
                        if store_key not in conveyor_destinations[key]:
                            conveyor_destinations[key].append(store_key)

    # --- 1.6 CONVEYOR DISCOVERY (Legacy - removed in favor of explicit sheet) ---
    clean_data['Move_CONVEYOR'] = pd.DataFrame()

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
            loc, eq, prod_in, prod_out = row['Location'], row['Equipment'], row['Input_Product'], row['Product_Class']
            
            # Find Input Stores
            inp_keys = []
            # 1. Try Network Map (Highest Priority)
            if (loc, eq, prod_in) in network_make_input_map:
                inp_keys = list(network_make_input_map[(loc, eq, prod_in)])
            
            # 2. Fallback ONLY if no network match found
            if not inp_keys and prod_in:
                # Exact match (Location, Equipment, Product) - Silo with same name as machine?
                exact_key = store_lookup_map.get((loc, eq, prod_in))
                if exact_key:
                    inp_keys = [exact_key]
                else:
                    # All other stores at this location for same product - use only the first one
                    all_at_loc = loc_prod_map.get((loc, prod_in), [])
                    if all_at_loc:
                        inp_keys = [all_at_loc[0]]
            
            # Find Output Stores
            out_keys = []
            # 1. Try Network Map (Highest Priority)
            if (loc, eq, prod_out) in network_make_output_map:
                out_keys = list(network_make_output_map[(loc, eq, prod_out)])
            
            # 2. Fallback ONLY if no network match found
            if not out_keys:
                all_at_loc_out = loc_prod_map.get((loc, prod_out), [])
                if all_at_loc_out:
                    out_keys = [all_at_loc_out[0]]
            
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
        df_deliver = _rename_cols(df_deliver, {"Input": "Product_Class", "Demand Store": "Demand_Store"})

        def resolve_deliver_key(row):
            loc = row['Location']
            prod = row['Product_Class']
            demand_store = str(row.get('Demand_Store', '')).strip()
            
            # Default candidates from all at location
            candidates = loc_prod_map.get((loc, prod), [])
            
            # If a specific demand store is provided, strictly follow it (or a comma list)
            if demand_store and demand_store.lower() != 'nan':
                # Support comma-separated list of store names
                store_names = [s.strip() for s in demand_store.split(',') if s.strip()]
                resolved_keys = []
                for sn in store_names:
                    # Try to find a store key that matches this demand_store name (as Equipment)
                    match_key = store_lookup_map.get((loc, sn.upper(), prod)) or \
                                store_lookup_map.get((loc, sn, prod))
                    if match_key:
                        resolved_keys.append(match_key)
                
                if resolved_keys:
                    candidates = resolved_keys
            
            # If still no candidates, fallback to first one at location (legacy)
            if not candidates:
                candidates = loc_prod_map.get((loc, prod), [])[:1]
            
            first_key = candidates[0] if candidates else None
            return pd.Series([first_key, candidates])

        df_deliver[['Store_Key', 'Store_Keys']] = df_deliver.apply(resolve_deliver_key, axis=1)
        clean_data['Deliver'] = df_deliver.dropna(subset=['Store_Key'])

    # --- 4. MOVE (TRAIN) ---
    df_train = raw_data.get('Move_TRAIN', pd.DataFrame())
    if not df_train.empty:
        # Standardize columns
        if 'Product Class' in df_train.columns and 'Product' in df_train.columns:
            df_train = df_train.drop(columns=['Product Class'])

        df_train = df_train.rename(columns={
            "Product Class": "Product_Class",
            "Product": "Product_Class",
            "Origin Location": "Origin_Location",
            "Destination Location": "Dest_Location",
            "Origin Store": "Origin_Store",
            "Destination Store": "Dest_Store",
            "Equipment Name": "Equipment",
            "Equipment": "Equipment",
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
        df_train = clean_df_cols_str(df_train, ['Product_Class', 'Origin_Location', 'Dest_Location', 'Origin_Store', 'Dest_Store', 'Equipment'])
        # Ensure equipment/store names are uppercase to match Store sheet processing
        if 'Origin_Store' in df_train.columns: df_train['Origin_Store'] = df_train['Origin_Store'].astype(str).str.upper()
        if 'Dest_Store' in df_train.columns: df_train['Dest_Store'] = df_train['Dest_Store'].astype(str).str.upper()
        if 'Equipment' in df_train.columns: df_train['Equipment'] = df_train['Equipment'].astype(str).str.upper()

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
                # Support comma-separated list of store names
                store_names = [s.strip() for s in orig_store.split(',') if s.strip()]
                for sn in store_names:
                    # Try with Equipment name if available (sometimes Store sheet uses it as name)
                    k = store_lookup_map.get((row['Origin_Location'], sn.upper(), row['Product_Class']))
                    if k:
                        origs_list.append(k)
                    else:
                        # Search all stores at this location for this product and see if 'Equipment' (Silo name) matches
                        all_at_loc = loc_prod_map.get((row['Origin_Location'], row['Product_Class']), [])
                        for candidate_key in all_at_loc:
                            # Store keys are Product|Location|Equipment|Product
                            parts = candidate_key.split('|')
                            if len(parts) >= 3 and parts[2].upper() == sn.upper():
                                origs_list.append(candidate_key)
                                break

            # Fallback: all location+product stores (preserve Store sheet order)
            if not origs_list:
                origs_list = loc_prod_map.get((row['Origin_Location'], row['Product_Class']), [])

            # Resolve explicit destination store to store key if possible
            if dest_store:
                # Support comma-separated list of store names
                store_names = [s.strip() for s in dest_store.split(',') if s.strip()]
                for sn in store_names:
                    k2 = store_lookup_map.get((row['Dest_Location'], sn.upper(), row['Product_Class']))
                    if k2:
                        dests_list.append(k2)
                    else:
                        # Search all stores at this location for this product and see if 'Equipment' (Silo name) matches
                        all_at_loc_dest = loc_prod_map.get((row['Dest_Location'], row['Product_Class']), [])
                        for candidate_key in all_at_loc_dest:
                            parts = candidate_key.split('|')
                            if len(parts) >= 3 and parts[2].upper() == sn.upper():
                                dests_list.append(candidate_key)
                                break

            # Fallback: all location+product stores (preserve Store sheet order)
            if not dests_list:
                dests_list = loc_prod_map.get((row['Dest_Location'], row['Product_Class']), [])

            return pd.Series([",".join(origs_list), ",".join(dests_list)])

        df_train[['Store_Keys_Origin', 'Store_Keys_Dest']] = df_train.apply(resolve_route_keys, axis=1)

        # Force Mode to TRAIN for all entries in Move_TRAIN sheet
        df_train['Mode'] = 'TRAIN'

        for col in ['N_Units', 'Load_Rate_TPH', 'Unload_Rate_TPH', 'To_Min', 'Back_Min']:
            if col in df_train.columns:
                df_train[col] = pd.to_numeric(df_train[col], errors='coerce').fillna(0.0)

        clean_data['Move_TRAIN'] = df_train.dropna(subset=['Product_Class'])

    # --- 4.5 MOVE (CONVEYOR) ---
    df_conv = raw_data.get('Move_CONVEYOR', pd.DataFrame())
    if not df_conv.empty:
        df_conv = df_conv.rename(columns={
            "Product Class": "Product_Class",
            "Product": "Product_Class",
            "Origin Store": "Origin_Store",
            "Destination Store": "Dest_Store",
            "Speed (tons/hr)": "Speed_TPH"
        })
        df_conv = df_conv.loc[:, ~df_conv.columns.duplicated()]
        df_conv = clean_df_cols_str(df_conv, ['Product_Class', 'Location', 'Origin_Store', 'Dest_Store'])
        if 'Origin_Store' in df_conv.columns: df_conv['Origin_Store'] = df_conv['Origin_Store'].astype(str).str.upper()
        if 'Dest_Store' in df_conv.columns: df_conv['Dest_Store'] = df_conv['Dest_Store'].astype(str).str.upper()

        def resolve_conv_keys(row):
            loc = row['Location']
            prod = row['Product_Class']
            orig_store = (row.get('Origin_Store') or '').strip()
            dest_store = (row.get('Dest_Store') or '').strip()
            
            origs_list = []
            if orig_store:
                # Support comma-separated list of store names
                store_names = [s.strip() for s in orig_store.split(',') if s.strip()]
                for sn in store_names:
                    k = store_lookup_map.get((loc, sn.upper(), prod))
                    if k: origs_list.append(k)
                    else:
                        all_at_loc = loc_prod_map.get((loc, prod), [])
                        for ck in all_at_loc:
                            if ck.split('|')[2].upper() == sn.upper():
                                origs_list.append(ck); break
            if not origs_list: origs_list = loc_prod_map.get((loc, prod), [])

            dests_list = []
            if dest_store:
                # Support comma-separated list of store names
                store_names = [s.strip() for s in dest_store.split(',') if s.strip()]
                for sn in store_names:
                    k2 = store_lookup_map.get((loc, sn.upper(), prod))
                    if k2: dests_list.append(k2)
                    else:
                        all_at_loc_dest = loc_prod_map.get((loc, prod), [])
                        for ck in all_at_loc_dest:
                            if ck.split('|')[2].upper() == sn.upper():
                                dests_list.append(ck); break
            if not dests_list: dests_list = loc_prod_map.get((loc, prod), [])

            return pd.Series([",".join(origs_list), ",".join(dests_list)])

        df_conv[['Store_Keys_Origin', 'Store_Keys_Dest']] = df_conv.apply(resolve_conv_keys, axis=1)
        df_conv['Mode'] = 'CONVEYOR'
        df_conv['Origin_Location'] = df_conv['Location']
        df_conv['Dest_Location'] = df_conv['Location']
        
        for col in ['Speed_TPH']:
            if col in df_conv.columns:
                df_conv[col] = pd.to_numeric(df_conv[col], errors='coerce').fillna(0.0)

        clean_data['Move_CONVEYOR'] = df_conv.dropna(subset=['Product_Class'])

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
        "Max Wait Product (H)": "Max_Wait_Product_H",
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
        
        for col in ['N_Units', 'Speed_Knots', 'Holds_Per_Vessel', 'Payload_Per_Hold_T', 'Max_Wait_Product_H']:
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