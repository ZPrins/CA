# sim_run_grok_data_clean.py
# Layer 2: Cleaning (Standardizes data types and fills missing values)

import pandas as pd
from typing import Dict, Any, Tuple
# UPDATE: Import from new utils file
from sim_run_grok_utils import clean_df_cols_str, nan_to_none


def clean_all_data(raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Applies specific cleaning rules to each DataFrame type."""
    clean_data = {}

    # --- Store ---
    df_store = raw_data.get('Store', pd.DataFrame())
    if not df_store.empty:
        df_store = clean_df_cols_str(df_store, ['Store_Key', 'Product_Class', 'Location'])
        for col in ['Capacity_T', 'Opening_Low_T', 'Opening_High_T']:
            df_store[col] = pd.to_numeric(df_store[col], errors='coerce').fillna(0.0)
        clean_data['Store'] = df_store.dropna(subset=['Store_Key'])

    # --- Make ---
    df_make = raw_data.get('Make', pd.DataFrame())
    if not df_make.empty:
        df_make = clean_df_cols_str(df_make,
                                    ['Location', 'Equipment', 'Output_Store_Key', 'Input_Store_Key', 'Product_Class',
                                     'Choice_Rule'])
        for col in ['Rate_TPH', 'Consumption_Pct', 'Step_Hours']:
            df_make[col] = pd.to_numeric(df_make[col], errors='coerce').fillna(0.0)
        clean_data['Make'] = df_make.dropna(subset=['Location', 'Equipment', 'Output_Store_Key'])

    # --- Deliver ---
    df_deliver = raw_data.get('Deliver', pd.DataFrame())
    if not df_deliver.empty:
        df_deliver = clean_df_cols_str(df_deliver, ['Store_Key'])
        df_deliver['Rate_Per_Hour'] = pd.to_numeric(df_deliver['Rate_Per_Hour'], errors='coerce').fillna(0.0)
        clean_data['Deliver'] = df_deliver.dropna(subset=['Store_Key'])

    # --- Move_TRAIN ---
    df_train = raw_data.get('Move_TRAIN', pd.DataFrame())
    if not df_train.empty:
        df_train = clean_df_cols_str(df_train,
                                     ['Product_Class', 'Origin_Location', 'Dest_Location', 'Store_Keys_Origin',
                                      'Store_Keys_Dest'])
        for col in ['N_Units', 'Payload_T', 'Load_Rate_TPH', 'Unload_Rate_TPH', 'To_Min', 'Back_Min']:
            df_train[col] = pd.to_numeric(df_train[col], errors='coerce').fillna(0.0)
        clean_data['Move_TRAIN'] = df_train.dropna(subset=['Product_Class'])

    # --- Complex Ship Data ---
    clean_data['Move_SHIP'] = clean_df_cols_str(raw_data.get('Move_SHIP', pd.DataFrame()),
                                                ['Product_Class', 'Route_Group'])
    clean_data['SHIP_ROUTES'] = clean_df_cols_str(raw_data.get('SHIP_ROUTES', pd.DataFrame()),
                                                  ['Route_Group', 'Kind', 'Location', 'Product_Class'])
    clean_data['SHIP_BERTHS'] = clean_df_cols_str(raw_data.get('SHIP_BERTHS', pd.DataFrame()),
                                                  ['Location', 'Store_Key'])
    clean_data['SHIP_DISTANCES'] = clean_df_cols_str(raw_data.get('SHIP_DISTANCES', pd.DataFrame()), [0, 1])

    return clean_data