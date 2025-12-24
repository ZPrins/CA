# sim_run_grok_data_ingest.py
# Layer 1: Ingestion (Reads Excel OR CSVs)

import pandas as pd
from typing import Dict, Any
import os


def get_config_map(df: pd.DataFrame) -> Dict[str, Any]:
    """Converts a 'Settings' DataFrame into a key-value dictionary."""
    if df.empty:
        return {}
    
    # Fast conversion: set first column as index, take second column as series
    # Using iloc to handle varying column names
    try:
        s = df.set_index(df.columns[0]).iloc[:, 0]
        settings_raw = s.to_dict()
    except Exception:
        return {}

    settings = {}
    for k, v in settings_raw.items():
        if pd.isna(k) or pd.isna(v): continue
        key = str(k).strip()
        val_str = str(v).strip()
        val_lower = val_str.lower()
        
        if val_lower in ('true', 'yes', 'y', '1'):
            settings[key] = True
        elif val_lower in ('false', 'no', 'n', '0'):
            settings[key] = False
        elif val_lower in ('none', 'nan', ''):
            continue # Skip empty/null settings to allow defaults to persist
        else:
            try:
                if '.' in val_str:
                    settings[key] = float(val_str)
                else:
                    settings[key] = int(val_str)
            except ValueError:
                settings[key] = val_str
    return settings


def load_data_frames(input_file: str) -> Dict[str, pd.DataFrame]:
    """
    Loads data from a single Excel file OR multiple CSVs.
    """
    sheets = [
        "Settings", "Store", "Make", "Deliver",
        "Move_TRAIN", "Move_SHIP", "SHIP_ROUTES",
        "SHIP_BERTHS", "SHIP_DISTANCES", "Network"
    ]
    raw_data = {}

    print(f"\n--- Loading Data from: '{input_file}' ---")

    # 1. Try reading as a single Excel file
    if os.path.exists(input_file) and input_file.endswith(('.xlsx', '.xls')):
        try:
            print("  [INFO] Detected Excel file. Reading sheets...")
            # Try to use calamine engine if available for 5-10x faster Excel reading
            try:
                import python_calamine
                xls = pd.read_excel(input_file, sheet_name=None, engine='calamine')
                print("  [INFO] Using 'calamine' engine for fast Excel reading.")
            except ImportError:
                xls = pd.read_excel(input_file, sheet_name=None)

            for sheet in sheets:
                # Handle case-insensitive sheet matching
                found_sheet = None
                for k in xls.keys():
                    if k.lower() == sheet.lower():
                        found_sheet = k
                        break

                if found_sheet:
                    df = xls[found_sheet]
                    # Basic Cleanup
                    df.dropna(how='all', inplace=True)
                    df.dropna(axis=1, how='all', inplace=True)
                    df.columns = [str(col).strip() for col in df.columns]
                    raw_data[sheet] = df
                    print(f"  [OK] Loaded sheet '{sheet}' ({len(df)} rows)")
                else:
                    print(f"  [MISSING] Sheet '{sheet}' not found in Excel.")
                    raw_data[sheet] = pd.DataFrame()
            return raw_data

        except Exception as e:
            print(f"  [ERROR] Failed to read Excel file: {e}")
            print("  [INFO] Falling back to CSV search...")

    # 2. Fallback: Try reading as split CSVs
    for sheet in sheets:
        # Check for various naming conventions
        candidates = [
            f"{input_file} - {sheet}.csv",
            f"{sheet}.csv"
        ]

        df = pd.DataFrame()
        for csv_path in candidates:
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    df.dropna(how='all', inplace=True)
                    df.dropna(axis=1, how='all', inplace=True)
                    df.columns = [str(col).strip() for col in df.columns]
                    print(f"  [OK] Loaded CSV '{csv_path}' ({len(df)} rows)")
                    break
                except Exception as e:
                    print(f"  [ERROR] Failed to read '{csv_path}': {e}")

        if df.empty and sheet not in raw_data:
            print(f"  [MISSING] Could not find data for '{sheet}'")

        raw_data[sheet] = df

    return raw_data