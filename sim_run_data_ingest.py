# sim_run_grok_data_ingest.py
# Layer 1: Ingestion (Reads Excel OR CSVs)

import pandas as pd
from typing import Dict, Any
import os


def get_config_map(df: pd.DataFrame) -> Dict[str, Any]:
    """Converts a 'Settings' DataFrame into a key-value dictionary."""
    if df.empty:
        return {}
    settings = {}
    df = df.astype(str)
    for index, row in df.iterrows():
        try:
            key = str(row.iloc[0])  # Use index 0 in case col name varies
            value = str(row.iloc[1])
            if value.lower() in ('true', 'false'):
                settings[key] = value.lower() == 'true'
            elif '.' in value:
                try:
                    settings[key] = float(value)
                except:
                    settings[key] = value
            else:
                try:
                    settings[key] = int(value)
                except:
                    settings[key] = value
        except Exception:
            pass
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
            # Load all sheets at once for performance
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