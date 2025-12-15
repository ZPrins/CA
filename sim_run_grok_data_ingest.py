# sim_run_grok_data_ingest.py
# Layer 1: Ingestion (Reads CSV files into raw DataFrames)

import pandas as pd
from typing import Dict, Any


def get_config_map(df: pd.DataFrame) -> Dict[str, Any]:
    """Converts a 'Settings' DataFrame into a key-value dictionary."""
    if df.empty:
        return {}

    settings = {}
    df = df.astype(str)
    for index, row in df.iterrows():
        try:
            key = str(row['Setting'])
            value = str(row['Value'])
            # Attempt basic type conversion
            if value.lower() in ('true', 'false'):
                settings[key] = value.lower() == 'true'
            elif '.' in value:
                settings[key] = float(value)
            else:
                settings[key] = int(value)
        except ValueError:
            settings[key] = value
        except KeyError:
            pass  # Skip rows missing 'Setting' or 'Value'
    return settings


def load_data_frames(input_file: str) -> Dict[str, pd.DataFrame]:
    """
    Loads all relevant sheets from the input file into a dictionary of DataFrames.
    Assumes CSV files named 'input_file - SheetName.csv'
    """
    sheets = [
        "Settings", "Store", "Make", "Deliver",
        "Move_TRAIN", "Move_SHIP", "SHIP_ROUTES",
        "SHIP_BERTHS", "SHIP_DISTANCES", "Network"
    ]

    raw_data = {}

    for sheet in sheets:
        # NOTE: The input_file contains the common prefix (e.g. "Model Inputs.xlsx" or "generated_model_inputs.xlsx")
        csv_file = f"{input_file} - {sheet}.csv"
        try:
            # Assume no header for Network based on previous context, otherwise use header=0
            if sheet == "Network":
                df = pd.read_csv(csv_file, header=None)
            else:
                df = pd.read_csv(csv_file)

            # Remove entirely empty rows and columns
            df.dropna(how='all', inplace=True)
            df.dropna(axis=1, how='all', inplace=True)

            # Ensure column names are stripped of whitespace for robustness
            df.columns = [col.strip() for col in df.columns]

            raw_data[sheet] = df
        except FileNotFoundError:
            # This is fine, some sheets might be optional
            raw_data[sheet] = pd.DataFrame()
        except Exception as e:
            print(f"Warning: Could not load sheet '{sheet}' from '{csv_file}'. Error: {e}")
            raw_data[sheet] = pd.DataFrame()

    return raw_data