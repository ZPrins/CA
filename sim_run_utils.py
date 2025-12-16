# sim_run_grok_utils.py
import pandas as pd
import math

def clean_df_cols_str(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Ensures specified columns are strings, stripped of whitespace,
    and 'nan' strings are converted to None.
    """
    for col in cols:
        if col in df.columns:
            # Force to string, strip whitespace
            df[col] = df[col].astype(str).str.strip()
            # Replace string 'nan' (from pandas string conversion of NaNs) with None
            df[col] = df[col].replace({'nan': None, 'NaN': None, '': None})
    return df

def nan_to_none(val):
    """Safely converts NaN/Inf float values to None."""
    try:
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        return val
    except Exception:
        return val