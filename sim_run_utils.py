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
            # Use .apply if it's a DataFrame (multiple columns with same name)
            target = df[col]
            if isinstance(target, pd.DataFrame):
                df[col] = target.astype(str).apply(lambda s: s.str.strip())
            else:
                df[col] = target.astype(str).str.strip()
            
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

def try_import_orjson():
    """Try to import orjson and return it if successful, otherwise return None."""
    try:
        import orjson
        return orjson
    except ImportError:
        return None