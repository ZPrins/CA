import pandas as pd
from pathlib import Path

wb = Path(r"C:\Users\Zephien\oneforecast.com.au\OneForecast - Documents\Projects\Cement Australia\13 Simulation\06 Python\generated_model_inputs.xlsx")  # adjust if different
store = pd.read_excel(wb, sheet_name="Store")

# Helper to parse floats robustly
def f(x, default):
    try:
        # allow commas and stray spaces
        return float(str(x).replace(',', '').strip())
    except Exception:
        return default

problems = []
for i, r in store.iterrows():
    cap = f(r.get("Silo Max Capacity", ""), 1e12)
    hi  = f(r.get("Silo Opening Stock (High)", ""), 0.0)
    lo  = f(r.get("Silo Opening Stock (Low)",  ""), 0.0)
    opening = (hi + lo) / 2.0 if (hi or lo) else 0.0
    if cap <= 0 and opening > 0:
        problems.append((i, r.get("Location"), r.get("Equipment Name"), r.get("Input"), cap, opening, "capacity<=0"))
    elif opening > cap:
        problems.append((i, r.get("Location"), r.get("Equipment Name"), r.get("Input"), cap, opening, "> opening>cap"))

problems  # inspect this list