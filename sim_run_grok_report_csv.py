# sim_run_grok_report_csv.py
import pandas as pd
from pathlib import Path


def write_csv_outputs(sim, out_dir: Path):
    """
    Writes simulation artifacts (snapshots, logs, store levels) to CSV.
    Accepts the 'sim' object to access results.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Inventory snapshots with day number
    if sim.inventory_snapshots:
        df_snap = pd.DataFrame(sim.inventory_snapshots)
        # Ensure time_h is numeric
        df_snap["time_h"] = pd.to_numeric(df_snap["time_h"], errors="coerce")
        df_snap = df_snap.dropna(subset=["time_h"])
        df_snap["day"] = (df_snap["time_h"] / 24).astype(int) + 1
        # Sort for readability
        df_snap = df_snap.sort_values(["time_h", "store_key"])
        df_snap.to_csv(out_dir / "sim_outputs_inventory_daily.csv", index=False)

    # Action log with day
    if sim.action_log:
        df_log = pd.DataFrame(sim.action_log)
        df_log["time_h"] = pd.to_numeric(df_log["time_h"], errors="coerce")
        df_log = df_log.dropna(subset=["time_h"])

        # Round known quantity fields to 2 decimals
        for qc in ["qty", "qty_t", "unmet"]:
            if qc in df_log.columns:
                df_log[qc] = pd.to_numeric(df_log[qc], errors="coerce").round(2)

        df_log["day"] = (df_log["time_h"] / 24).astype(int) + 1
        df_log["hour"] = (df_log["time_h"] // 1).astype(int) + 1

        # Reorder columns to put time first
        cols = ["day", "hour", "time_h", "event"] + [c for c in df_log.columns if
                                                     c not in ["day", "hour", "time_h", "event"]]
        df_log[cols].to_csv(out_dir / "sim_outputs_sim_log.csv", index=False)

    # Final Store levels
    ending = []
    for key, cont in sim.stores.items():
        parts = key.split("|")
        pc = parts[0] if len(parts) > 0 else ""
        loc = parts[1] if len(parts) > 1 else ""
        eq = parts[2] if len(parts) > 2 else ""
        inp = parts[3] if len(parts) > 3 else ""
        ending.append({
            "Store": key,
            "Product_Class": pc,
            "Location": loc,
            "Equipment": eq,
            "Input": inp,
            "Level": round(cont.level, 1),
            "Capacity": cont.capacity,
            "Fill_Pct": round(cont.level / cont.capacity, 3) if cont.capacity > 0 else 0
        })
    pd.DataFrame(ending).to_csv(out_dir / "sim_outputs_store_levels.csv", index=False)

    # Unmet demand
    unmet_rows = [{"Key": k, "Unmet": round(float(v), 2)} for k, v in sim.unmet.items()]
    pd.DataFrame(unmet_rows).to_csv(out_dir / "sim_outputs_unmet_demand.csv", index=False)

    print(f"CSV outputs written to {out_dir}")