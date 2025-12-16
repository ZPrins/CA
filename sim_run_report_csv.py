# sim_run_grok_report_csv.py
import pandas as pd
from pathlib import Path


def write_csv_outputs(sim, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Inventory Snapshots
    if sim.inventory_snapshots:
        df_snap = pd.DataFrame(sim.inventory_snapshots)
        # Sort and Save
        df_snap.to_csv(out_dir / "sim_outputs_inventory_daily.csv", index=False)

    # 2. Standardized Action Log
    if sim.action_log:
        df_log = pd.DataFrame(sim.action_log)

        # Define clean column order based on User Request
        cols = [
            "day", "time_h",
            "process", "event",
            "location", "equipment", "product", "qty",
            "from_store", "from_level",
            "to_store", "to_level",
            "route_id"
        ]

        # Ensure day/time exist
        df_log["day"] = (pd.to_numeric(df_log["time_h"], errors='coerce') / 24).astype(int) + 1

        # Filter only existing columns
        final_cols = [c for c in cols if c in df_log.columns]

        df_log[final_cols].to_csv(out_dir / "sim_outputs_sim_log.csv", index=False)

    # 3. Final Levels & Unmet (No changes needed here)
    ending = []
    for key, cont in sim.stores.items():
        ending.append({"Store": key, "Level": round(cont.level, 1), "Capacity": cont.capacity})
    pd.DataFrame(ending).to_csv(out_dir / "sim_outputs_store_levels.csv", index=False)

    unmet_rows = [{"Key": k, "Unmet": round(float(v), 2)} for k, v in sim.unmet.items()]
    pd.DataFrame(unmet_rows).to_csv(out_dir / "sim_outputs_unmet_demand.csv", index=False)

    print(f"CSV outputs written to {out_dir}")