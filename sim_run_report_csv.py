# sim_run_grok_report_csv.py
import pandas as pd
from pathlib import Path


def write_csv_outputs(sim, out_dir: Path, report_data: dict = None):
    """
    Write CSV outputs from simulation data.
    If report_data is provided, use precomputed DataFrames for speed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Inventory Snapshots
    if report_data and "df_inv" in report_data and not report_data["df_inv"].empty:
        report_data["df_inv"].to_csv(out_dir / "sim_outputs_inventory_daily.csv", index=False)
    elif sim.inventory_snapshots:
        df_snap = pd.DataFrame(sim.inventory_snapshots)
        df_snap.to_csv(out_dir / "sim_outputs_inventory_daily.csv", index=False)

    # 2. Standardized Action Log (with new double-entry columns)
    if report_data and "df_log" in report_data and not report_data["df_log"].empty:
        df_log = report_data["df_log"].copy()
        cols = [
            "day", "time_h", "time_d",
            "process", "event",
            "location", "equipment", "product", "qty",
            "store_key", "level",
            "route_id", "vessel_id", "ship_state"
        ]
        final_cols = [c for c in cols if c in df_log.columns]
        df_log[final_cols].to_csv(out_dir / "sim_outputs_sim_log.csv", index=False)
    elif sim.action_log:
        df_log = pd.DataFrame(sim.action_log)
        cols = [
            "day", "time_h", "time_d",
            "process", "event",
            "location", "equipment", "product", "qty",
            "store_key", "level",
            "route_id", "vessel_id", "ship_state"
        ]
        df_log["time_h"] = pd.to_numeric(df_log["time_h"], errors='coerce')
        df_log["time_d"] = pd.to_numeric(df_log.get("time_d", df_log["time_h"] // 24), errors='coerce').fillna(0).astype(int)
        df_log["day"] = df_log["time_d"].astype(int) + 1
        final_cols = [c for c in cols if c in df_log.columns]
        df_log[final_cols].to_csv(out_dir / "sim_outputs_sim_log.csv", index=False)

    # 3. Final Levels & Unmet
    if report_data and "df_store_levels" in report_data:
        report_data["df_store_levels"].to_csv(out_dir / "sim_outputs_store_levels.csv", index=False)
    else:
        ending = []
        for key, cont in sim.stores.items():
            ending.append({"Store": key, "Level": round(cont.level, 1), "Capacity": cont.capacity})
        pd.DataFrame(ending).to_csv(out_dir / "sim_outputs_store_levels.csv", index=False)

    if report_data and "df_unmet" in report_data:
        report_data["df_unmet"].to_csv(out_dir / "sim_outputs_unmet_demand.csv", index=False)
    else:
        unmet_rows = [{"Key": k, "Unmet": round(float(v), 2)} for k, v in sim.unmet.items()]
        pd.DataFrame(unmet_rows).to_csv(out_dir / "sim_outputs_unmet_demand.csv", index=False)
