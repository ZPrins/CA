import pandas as pd
import duckdb
from pathlib import Path


def write_csv_outputs(sim, out_dir: Path, report_data: dict = None):
    """
    Write CSV outputs from simulation data.
    Uses duckdb for high-performance CSV export.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Inventory Snapshots
    inv_file = out_dir / "sim_outputs_inventory_daily.csv"
    if report_data and "df_inv" in report_data and not report_data["df_inv"].empty:
        df_inv = report_data["df_inv"]
        duckdb.query(f"COPY df_inv TO '{str(inv_file).replace('\\', '/')}' (HEADER, DELIMITER ',')")
    elif sim.inventory_snapshots:
        cols_inv = ["day", "time_h", "product_class", "location", "equipment", "input", "store_key", "level", "capacity", "fill_pct"]
        df_snap = pd.DataFrame.from_records(sim.inventory_snapshots, columns=cols_inv)
        duckdb.query(f"COPY df_snap TO '{str(inv_file).replace('\\', '/')}' (HEADER, DELIMITER ',')")

    # 2. Standardized Action Log
    log_file = out_dir / "sim_outputs_sim_log.csv"
    cols = [
        "day", "time_h", "time_d",
        "process", "event",
        "location", "equipment", "product", "qty", "qty_in",
        "from_store", "from_level",
        "to_store", "to_level",
        "route_id", "vessel_id", "ship_state"
    ]
    
    if report_data and "df_log" in report_data and not report_data["df_log"].empty:
        df_log = report_data["df_log"]
        col_str = ", ".join([c for c in cols if c in df_log.columns])
        duckdb.query(f"COPY (SELECT {col_str} FROM df_log) TO '{str(log_file).replace('\\', '/')}' (HEADER, DELIMITER ',')")
    elif sim.action_log:
        df_log_raw = pd.DataFrame.from_records(sim.action_log, columns=cols)
        duckdb.query(f"COPY df_log_raw TO '{str(log_file).replace('\\', '/')}' (HEADER, DELIMITER ',')")

    # 3. Final Levels & Unmet
    levels_file = out_dir / "sim_outputs_store_levels.csv"
    if report_data and "df_store_levels" in report_data:
        df_store_levels = report_data["df_store_levels"]
        duckdb.query(f"COPY df_store_levels TO '{str(levels_file).replace('\\', '/')}' (HEADER, DELIMITER ',')")
    else:
        df_levels = pd.DataFrame([{"Store": k, "Level": round(c.level, 1), "Capacity": c.capacity} for k, c in sim.stores.items()])
        duckdb.query(f"COPY df_levels TO '{str(levels_file).replace('\\', '/')}' (HEADER, DELIMITER ',')")

    unmet_file = out_dir / "sim_outputs_unmet_demand.csv"
    if report_data and "df_unmet" in report_data:
        df_unmet = report_data["df_unmet"]
        duckdb.query(f"COPY df_unmet TO '{str(unmet_file).replace('\\', '/')}' (HEADER, DELIMITER ',')")
    else:
        df_unmet = pd.DataFrame([{"Key": k, "Unmet": round(float(v), 2)} for k, v in sim.unmet.items()])
        duckdb.query(f"COPY df_unmet TO '{str(unmet_file).replace('\\', '/')}' (HEADER, DELIMITER ',')")