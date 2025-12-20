import pandas as pd
import duckdb
from pathlib import Path
import time
import os


def _safe_duckdb_copy(df, file_path: Path):
    """
    Safely copy a DataFrame to CSV using DuckDB.
    If the file is locked, attempts to write to a fallback path.
    """
    target_path = str(file_path).replace('\\', '/')
    
    # Try to remove the file if it exists to check if it's locked
    if file_path.exists():
        try:
            file_path.unlink()
        except OSError:
            # File is likely locked. Use a fallback name.
            timestamp = int(time.time())
            fallback_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
            file_path = file_path.parent / fallback_name
            target_path = str(file_path).replace('\\', '/')
            print(f"  [WARN] File locked. Writing to fallback: {file_path.name}")

    try:
        duckdb.query(f"COPY df TO '{target_path}' (HEADER, DELIMITER ',')")
    except Exception as e:
        print(f"  [ERROR] Failed to write CSV to {target_path}: {e}")


def write_csv_outputs(sim, out_dir: Path, report_data: dict = None):
    """
    Write CSV outputs from simulation data.
    Uses duckdb for high-performance CSV export.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Inventory Snapshots
    inv_file = out_dir / "sim_outputs_inventory_daily.csv"
    if report_data and "df_inv" in report_data and not report_data["df_inv"].empty:
        df = report_data["df_inv"]
        _safe_duckdb_copy(df, inv_file)
    elif sim.inventory_snapshots:
        cols_inv = ["day", "time_h", "product_class", "location", "equipment", "input", "store_key", "level", "capacity", "fill_pct"]
        df = pd.DataFrame.from_records(sim.inventory_snapshots, columns=cols_inv)
        _safe_duckdb_copy(df, inv_file)

    # 2. Standardized Action Log
    log_file = out_dir / "sim_outputs_sim_log.csv"
    cols = [
        "day", "time_h", "time_d",
        "process", "event",
        "location", "equipment", "product", "qty", "time", "qty_in",
        "from_store", "from_level",
        "to_store", "to_level",
        "route_id", "vessel_id", "ship_state"
    ]
    
    if report_data and "df_log" in report_data and not report_data["df_log"].empty:
        df_log = report_data["df_log"]
        col_str = ", ".join([c for c in cols if c in df_log.columns])
        df = duckdb.query(f"SELECT {col_str} FROM df_log").df()
        _safe_duckdb_copy(df, log_file)
    elif sim.action_log:
        df = pd.DataFrame.from_records(sim.action_log, columns=cols)
        _safe_duckdb_copy(df, log_file)

    # 3. Final Levels & Unmet
    levels_file = out_dir / "sim_outputs_store_levels.csv"
    if report_data and "df_store_levels" in report_data:
        df = report_data["df_store_levels"]
    else:
        df = pd.DataFrame([{"Store": k, "Level": round(c.level, 1), "Capacity": c.capacity} for k, c in sim.stores.items()])
    if not df.empty:
        _safe_duckdb_copy(df, levels_file)

    unmet_file = out_dir / "sim_outputs_unmet_demand.csv"
    if report_data and "df_unmet" in report_data:
        df = report_data["df_unmet"]
    else:
        df = pd.DataFrame([{"Key": k, "Unmet": round(float(v), 2)} for k, v in sim.unmet.items()])
    if not df.empty:
        _safe_duckdb_copy(df, unmet_file)