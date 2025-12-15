# sim_run_grok_config.py
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Config:
    excel_file: Path = Path("generated_model_inputs.xlsx")
    out_dir: Path = Path("sim_outputs_grok")

    # Simulation settings
    horizon_days: int = 365
    random_opening: bool = True
    random_seed: int | None = None

    # Demand modeling
    demand_truck_load_tons: float = 25.0  # average truck load size used to satisfy demand
    demand_step_hours: float = 1.0        # time unit for demand processing (hours per step)

    # Transport policy
    require_full_payload: bool = True     # If True, trains only move when full payload can be loaded and unloaded
    debug_full_payload: bool = False      # If True, log feasibility waits for full-payload routes

    # Output settings
    write_csvs: bool = True
    write_inventory_csv: bool = True
    write_action_log: bool = True
    write_plots: bool = True
    plot_individual_stores: bool = True
    plot_summary: bool = True
    # Plot behavior
    autoscale_default: bool = True  # If True, charts open with autorange on all axes

    # Progress logging granularity (percent). Printed approximately every N% during run.
    progress_step_pct: int = 10

    prefer_lowest_fill_origin: bool = True
    prefer_highest_room_dest: bool = True


config = Config()
