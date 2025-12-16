# sim_run_config.py
from pathlib import Path
from dataclasses import dataclass, asdict

@dataclass
class Config:
    excel_file: Path = Path("generated_model_inputs.xlsx")
    out_dir: Path = Path("sim_outputs")

    # Simulation settings
    horizon_days: int = 365
    random_opening: bool = True
    random_seed: int | None = None

    # Demand modeling
    demand_truck_load_tons: float = 25.0
    demand_step_hours: float = 1.0

    # Transport policy
    require_full_payload: bool = False    # CHANGED: Allow partial loads so trains aren't stuck!
    debug_full_payload: bool = True       # Keep enabled to see decisions

    # Output settings
    write_csvs: bool = True
    write_inventory_csv: bool = True
    write_action_log: bool = True
    write_plots: bool = True
    plot_individual_stores: bool = True
    plot_summary: bool = True
    autoscale_default: bool = True

    # Progress logging granularity
    progress_step_pct: int = 10

    prefer_lowest_fill_origin: bool = True
    prefer_highest_room_dest: bool = True

config = Config()
run_settings = asdict(config)