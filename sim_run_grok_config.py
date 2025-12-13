# sim_run_grok_config.py
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Config:
    excel_file: Path = Path("generated_model_inputs.xlsx")
    out_dir: Path = Path("sim_outputs")

    # Simulation settings
    horizon_days: int = 365
    random_opening: bool = True
    random_seed: int | None = None

    # Output settings
    write_csvs: bool = True
    write_inventory_csv: bool = True
    write_action_log: bool = True
    write_plots: bool = True
    plot_individual_stores: bool = True
    plot_summary: bool = True

    prefer_lowest_fill_origin: bool = True
    prefer_highest_room_dest: bool = True


config = Config()
