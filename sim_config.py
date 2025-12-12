"""
Simulation configuration for running sim_from_generated.py without CLI.

How to use:
- Edit values below to suit your project.
- Double‑click sim_run.py (or run with Python) — it will load this Config.

Notes:
- Booleans: True/False
- If in_xlsx is not found and auto_generate=True, the script will attempt to
  generate a normalized workbook from 'Model Inputs.xlsx' using helpers
  in supply_chain_viz.py and save as generated_output_name.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    # Input workbook
    in_xlsx: str | Path = "generated_model_inputs.xlsx"  # Preferred generated workbook
    product_class: Optional[str] = None                   # e.g., "GP"; None uses all

    # Overrides (optional). If None, values will be taken from Settings sheet
    override_days: Optional[int] = None                   # e.g., 60
    override_runs: Optional[int] = None                   # currently informational

    # Randomness
    use_random_opening: bool = True                       # If True, sample opening stock uniformly between Low/High
    random_seed: Optional[int] = None                     # Seed for reproducible random openings (None = unseeded)

    # Make process behavior
    # Rule to choose which product to produce when a Make unit has multiple possible outputs
    # Options: 'min_fill_pct' (produce to the store with lowest % full), 'min_level' (lowest absolute level)
    make_output_choice: str = "min_fill_pct"

    # Convenience: auto-generate a normalized workbook if missing
    auto_generate: bool = True
    source_model_xlsx: str | Path = "Model Inputs.xlsx"  # Source to generate from
    generated_output_name: str | Path = "generated_model_inputs.xlsx"

    # Outputs
    write_csvs: bool = False
    write_log: bool = False  # Write per-hour detailed simulation log (text and CSV)
    write_model_source: bool = True  # Write a standalone Python script of the built model

    # Inventory snapshot monitor
    write_daily_snapshots: bool = False                    # Write daily (or configured interval) inventory snapshots
    snapshot_hours: float = 24.0                          # Interval in hours between snapshots
    out_dir: str | Path = "sim_outputs"
    open_folder_after: bool = True

    # Plots
    plot_output_graphs: bool = True                       # Generate output graphs for Make and Store equipment
    plot_parallel: bool = True                            # Use parallel workers to speed up plotting (saves images)
    plot_workers: Optional[int] = None                    # None = auto (cpu_count); else specify number of workers
    plot_save_images: bool = True                         # Save plots as PNGs instead of interactive windows
    plot_full_horizon: bool = True                        # Pad/extend x-axis to the full simulation horizon (e.g., 8760 h)

    # Performance/behavior toggles
    fast_transport_cycle: bool = True                   # When True, coalesce unload+return timing in transporter (same outcomes)

    # Console logging / UX
    verbose_logging: bool = True                      # Print detailed progress to console
    log_with_timestamps: bool = True                  # Prefix log lines with local time
    progress_step_pct: int = 10                       # Progress update interval during run (e.g., 10%)

    # UX when double-clicking
    pause_on_finish: bool = True

    # Optional: filter to a single sheet/time bucket if your builder supports it
    # time_buckets_override: Optional[str] = None  # e.g., "Hours" | "Half Days" | "Days"

    def resolve_in_path(self) -> Path:
        return Path(self.in_xlsx)

    def resolve_out_dir(self) -> Path:
        return Path(self.out_dir)

    def resolve_source_model(self) -> Path:
        return Path(self.source_model_xlsx)

    def resolve_generated_output(self) -> Path:
        return Path(self.generated_output_name)
