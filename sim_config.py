"""
Simulation configuration for running sim_from_generated.py without CLI.

How to use:
- Edit values below to suit your project.
- Double‑click run_sim.py (or run with Python) — it will load this Config.

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

    # Convenience: auto-generate a normalized workbook if missing
    auto_generate: bool = True
    source_model_xlsx: str | Path = "Model Inputs.xlsx"  # Source to generate from
    generated_output_name: str | Path = "generated_model_inputs.xlsx"

    # Outputs
    write_csvs: bool = True
    out_dir: str | Path = "sim_outputs"
    open_folder_after: bool = True

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
