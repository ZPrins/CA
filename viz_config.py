"""
Visualization and model configuration for supply_chain_viz.py

How to use:
- Edit values below to suit your project. All options have sensible defaults.
- Run:  python supply_chain_viz.py
  The script will look for viz_config.Config in the current working directory and use it.
- CLI flags are no longer required. A minimal --config path remains available for power users.

Notes:
- Booleans: True/False
- Distances are pixels on the canvas.
- If you change geometry/label settings, re-render the HTML to see the effect.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    # Input/Output
    in_path: str | Path = "Model Inputs.xlsx"  # Excel or CSV. Excel defaults to sheet 'Network' unless overridden
    sheet: Optional[str | int] = None           # None uses 'Network' for Excel, 0 for CSV N/A
    out_html: str | Path = "my_supply_chain.html"
    open_after: bool = True                     # Open the generated HTML in your default browser

    # Data filter
    product_class: Optional[str] = None         # e.g., "GP" to filter. None = include all

    # Layout and behavior
    height: str = "90vh"                        # Canvas height, e.g., "800px" or "90vh"
    physics: bool = False                       # Interactive physics. False keeps nodes fixed to swimlanes

    # Move simplification
    simplify_move: bool = True                  # Hide long Move edges and render pitchfork annotations

    # Swimlane/grid spacing
    grid_x_sep: int = 120                       # Horizontal distance between consecutive Levels
    grid_y_sep: int = 320                       # Vertical distance between Locations (swimlanes) — doubled
    cell_stack_sep: int = 140                   # Extra vertical separation when multiple nodes share the same lane+level

    # Pitchfork geometry
    fork_junction_dx: int = 45                  # Distance from node center to pitchfork junction (right/left)
    fork_leaf_dx: int = 120                     # Distance from node center to leaf dot (arrow tip)
    fork_prong_dy: int = 18                     # Vertical spacing between prongs

    # Colors
    move_color: str = "#ff7f0e"

    # Move label rendering (placed as tiny fixed nodes so text starts at arrow tip and is horizontal)
    move_label_font_size: int = 8               # Smaller to reduce clutter
    move_label_pad: int = 6                     # Extra pixels after the arrow tip before text starts
    move_label_bg: bool = True                  # Draw a white translucent box behind label text
    move_label_text_color: str = "#333333"
    move_label_bg_rgba: str = "rgba(255,255,255,0.85)"
    move_label_max_per_side: Optional[int] = None  # Limit prongs per side; None = unlimited

    # Input preparation (Excel)
    prepare_inputs: bool = False                # Ensure Settings/Make sheets exist and are populated

    # Legend
    show_legend: bool = True

    def resolve_in_path(self) -> Path:
        return Path(self.in_path)

    def resolve_out_path(self) -> Path:
        return Path(self.out_html)
