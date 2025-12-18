"""
Visualization and model configuration for supply_chain_viz.py

How to use:
- Edit values below to suit your project. All options have sensible defaults.
- Run:  python supply_chain_viz.py
  The script will look for supply_chain_viz_config.Config in the current working directory and use it.
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
    pc_order: Optional[list[str]] = None        # Optional fixed order of product-class columns (e.g., ["GP","FA","SG","FC"])

    # Layout and behavior
    height: str = "90vh"                        # Canvas height, e.g., "800px" or "90vh"
    physics: bool = False                       # Interactive physics. False keeps nodes fixed to swimlanes

    # Move simplification
    simplify_move: bool = True                  # Hide long Move edges and render pitchfork annotations

    # Swimlane/grid spacing
    grid_x_sep: int = 200                       # Horizontal distance between consecutive Levels (within a product-class panel)
    grid_y_sep: int = 800                       # Vertical distance between Locations (swimlanes)
    grid_pc_sep: int = 1400                     # Extra horizontal separation between Product Class panels (matrix columns)
    cell_stack_sep: int = 250                   # Extra vertical separation when multiple nodes share the same lane+level

    # Pitchfork geometry
    fork_junction_dx: int = 45                  # Distance from node center to pitchfork junction (right/left)
    fork_leaf_dx: int = 120                     # Distance from node center to leaf dot (arrow tip)
    fork_prong_dy: int = 18                     # Vertical spacing between prongs

    # Swimlane wireframes (one rectangle per Location across all product-class columns)
    swimlane_wireframes: bool = True            # Draw wireframe rectangles for each location row
    swimlane_wireframe_color: str = "rgba(0,0,0,0.25)"  # Stroke color
    swimlane_wireframe_stroke: int = 2          # Stroke width in pixels
    swimlane_wireframe_corner_radius: int = 10  # Corner radius for rounded rects (DOM pixels)
    swimlane_wireframe_margin_x: int = 300       # Extra horizontal margin beyond min/max content (world px)
    swimlane_wireframe_margin_y: int = 80       # Extra vertical margin within the swimlane band (world px)

    # Swimlane labels (drawn in the top-left of each wireframe)
    swimlane_labels: bool = True                # Show location name labels in each swimlane
    swimlane_label_text_color: str = "#FF0101"  # Label text color
    swimlane_label_font_px: int = 48            # Font size in pixels
    swimlane_label_font_family: str = "Arial, sans-serif"  # Canvas font family
    swimlane_label_margin_left: int = 10        # Left inset from the wireframe's left edge (world px)
    swimlane_label_margin_top: int = 6          # Top inset from the wireframe's top edge (world px)

    # Per-product-class labels inside each swimlane (bottom-centered in each PC column)
    swimlane_pc_labels: bool = True             # Show product-class codes (e.g., GP, SG, FA) at the bottom of each lane
    swimlane_pc_label_text_color: str = "#0000FF"  # Label text color
    swimlane_pc_label_font_px: int = 36        # Font size in pixels
    swimlane_pc_label_font_family: str = "Arial, sans-serif"  # Canvas font family
    swimlane_pc_label_margin_bottom: int = 10   # Bottom inset from the wireframe's bottom edge (world px)

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
    # Ensure Settings, Make, Store, and Delivery sheets exist and are populated
    prepare_inputs: bool = True

    # Legend
    show_legend: bool = True

    def resolve_in_path(self) -> Path:
        return Path(self.in_path)

    def resolve_out_path(self) -> Path:
        return Path(self.out_html)
