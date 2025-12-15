r"""
Supply Chain Visualizer and SimPy model scaffold

This script reads an Excel or CSV describing an end-to-end supply chain using the
following minimal columns (case-insensitive; spaces/underscores tolerated):

- Level
- Product Class
- Location
- Equipment Name
- Input
- Process
- Output
- Next Process
- Next Location
- Next Equipment

It builds a directed multi-graph of equipment-at-location nodes and process edges
and exports an interactive HTML visualization (hierarchical by Level). It also
contains a small SimPy scaffold showing how the same dataframe could be used to
construct resources and processes for a discrete-event simulation.

Usage examples (PowerShell):

  # Visualize from Excel
  python .\supply_chain_viz.py --in "C:\\path\\to\\supply_chain.xlsx" --sheet 0 --out supply_chain.html

  # Visualize from CSV and filter a product class
  python .\supply_chain_viz.py --in .\sample_supply_chain.csv --product-class GP

  # Write the sample rows (from the screenshot) to CSV for a quick try
  python .\supply_chain_viz.py --write-sample sample_supply_chain.csv --out sample_sc.html

Dependencies: pandas, networkx, pyvis, simpy
Install:  pip install -r requirements.txt

Author: Junie (JetBrains autonomous programmer)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

import pandas as pd
import networkx as nx
import webbrowser

try:
    from pyvis.network import Network
except Exception:  # pragma: no cover - optional at import time
    Network = None  # type: ignore

try:
    import simpy  # noqa: F401  # used in scaffold
except Exception:  # pragma: no cover - optional at import time
    simpy = None  # type: ignore

# pyvis relies on Jinja2 for HTML templating
try:
    import jinja2  # noqa: F401
except Exception:  # pragma: no cover - optional at import time
    jinja2 = None  # type: ignore

# Excel IO helpers
try:
    import openpyxl  # noqa: F401
except Exception:  # pragma: no cover - optional at import time
    openpyxl = None  # type: ignore


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to snake_case keys used internally.

    Accepts variants like 'Equipment', 'Equipment Name', 'Next equipment', etc.
    """
    mapping: Dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_").replace("-", "_")
        key = key.replace("__", "_")
        # standardize common variants
        if key in {"equipment", "equipment_name", "equip_name"}:
            key = "equipment_name"
        elif key in {"product_class", "product", "class"}:
            key = "product_class"
        elif key in {"next_equipment", "nextequipment", "next_equip"}:
            key = "next_equipment"
        elif key in {"next_location", "nextloc"}:
            key = "next_location"
        elif key in {"next_process", "nextproc"}:
            key = "next_process"
        mapping[col] = key
    df = df.rename(columns=mapping)
    return df


REQUIRED_COLS = [
    "level",
    "product_class",
    "location",
    "equipment_name",
    "input",
    "process",
    "output",
    "next_process",
    "next_location",
    "next_equipment",
]


def _validate_df(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input is missing required columns: {missing}. Columns present: {list(df.columns)}"
        )


def read_table(path: Path, sheet: Optional[str | int] = None) -> pd.DataFrame:
    """Read Excel or CSV by extension and normalize columns.

    Excel default behavior: if no sheet is specified, try 'Network' first, otherwise fall back to the first sheet.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        if sheet is None:
            try:
                df = pd.read_excel(path, sheet_name="Network")
            except Exception:
                df = pd.read_excel(path, sheet_name=0)
        else:
            df = pd.read_excel(path, sheet_name=sheet)
    elif path.suffix.lower() in {".csv"}:
        df = pd.read_csv(path)
    else:
        raise ValueError(
            f"Unsupported file type: {path.suffix}. Use Excel (.xlsx) or CSV (.csv)."
        )

    df = _normalize_columns(df)
    # Coerce to expected types and strip whitespace
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
    # Some spreadsheets put 'nan' strings for blanks; normalize to empty
    df = df.replace({"nan": "", "None": "", "NONE": ""})

    _validate_df(df)
    # Coerce level to int when possible
    try:
        df["level"] = pd.to_numeric(df["level"], errors="coerce").astype("Int64")
    except Exception:
        pass
    return df


def node_id(location: str, equipment: str, product_class: Optional[str] = None, material: Optional[str] = None) -> str:
    """Return a stable node identifier including optional product class and material.

    Composition rules (backward compatible):
    - Base: "Equipment@Location"
    - If product_class provided and non-empty: append "#<product_class>"
    - If material (input/output) provided and non-empty: append "|<material>"
    This ensures nodes for different product classes and materials do not merge.
    """
    base = f"{equipment}@{location}"
    suffix = ""
    pc = None if product_class is None else str(product_class).strip()
    mat = None if material is None else str(material).strip()
    if pc:
        suffix += f"#{pc}"
    if mat:
        suffix += f"|{mat}"
    return base + suffix


def build_graph(df: pd.DataFrame, product_class: Optional[str] = None) -> nx.MultiDiGraph:
    """Build a MultiDiGraph with nodes as Equipment@Location and edges by process transitions.

    Also infers destination node levels when possible: if a row has level=L for the
    source node, the destination node is assigned level L+1 (preserving the max
    level across multiple incoming edges). This improves hierarchical layout.
    """
    if product_class:
        df = df[df["product_class"].str.upper() == product_class.upper()].copy()
        if df.empty:
            raise ValueError(f"No rows found for product class '{product_class}'.")

    G = nx.MultiDiGraph()

    # Compute row order by Location (independent of Product Class) so that all product classes
    # for the same location share the same swimlane row in the matrix layout.
    df_idx = df.reset_index().rename(columns={"index": "_row_idx"})
    df_idx["location"] = df_idx["location"].astype(str).str.strip()
    df_idx["next_location"] = df_idx["next_location"].astype(str).str.strip()
    df_idx["product_class"] = df_idx["product_class"].astype(str).str.strip()

    # Track first appearance index and minimum level per Location
    loc_first_idx: Dict[str, int] = {}
    loc_min_level: Dict[str, int] = {}

    for i, r in df_idx.iterrows():
        # Current row's source location
        loc = r["location"]
        if loc and loc not in loc_first_idx:
            loc_first_idx[loc] = int(r.get("_row_idx", i))
        try:
            lvl = int(r["level"]) if pd.notna(r["level"]) else 10**9
        except Exception:
            lvl = 10**9
        cur_min = loc_min_level.get(loc, 10**9)
        if lvl < cur_min:
            loc_min_level[loc] = lvl
        # Next location also influences first appearance (ordering) even if level is unknown here
        nloc = r["next_location"]
        if nloc and nloc not in loc_first_idx:
            loc_first_idx[nloc] = int(r.get("_row_idx", i))

    # All locations observed across both columns
    all_locs = set(str(x).strip() for x in df_idx["location"]) | set(str(x).strip() for x in df_idx["next_location"]) 
    all_locs = [loc for loc in all_locs if loc]

    def _loc_rank(loc: str) -> tuple:
        return (loc_min_level.get(loc, 10**9), loc_first_idx.get(loc, 10**9), loc)

    ordered_locations = sorted(all_locs, key=_loc_rank)
    loc_to_row_index: Dict[str, int] = {loc: i for i, loc in enumerate(ordered_locations)}

    # Add nodes and infer levels for destinations
    for _, row in df.iterrows():
        src_level_val = int(row["level"]) if pd.notna(row["level"]) else 0
        pc = str(row.get("product_class", "")).strip()
        # Node id includes product class and material (input) to avoid overlap; assign row index by Location
        src_node = node_id(row["location"], row["equipment_name"], pc, str(row.get("input", "")).strip()) if row["equipment_name"] else None
        if src_node:
            G.add_node(
                src_node,
                title=f"Product Class: {pc}<br>Location: {row['location']}<br>Equipment: {row['equipment_name']}",
                level=src_level_val,
                product_class=pc,
                location=str(row["location"]).strip(),
                equipment=row["equipment_name"],
                material=str(row.get("input", "")).strip(),
                loc_index=loc_to_row_index.get(str(row["location"]).strip(), 0),
            )
        # destination node might be blank (terminal)
        if str(row["next_location"]).strip() and str(row["next_equipment"]).strip():
            dst_node = node_id(row["next_location"], row["next_equipment"], pc, str(row.get("output", "")).strip())
            # infer destination level from source (place to the right)
            dst_level = src_level_val + 1
            if dst_node not in G:
                G.add_node(
                    dst_node,
                    title=f"Product Class: {pc}<br>Location: {row['next_location']}<br>Equipment: {row['next_equipment']}",
                    product_class=pc,
                    location=str(row["next_location"]).strip(),
                    equipment=row["next_equipment"],
                    material=str(row.get("output", "")).strip(),
                    level=dst_level,
                    loc_index=loc_to_row_index.get(str(row["next_location"]).strip(), 0),
                )
            else:
                prev = G.nodes[dst_node].get("level")
                if prev is None or dst_level > prev:
                    G.nodes[dst_node]["level"] = dst_level

    # Add edges with attributes and simultaneously collect per-node process/output counts
    color_map = {
        "make": "#2ca02c",
        "store": "#1f77b4",
        "move": "#ff7f0e",
        "deliver": "#9467bd",
        "none": "#7f7f7f",
    }

    proc_counts: Dict[str, Dict[str, int]] = {}
    out_counts: Dict[str, Dict[str, int]] = {}
    dest_proc_counts: Dict[str, Dict[str, int]] = {}

    for _, row in df.iterrows():
        pc_row = str(row.get("product_class", "")).strip()
        src = node_id(row["location"], row["equipment_name"], pc_row, str(row.get("input", "")).strip()) if row["equipment_name"] else None
        if not src or src not in G:
            continue
        next_loc = str(row["next_location"]).strip()
        next_eq = str(row["next_equipment"]).strip()
        if not (next_loc and next_eq):
            # terminal step — no outgoing edge
            continue
        dst = node_id(next_loc, next_eq, pc_row, str(row.get("output", "")).strip())
        label = f"{row['output']}"
        process_key = str(row["process"]).strip().lower()
        # Count primary process and output for the SOURCE node
        if src:
            proc_counts.setdefault(src, {})[process_key] = proc_counts.setdefault(src, {}).get(process_key, 0) + 1
            out_val = str(row["output"]).strip()
            out_counts.setdefault(src, {})[out_val] = out_counts.setdefault(src, {}).get(out_val, 0) + 1
        # Also count incoming process at the DEST node (for terminals with no outgoing)
        dest_proc_counts.setdefault(dst, {})[process_key] = dest_proc_counts.setdefault(dst, {}).get(process_key, 0) + 1
        # Use neutral edge color; node colors will reflect process legend
        edge_color = "#888888"
        G.add_edge(
            src,
            dst,
            label=label,
            title=f"Process: {row['process']}<br>Input: {row['input']}<br>Output: {row['output']}",
            color=edge_color,
            process=row["process"],
            level=int(row["level"]) if pd.notna(row["level"]) else None,
        )

    # Determine primary process and output for each node (by most frequent outgoing)
    for n in G.nodes:
        pc = proc_counts.get(n, {})
        oc = out_counts.get(n, {})
        pk = None
        if pc:
            pk = max(pc.items(), key=lambda kv: kv[1])[0]
        elif dest_proc_counts.get(n, {}):
            pk = max(dest_proc_counts[n].items(), key=lambda kv: kv[1])[0]
        # Heuristic: enforce Deliver for TRUCK (and ROAD) equipment
        eq_name = str(G.nodes[n].get("equipment", "")).lower()
        if "truck" in eq_name:
            pk = "deliver"
        elif pk is None and ("road" in eq_name):
            pk = "deliver"
        # Prefer Deliver if incoming contains it and node has no outgoing edges
        if G.out_degree(n) == 0 and dest_proc_counts.get(n, {}).get("deliver", 0) > 0:
            pk = "deliver"
        if pk:
            pretty_proc = pk.capitalize()
            G.nodes[n]["primary_process_key"] = pk
            G.nodes[n]["primary_process_label"] = pretty_proc
            G.nodes[n]["node_color"] = color_map.get(pk, "#7f7f7f")
        if oc:
            primary_out = max(oc.items(), key=lambda kv: kv[1])[0]
            G.nodes[n]["primary_output"] = primary_out

    return G


def export_pyvis(G: nx.MultiDiGraph, out_html: Path, config) -> None:
    if Network is None:
        raise RuntimeError(
            "pyvis is not installed. Install with 'pip install pyvis' or via requirements.txt."
        )
    if 'jinja2' not in sys.modules and jinja2 is None:
        raise RuntimeError(
            "Jinja2 is required by pyvis but is not installed. Install with 'pip install jinja2' or via requirements.txt."
        )

    # Short aliases
    height = getattr(config, 'height', '800px')
    enable_physics = bool(getattr(config, 'physics', False))
    simplify_move = bool(getattr(config, 'simplify_move', True))

    net = Network(height=height, width="100%", directed=True, notebook=False)
    net.toggle_hide_edges_on_drag(True)

    # Choose layout mode: if all nodes have level and loc_index, fix positions by (location column, level row).
    all_have_level = all(G.nodes[n].get("level") is not None for n in G.nodes)
    all_have_locidx = all("loc_index" in G.nodes[n] for n in G.nodes)
    use_fixed_grid = all_have_level and all_have_locidx

    XSEP = int(getattr(config, 'grid_x_sep', 120))
    YSEP = int(getattr(config, 'grid_y_sep', 160))
    PCSEP = int(getattr(config, 'grid_pc_sep', 560))

    # Determine product-class columns present in the graph
    pcs_present: list[str] = []
    for _n, _d in G.nodes(data=True):
        pcv = str(_d.get('product_class', '')).strip()
        if pcv and pcv not in pcs_present:
            pcs_present.append(pcv)
    # Apply preferred order if provided in config; append any extras not listed
    try:
        pc_order_cfg = getattr(config, 'pc_order', None)
    except Exception:
        pc_order_cfg = None
    if isinstance(pc_order_cfg, (list, tuple)):
        pc_order = [str(x).strip() for x in pc_order_cfg if str(x).strip() in pcs_present]
        for pc in pcs_present:
            if pc not in pc_order:
                pc_order.append(pc)
    else:
        pc_order = sorted(pcs_present)
    pc_to_col: Dict[str, int] = {pc: i for i, pc in enumerate(pc_order)}

    if use_fixed_grid and not enable_physics:
        # Disable physics and hierarchical engine; we will pin nodes at grid coordinates.
        net.toggle_physics(False)
        net.set_options(
            '{"interaction": {"navigationButtons": true, "keyboard": true, "zoomView": true, "dragView": true, "dragNodes": false, "multiselect": true}, "edges": {"smooth": false, "arrows": {"to": {"enabled": true, "scaleFactor": 0.7}}, "font": {"size": 10, "align": "middle", "vadjust": 0, "strokeWidth": 3, "strokeColor": "#ffffff", "background": "rgba(255,255,255,0.6)"}}, "nodes": {"font": {"size": 0}, "shapeProperties": {"useImageSize": true, "useBorderWithImage": false}, "scaling": {"min": 1, "max": 1}}}'
        )
        # Pre-compute positions: rows by Location (loc_index), columns by Product Class panel + Level within panel
        positions: Dict[str, Tuple[int, int]] = {}
        for n, d in G.nodes(data=True):
            pcv = str(d.get("product_class", "")).strip()
            pc_col = pc_to_col.get(pcv, 0)
            positions[n] = (pc_col * PCSEP + int(d.get("level", 0)) * XSEP, int(d.get("loc_index", 0)) * YSEP)
        # De-overlap within each (location, product_class, level) cell using a barycentric heuristic
        groups: Dict[Tuple[int, int, int], list[str]] = {}
        base_y_cache: Dict[str, int] = {}
        for n, d in G.nodes(data=True):
            pcv = str(d.get("product_class", "")).strip()
            pc_col = pc_to_col.get(pcv, 0)
            key = (int(d.get("loc_index", 0)), pc_col, int(d.get("level", 0)))
            groups.setdefault(key, []).append(n)
            base_y_cache[n] = int(d.get("loc_index", 0)) * YSEP

        def _avg_neighbor_y(nodes_list: list[str], lvl: int) -> Dict[str, float]:
            scores: Dict[str, float] = {}
            for nn in nodes_list:
                ys: list[int] = []
                # Prefer neighbors at next level (lvl+1); otherwise look at previous level (lvl-1)
                for _, dst, edata in G.out_edges(nn, data=True):
                    try:
                        dst_lvl = int(G.nodes[dst].get("level", -999))
                    except Exception:
                        dst_lvl = -999
                    if dst_lvl == lvl + 1:
                        ys.append(base_y_cache.get(dst, 0))
                if not ys:
                    for src, _, edata in G.in_edges(nn, data=True):
                        try:
                            src_lvl = int(G.nodes[src].get("level", -999))
                        except Exception:
                            src_lvl = -999
                        if src_lvl == lvl - 1:
                            ys.append(base_y_cache.get(src, 0))
                if ys:
                    scores[nn] = sum(ys) / len(ys)
                else:
                    scores[nn] = base_y_cache.get(nn, 0)
            return scores
        
        STAG = int(getattr(config, 'cell_stack_sep', 140))
        for (loc_i, _pc_col, lvl), ns in groups.items():
            if len(ns) > 1:
                scores = _avg_neighbor_y(ns, lvl)
                ns_sorted = sorted(ns, key=lambda n: (scores.get(n, 0), n))
                k = len(ns_sorted)
                for i, nn in enumerate(ns_sorted):
                    x, y = positions[nn]
                    y_off = (i - (k - 1) / 2) * STAG  # configurable vertical stagger
                    positions[nn] = (x, y + y_off)
    else:
        # Keep hierarchical or grid with physics for interactive exploration
        net.toggle_physics(True)
        net.set_options(
            '{"layout": {"hierarchical": {"enabled": true, "levelSeparation": 160, "nodeSpacing": 200, "treeSpacing": 220, "direction": "LR", "sortMethod": "hubsize", "blockShifting": true, "edgeMinimization": true, "parentCentralization": true}}, "physics": {"enabled": true, "solver": "hierarchicalRepulsion", "stabilization": {"enabled": true, "iterations": 200, "fit": true}}, "interaction": {"navigationButtons": true, "keyboard": true, "zoomView": true, "dragView": true, "dragNodes": true, "multiselect": true}, "nodes": {"font": {"size": 0}, "shapeProperties": {"useImageSize": true, "useBorderWithImage": false}, "scaling": {"min": 1, "max": 1}}, "edges": {"smooth": false, "arrows": {"to": {"enabled": true, "scaleFactor": 0.7}}, "font": {"size": 10, "align": "middle", "vadjust": 0, "strokeWidth": 3, "strokeColor": "#ffffff", "background": "rgba(255,255,255,0.6)"}}}'
        )
        # If we have grid positions but physics is enabled, use them as initial hints (not fixed)
        positions: Dict[str, Tuple[int, int]] = {}
        if use_fixed_grid:
            for n, d in G.nodes(data=True):
                pcv = str(d.get("product_class", "")).strip()
                pc_col = pc_to_col.get(pcv, 0)
                positions[n] = (pc_col * PCSEP + int(d.get("level", 0)) * XSEP, int(d.get("loc_index", 0)) * YSEP)
            # Apply initial de-overlap within each (location, product_class, level) cell so nodes start spaced apart
            groups: Dict[Tuple[int, int, int], list[str]] = {}
            base_y_cache: Dict[str, int] = {}
            for n, d in G.nodes(data=True):
                pcv = str(d.get("product_class", "")).strip()
                pc_col = pc_to_col.get(pcv, 0)
                key = (int(d.get("loc_index", 0)), pc_col, int(d.get("level", 0)))
                groups.setdefault(key, []).append(n)
                base_y_cache[n] = int(d.get("loc_index", 0)) * YSEP

            def _avg_neighbor_y_init(nodes_list: list[str], lvl: int) -> Dict[str, float]:
                scores: Dict[str, float] = {}
                for nn in nodes_list:
                    ys: list[int] = []
                    for _, dst, edata in G.out_edges(nn, data=True):
                        try:
                            dst_lvl = int(G.nodes[dst].get("level", -999))
                        except Exception:
                            dst_lvl = -999
                        if dst_lvl == lvl + 1:
                            ys.append(base_y_cache.get(dst, 0))
                    if not ys:
                        for src, _, edata in G.in_edges(nn, data=True):
                            try:
                                src_lvl = int(G.nodes[src].get("level", -999))
                            except Exception:
                                src_lvl = -999
                            if src_lvl == lvl - 1:
                                ys.append(base_y_cache.get(src, 0))
                    if ys:
                        scores[nn] = sum(ys) / len(ys)
                    else:
                        scores[nn] = base_y_cache.get(nn, 0)
                return scores

            STAG = int(getattr(config, 'cell_stack_sep', 140))
            for (loc_i, lvl), ns in groups.items():
                if len(ns) > 1:
                    scores = _avg_neighbor_y_init(ns, lvl)
                    ns_sorted = sorted(ns, key=lambda n: (scores.get(n, 0), n))
                    k = len(ns_sorted)
                    for i, nn in enumerate(ns_sorted):
                        x, y = positions[nn]
                        y_off = (i - (k - 1) / 2) * STAG
                        positions[nn] = (x, y + y_off)
        else:
            positions = {}

    # Add nodes (SVG images) with hidden labels
    for n, data in G.nodes(data=True):
        title = data.get("title", n)
        loc = data.get("location") or (str(n).split("@")[-1] if "@" in str(n) else "")
        eq = data.get("equipment") or (str(n).split("@")[0] if "@" in str(n) else str(n))
        proc = data.get("primary_process_label", "")
        outv = data.get("primary_output", "")
        node_color = data.get("node_color", "#97C2FC")
        try:
            import urllib.parse as _urlparse  # local import to avoid top-level dependency
            def _esc(x: str) -> str:
                s = str(x)
                return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            letter = (proc[:1].upper() if proc else '?')
            loc_e = _esc(loc)
            eq_e = _esc(eq)
            proc_e = _esc(proc)
            out_e = _esc(outv)
            svg = (
                "<svg xmlns='http://www.w3.org/2000/svg' width='260' height='120'>"
                "<style> .t{font-family:Arial; font-size:12px; fill:#111;} </style>"
                f"<text x='130' y='16' class='t' text-anchor='middle'>{loc_e}</text>"
                f"<text x='130' y='32' class='t' text-anchor='middle'>{eq_e}</text>"
                f"<circle cx='130' cy='58' r='16' fill='{node_color}' stroke='#333' stroke-width='1.2'/>"
                f"<text x='130' y='63' text-anchor='middle' font-size='14' font-family='Arial' fill='#fff'>{letter}</text>"
                f"<text x='130' y='92' class='t' text-anchor='middle'>{proc_e}</text>"
                f"<text x='130' y='108' class='t' text-anchor='middle'>{out_e}</text>"
                "</svg>"
            )
            data_url = "data:image/svg+xml;utf8," + _urlparse.quote(svg)
        except Exception:
            data_url = None
        node_kwargs = {}
        if 'positions' in locals() and n in positions:
            x, y = positions[n]
            node_kwargs.update({"x": x, "y": y})
            if use_fixed_grid and not enable_physics:
                node_kwargs.update({"fixed": True, "physics": False})
        if data_url:
            net.add_node(n, label="", title=title, color=node_color, shape="image", image=data_url, **node_kwargs)
        else:
            net.add_node(n, label="", title=title, color=node_color, **node_kwargs)

    # Build Move stub lists if simplify_move is enabled and we have fixed positions to place stubs
    outgoing_moves: Dict[str, list[tuple]] = {}
    incoming_moves: Dict[str, list[tuple]] = {}
    stub_capable = ('positions' in locals()) and isinstance(positions, dict) and len(positions) > 0

    for u, v, data in G.edges(data=True):
        proc = str(data.get("process", "")).strip().lower()
        if simplify_move and proc == "move":
            # Skip drawing the long Move edge; collect for stubs when possible
            if stub_capable:
                dst_loc = G.nodes.get(v, {}).get("location", "")
                dst_eq = G.nodes.get(v, {}).get("equipment", "")
                dst_loc_idx = int(G.nodes.get(v, {}).get("loc_index", 10**9))
                out_lbl = str(data.get("label", "")).strip()
                outgoing_moves.setdefault(u, []).append((dst_loc_idx, dst_loc, dst_eq, out_lbl))

                src_loc = G.nodes.get(u, {}).get("location", "")
                src_eq = G.nodes.get(u, {}).get("equipment", "")
                src_loc_idx = int(G.nodes.get(u, {}).get("loc_index", 10**9))
                incoming_moves.setdefault(v, []).append((src_loc_idx, src_loc, src_eq, out_lbl))
                continue
        # Default: draw edge as-is
        net.add_edge(u, v, label=data.get("label", ""), title=data.get("title", ""), color=data.get("color"))

    # Helper to make a text sprite as an image node
    def _add_text_sprite(node_id: str, text: str, x_center: int, y_center: int, color_override: Optional[str] = None):
        try:
            import urllib.parse as _urlparse  # noqa
            def _esc(s: str) -> str:
                return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            font = int(getattr(config, 'move_label_font_size', 10))
            pad = int(getattr(config, 'move_label_pad', 6))
            color = color_override if (color_override is not None) else getattr(config, 'move_label_text_color', '#333333')
            bg_on = bool(getattr(config, 'move_label_bg', True))
            bg_rgba = getattr(config, 'move_label_bg_rgba', 'rgba(255,255,255,0.85)')
            # rough width estimate: 0.6 * font px per char, plus padding
            txt = str(text)
            w = int(len(txt) * font * 0.6 + pad * 2)
            h = int(font + 6)
            y_text = int(font + 2)
            rect = f"<rect x='0' y='0' rx='2' ry='2' width='{w}' height='{h}' fill='{bg_rgba}' stroke='none'/>" if bg_on else ""
            svg = (
                f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}'>"
                f"<style>.t{{font-family:Arial;font-size:{font}px;fill:{color};}}</style>"
                f"{rect}"
                f"<text x='{pad}' y='{y_text}' class='t'>{_esc(txt)}</text>"
                "</svg>"
            )
            data_url = "data:image/svg+xml;utf8," + _urlparse.quote(svg)
            net.add_node(node_id, label="", image=data_url, shape="image", size=1, physics=False, fixed=True, x=int(x_center), y=int(y_center))
        except Exception:
            # Fallback: simple label node
            net.add_node(node_id, label=str(text), shape="box", physics=False, fixed=True, x=int(x_center), y=int(y_center))

    # Add production rate labels under Make nodes (from Make sheet)
    try:
        _show_prod_rate = bool(getattr(config, 'show_prod_rate', True))
    except Exception:
        _show_prod_rate = True
    if _show_prod_rate and ('positions' in locals()) and isinstance(positions, dict) and len(positions) > 0:
        # Determine which workbook is in use (mirror main())
        in_path = Path(getattr(config, 'in_path', 'Model Inputs.xlsx'))
        gen_path = Path(getattr(config, 'generated_inputs_path', 'generated_model_inputs.xlsx'))
        use_generated = bool(getattr(config, 'use_generated_inputs', True))
        input_used = gen_path if (use_generated and gen_path.exists()) else in_path
        # Read Make sheet
        make_df = _read_sheet_df(input_used, 'Make')
        rate_map: Dict[Tuple[str, str], str] = {}
        if make_df is not None and not make_df.empty:
            # Best-effort column resolution
            cols_lower = {str(c).strip().lower(): c for c in make_df.columns}
            loc_col = cols_lower.get('location', 'Location' if 'Location' in make_df.columns else None)
            eq_col = cols_lower.get('equipment name', 'Equipment Name' if 'Equipment Name' in make_df.columns else None)
            # Prefer exact header; fallback to fuzzy contains
            rate_col = None
            for c in make_df.columns:
                if str(c).strip().lower() == 'mean production rate (tons/hr)':
                    rate_col = c
                    break
            if rate_col is None:
                for c in make_df.columns:
                    lc = str(c).strip().lower()
                    if ('mean' in lc) and ('production' in lc) and ('rate' in lc):
                        rate_col = c
                        break
            if loc_col and eq_col and rate_col:
                dfm = make_df[[loc_col, eq_col, rate_col]].copy()
                dfm[loc_col] = dfm[loc_col].astype(str).fillna('').str.strip()
                dfm[eq_col] = dfm[eq_col].astype(str).fillna('').str.strip()
                # Build first non-empty value per (Location, Equipment)
                for (locv, eqv), grp in dfm.groupby([loc_col, eq_col]):
                    val = None
                    for x in grp[rate_col]:
                        if pd.isna(x):
                            continue
                        s = str(x).strip()
                        if s == '':
                            continue
                        val = s
                        break
                    if val is not None:
                        rate_map[(locv, eqv)] = val
        red = getattr(config, 'prod_rate_text_color', '#d62728')
        # Place label directly under the node's bottom edge by default (tight gap),
        # but honor an explicit prod_rate_y_offset if provided for backward compatibility.
        y_off_cfg = getattr(config, 'prod_rate_y_offset', None)
        y_from_center: int
        if y_off_cfg is not None:
            try:
                y_from_center = int(y_off_cfg)
            except Exception:
                # Fallback to dynamic placement if the provided value is invalid
                NODE_IMG_H = 120  # must match the SVG node height defined above
                gap = int(getattr(config, 'prod_rate_gap', 2))  # small gap in pixels
                font = int(getattr(config, 'move_label_font_size', 10))
                h = int(font + 6)
                y_from_center = (NODE_IMG_H // 2) + gap + (h // 2)
        else:
            NODE_IMG_H = 120  # must match the SVG node height defined above
            gap = int(getattr(config, 'prod_rate_gap', 2))  # small gap in pixels
            font = int(getattr(config, 'move_label_font_size', 10))
            h = int(font + 6)
            y_from_center = (NODE_IMG_H // 2) + gap + (h // 2)
        for n, d in G.nodes(data=True):
            if str(d.get('primary_process_key', '')).lower() != 'make':
                continue
            if n not in positions:
                continue
            locv = str(d.get('location', '')).strip()
            eqv = str(d.get('equipment', '')).strip()
            rate_val = rate_map.get((locv, eqv))
            if rate_val is None or str(rate_val).strip() == '':
                txt = 'Prod Rate = ?? Ton/Hr'
            else:
                # Pretty print numeric if possible
                try:
                    num = float(str(rate_val).replace(',', ''))
                    if abs(num - round(num)) < 1e-6:
                        rate_str = f"{int(round(num))}"
                    else:
                        rate_str = f"{num:.2f}".rstrip('0').rstrip('.')
                except Exception:
                    rate_str = str(rate_val)
                txt = f"Prod Rate = {rate_str} Ton/Hr"
            x0, y0 = positions[n]
            y_lbl = int(y0 + y_from_center)
            sprite_id = f"{n}::prod_rate"
            _add_text_sprite(sprite_id, txt, int(x0), y_lbl, color_override=red)

    # Add storage capacity labels under Store nodes (from Store sheet)
    try:
        _show_store_cap = bool(getattr(config, 'show_store_capacity', True))
    except Exception:
        _show_store_cap = True
    if _show_store_cap and ('positions' in locals()) and isinstance(positions, dict) and len(positions) > 0:
        # Determine which workbook is in use (mirror main())
        in_path = Path(getattr(config, 'in_path', 'Model Inputs.xlsx'))
        gen_path = Path(getattr(config, 'generated_inputs_path', 'generated_model_inputs.xlsx'))
        use_generated = bool(getattr(config, 'use_generated_inputs', True))
        input_used = gen_path if (use_generated and gen_path.exists()) else in_path
        # Read Store sheet
        store_df = _read_sheet_df(input_used, 'Store')
        cap_map: Dict[Tuple[str, str, str], str] = {}
        if store_df is not None and not store_df.empty:
            # Best-effort column resolution
            cols_lower = {str(c).strip().lower(): c for c in store_df.columns}
            loc_col = cols_lower.get('location', 'Location' if 'Location' in store_df.columns else None)
            eq_col = cols_lower.get('equipment name', 'Equipment Name' if 'Equipment Name' in store_df.columns else None)
            inp_col = cols_lower.get('input', 'Input' if 'Input' in store_df.columns else None)
            # Prefer exact capacity header; fallback to fuzzy contains
            cap_col = None
            for c in store_df.columns:
                if str(c).strip().lower() == 'silo max capacity':
                    cap_col = c
                    break
            if cap_col is None:
                for c in store_df.columns:
                    lc = str(c).strip().lower()
                    if 'capacity' in lc:
                        cap_col = c
                        break
            if loc_col and eq_col and inp_col and cap_col:
                dfs = store_df[[loc_col, eq_col, inp_col, cap_col]].copy()
                dfs[loc_col] = dfs[loc_col].astype(str).fillna('').str.strip()
                dfs[eq_col] = dfs[eq_col].astype(str).fillna('').str.strip()
                dfs[inp_col] = dfs[inp_col].astype(str).fillna('').str.strip()
                # Build first non-empty capacity value per (Location, Equipment, Input)
                for (locv, eqv, inv), grp in dfs.groupby([loc_col, eq_col, inp_col]):
                    val = None
                    for x in grp[cap_col]:
                        if pd.isna(x):
                            continue
                        s = str(x).strip()
                        if s == '':
                            continue
                        val = s
                        break
                    if val is not None:
                        cap_map[(str(locv).upper(), str(eqv).upper(), str(inv).upper())] = val
        cap_color = getattr(config, 'store_capacity_text_color', '#d62728')
        # Place label under node similar to Make labels; allow override via store_capacity_y_offset
        y_off_store = getattr(config, 'store_capacity_y_offset', None)
        if y_off_store is not None:
            try:
                y_from_center_store = int(y_off_store)
            except Exception:
                NODE_IMG_H = 120  # must match the SVG node height defined above
                gap = int(getattr(config, 'prod_rate_gap', 2))  # small gap in pixels
                font = int(getattr(config, 'move_label_font_size', 10))
                h = int(font + 6)
                y_from_center_store = (NODE_IMG_H // 2) + gap + (h // 2)
        else:
            NODE_IMG_H = 120  # must match the SVG node height defined above
            gap = int(getattr(config, 'prod_rate_gap', 2))  # small gap in pixels
            font = int(getattr(config, 'move_label_font_size', 10))
            h = int(font + 6)
            y_from_center_store = (NODE_IMG_H // 2) + gap + (h // 2)
        for n, d in G.nodes(data=True):
            if str(d.get('primary_process_key', '')).lower() != 'store':
                continue
            if n not in positions:
                continue
            locv = str(d.get('location', '')).strip().upper()
            eqv = str(d.get('equipment', '')).strip().upper()
            # Infer Input from OUTGOING edges of this Store node where process is 'Store'
            inputs_found: list[str] = []
            for _, dst, edata in G.out_edges(n, data=True):
                if str(edata.get('process', '')).strip().lower() != 'store':
                    continue
                t = str(edata.get('title', ''))
                inp_val = ''
                try:
                    idx = t.find('Input:')
                    if idx >= 0:
                        s = t[idx + len('Input:'):]
                        s = s.replace('\r', '').replace('\n', '')
                        end_br = s.find('<')
                        if end_br >= 0:
                            s = s[:end_br]
                        inp_val = s.strip()
                except Exception:
                    inp_val = ''
                if inp_val:
                    inputs_found.append(inp_val.upper())
            inp_key = None
            if inputs_found:
                try:
                    from collections import Counter
                    inp_key = Counter(inputs_found).most_common(1)[0][0]
                except Exception:
                    inp_key = inputs_found[0]
            # Fallback: try node's primary_output if not found
            if not inp_key:
                po = str(d.get('primary_output', '')).strip()
                if po:
                    inp_key = po.upper()
            cap_val = cap_map.get((locv, eqv, inp_key)) if inp_key else None
            if cap_val is None or str(cap_val).strip() == '':
                txt = 'Capacity = ?? kt'
            else:
                try:
                    num = float(str(cap_val).replace(',', ''))  # capacity in tons
                    kt = num / 1000.0
                    kt_str = f"{kt:,.2f}".rstrip('0').rstrip('.')
                    txt = f"Capacity = {kt_str} kt"
                except Exception:
                    txt = 'Capacity = ?? kt'
            x0, y0 = positions[n]
            y_lbl_store = int(y0 + y_from_center_store)
            sprite_id_store = f"{n}::store_capacity"
            _add_text_sprite(sprite_id_store, txt, int(x0), y_lbl_store, color_override=cap_color)

    # Add annual demand labels to the RIGHT of Deliver nodes (from Deliver sheet)
    try:
        _show_demand = bool(getattr(config, 'show_demand', True))
    except Exception:
        _show_demand = True
    if _show_demand and ('positions' in locals()) and isinstance(positions, dict) and len(positions) > 0:
        # Determine which workbook is in use (mirror main())
        in_path = Path(getattr(config, 'in_path', 'Model Inputs.xlsx'))
        gen_path = Path(getattr(config, 'generated_inputs_path', 'generated_model_inputs.xlsx'))
        use_generated = bool(getattr(config, 'use_generated_inputs', True))
        input_used = gen_path if (use_generated and gen_path.exists()) else in_path
        # Read Deliver sheet (fallback to 'Delivery' if needed)
        deliver_df = _read_sheet_df(input_used, 'Deliver')
        if deliver_df is None or deliver_df.empty:
            deliver_df = _read_sheet_df(input_used, 'Delivery')
        demand_map: Dict[Tuple[str, str], str] = {}
        if deliver_df is not None and not deliver_df.empty:
            # Best-effort column resolution
            cols_lower = {str(c).strip().lower(): c for c in deliver_df.columns}
            loc_col = cols_lower.get('location', 'Location' if 'Location' in deliver_df.columns else None)
            inp_col = cols_lower.get('input', 'Input' if 'Input' in deliver_df.columns else None)
            # Prefer exact header; fallback to fuzzy contains
            dem_col = None
            for c in deliver_df.columns:
                if str(c).strip().lower() == 'demand per location':
                    dem_col = c
                    break
            if dem_col is None:
                for c in deliver_df.columns:
                    lc = str(c).strip().lower()
                    if ('demand' in lc) and ('location' in lc):
                        dem_col = c
                        break
            if loc_col and inp_col and dem_col:
                dfd = deliver_df[[loc_col, inp_col, dem_col]].copy()
                dfd[loc_col] = dfd[loc_col].astype(str).fillna('').str.strip()
                dfd[inp_col] = dfd[inp_col].astype(str).fillna('').str.strip()
                # Build first non-empty value per (Location, Input)
                for (locv, inv), grp in dfd.groupby([loc_col, inp_col]):
                    val = None
                    for x in grp[dem_col]:
                        if pd.isna(x):
                            continue
                        s = str(x).strip()
                        if s == '':
                            continue
                        val = s
                        break
                    if val is not None:
                        demand_map[(str(locv).upper(), str(inv).upper())] = val
        red_demand = getattr(config, 'demand_text_color', '#d62728')

        # Geometry for right-side placement
        NODE_IMG_W = 120  # keep in sync with node image size
        GAP_X = int(getattr(config, 'demand_gap', 6))  # horizontal gap to node edge
        # Optional fine-tuning: allow vertical nudge; default is vertically centered
        y_off_cfg2 = getattr(config, 'demand_y_offset', 0)
        try:
            Y_NUDGE = int(y_off_cfg2) if y_off_cfg2 is not None else 0
        except Exception:
            Y_NUDGE = 0
        # Optional explicit x offset from node center (overrides auto placement if provided)
        x_off_cfg = getattr(config, 'demand_x_offset', None)

        # Text width estimate needs to mirror _add_text_sprite's metrics
        FONT = int(getattr(config, 'move_label_font_size', 10))
        PAD = int(getattr(config, 'move_label_pad', 6))
        def _estimate_w(txt: str) -> int:
            return int(len(txt) * FONT * 0.6 + PAD * 2)

        # Number formatting helpers: convert tons/year -> kilotons/year and format with thousands separator, trim decimals
        HOURS_PER_YEAR = 364 * 24
        def _format_kt(tons_per_year: float) -> str:
            try:
                kt = float(tons_per_year) / 1000.0
            except Exception:
                return '??'
            # Two decimals with thousands separator, then trim trailing zeros and decimal point
            s = f"{kt:,.2f}".rstrip('0').rstrip('.')
            return s

        for n, d in G.nodes(data=True):
            if str(d.get('primary_process_key', '')).lower() != 'deliver':
                continue
            if n not in positions:
                continue
            locv = str(d.get('location', '')).strip()
            # Infer Input from incoming Deliver edges
            inputs_found: list[str] = []
            for src, _, edata in G.in_edges(n, data=True):
                t = str(edata.get('title', ''))
                inp_val = ''
                try:
                    # Prefer 'Input:' marker; fallback to 'Output:' if not found
                    for marker in ('Input:', 'Output:'):
                        idx = t.find(marker)
                        if idx >= 0:
                            s = t[idx + len(marker):]
                            s = s.replace('\r', '').replace('\n', '')
                            end_br = s.find('<')
                            if end_br >= 0:
                                s = s[:end_br]
                            inp_val = s.strip()
                            if inp_val:
                                break
                except Exception:
                    inp_val = ''
                if inp_val:
                    inputs_found.append(inp_val)
            # choose most common input if multiple
            inp = None
            if inputs_found:
                try:
                    from collections import Counter
                    inp = Counter([v.upper() for v in inputs_found]).most_common(1)[0][0]
                except Exception:
                    inp = inputs_found[0].upper()
            # Lookup demand (Demand per Location in tons/hour)
            demand_val = None
            if inp:
                demand_val = demand_map.get((locv.upper(), inp))
            # Build label text and position to the RIGHT
            if demand_val is None or str(demand_val).strip() == '':
                core_txt = 'Demand = ?? kt'
            else:
                try:
                    num = float(str(demand_val).replace(',', ''))
                    annual_ton = num * HOURS_PER_YEAR
                    core_txt = f"Demand = {_format_kt(annual_ton)} kt"
                except Exception:
                    core_txt = 'Demand = ?? kt'
            x0, y0 = positions[n]
            # Compute sprite center to the right of the node
            w_est = _estimate_w(core_txt)
            if x_off_cfg is not None:
                try:
                    x_from_center = int(x_off_cfg)
                except Exception:
                    x_from_center = (NODE_IMG_W // 2) + GAP_X + (w_est // 2)
            else:
                x_from_center = (NODE_IMG_W // 2) + GAP_X + (w_est // 2)
            x_lbl = int(x0 + x_from_center)
            y_lbl = int(y0 + Y_NUDGE)
            sprite_id2 = f"{n}::annual_demand"
            _add_text_sprite(sprite_id2, core_txt, x_lbl, y_lbl, color_override=red_demand)

    # Add pitchforks with label sprites so text starts at arrow tips
    if simplify_move and stub_capable:
        JUNC_DX = int(getattr(config, 'fork_junction_dx', 45))
        LEAF_DX = int(getattr(config, 'fork_leaf_dx', 120))
        DY = int(getattr(config, 'fork_prong_dy', 18))
        MOVE_COLOR = getattr(config, 'move_color', '#ff7f0e')
        MAX_PER_SIDE = getattr(config, 'move_label_max_per_side', None)
        LABEL_PAD = int(getattr(config, 'move_label_pad', 6))
        FONT = int(getattr(config, 'move_label_font_size', 10))

        def _width_estimate(txt: str) -> int:
            return int(len(txt) * FONT * 0.6 + LABEL_PAD * 2)

        # Outgoing (right side)
        for n, entries in outgoing_moves.items():
            if n not in positions:
                continue
            x0, y0 = positions[n]
            entries_sorted = sorted(entries, key=lambda t: (t[0], t[1], t[2], t[3]))
            if isinstance(MAX_PER_SIDE, int):
                entries_sorted = entries_sorted[:MAX_PER_SIDE]
            k = len(entries_sorted)
            if k == 0:
                continue
            jx = x0 + JUNC_DX
            jy = y0
            junc_id = f"{n}::junc_out"
            net.add_node(junc_id, label="", shape="dot", size=3, physics=False, fixed=True, x=jx, y=jy)
            net.add_edge(n, junc_id, color=MOVE_COLOR, width=1)
            y_start = y0 - (k - 1) * DY / 2
            for i, (_idx, to_loc, to_eq, out_lbl) in enumerate(entries_sorted):
                ly = int(y_start + i * DY)
                # Arrow tip (anchor) position
                ax = x0 + LEAF_DX
                anchor_id = f"{n}::anchor_out::{i}"
                net.add_node(anchor_id, label="", shape="dot", size=1, physics=False, fixed=True, x=ax, y=ly)
                # Prong from junction to anchor (arrow points right)
                net.add_edge(junc_id, anchor_id, color=MOVE_COLOR, width=1)
                label_text = f"To {to_loc} {to_eq} Move {out_lbl}".strip()
                w = _width_estimate(label_text)
                # Place label sprite so its LEFT edge starts at arrow tip + pad
                label_center_x = ax + LABEL_PAD + w // 2
                label_id = f"{n}::label_out::{i}"
                _add_text_sprite(label_id, label_text, label_center_x, ly)

        # Incoming (left side)
        for n, entries in incoming_moves.items():
            if n not in positions:
                continue
            x0, y0 = positions[n]
            entries_sorted = sorted(entries, key=lambda t: (t[0], t[1], t[2], t[3]))
            if isinstance(MAX_PER_SIDE, int):
                entries_sorted = entries_sorted[:MAX_PER_SIDE]
            k = len(entries_sorted)
            if k == 0:
                continue
            jx = x0 - JUNC_DX
            jy = y0
            junc_id = f"{n}::junc_in"
            net.add_node(junc_id, label="", shape="dot", size=3, physics=False, fixed=True, x=jx, y=jy)
            net.add_edge(junc_id, n, color=MOVE_COLOR, width=1)
            y_start = y0 - (k - 1) * DY / 2
            for i, (_idx, from_loc, from_eq, out_lbl) in enumerate(entries_sorted):
                ly = int(y_start + i * DY)
                # Arrow start (anchor) position to the left
                ax = x0 - LEAF_DX
                anchor_id = f"{n}::anchor_in::{i}"
                net.add_node(anchor_id, label="", shape="dot", size=1, physics=False, fixed=True, x=ax, y=ly)
                # Prong from anchor to junction (arrow points right to node)
                net.add_edge(anchor_id, junc_id, color=MOVE_COLOR, width=1)
                label_text = f"From {from_loc} {from_eq} Move {out_lbl}".strip()
                w = _width_estimate(label_text)
                # Place label sprite so its RIGHT edge ends at arrow tip - pad (text extends leftwards)
                label_center_x = ax - LABEL_PAD - w // 2
                label_id = f"{n}::label_in::{i}"
                _add_text_sprite(label_id, label_text, label_center_x, ly)

    # Legend will be added as an HTML block outside the canvas (see injection below)

    # Write HTML
    net.write_html(str(out_html), open_browser=False, notebook=False)

    # Inject legend (outside canvas) and auto-fit on load (keep physics ON for interactivity)
    try:
        html_text = out_html.read_text(encoding="utf-8")

        # 1) Legend block before the network div
        legend_base = (
            "\n<div id=\"legend\" style=\"margin:8px 0 12px 0;font-family:Arial, sans-serif;font-size:12px;\">"
            "<span style=\"display:inline-block;width:12px;height:12px;background:#2ca02c;border-radius:50%;margin-right:6px;vertical-align:middle;\"></span>Make"
            " &nbsp; <span style=\"display:inline-block;width:12px;height:12px;background:#1f77b4;border-radius:50%;margin-right:6px;vertical-align:middle;\"></span>Store"
            " &nbsp; <span style=\"display:inline-block;width:12px;height:12px;background:#ff7f0e;border-radius:50%;margin-right:6px;vertical-align:middle;\"></span>Move"
            " &nbsp; <span style=\"display:inline-block;width:12px;height:12px;background:#9467bd;border-radius:50%;margin-right:6px;vertical-align:middle;\"></span>Deliver"
        )
        if simplify_move and bool(getattr(config, 'show_legend', True)):
            legend_block = legend_base + " &nbsp; <span style=\"margin-left:10px;color:#555;\">Simplified Move: orange pitchforks list To/From. Labels start at arrow tips.</span></div>\n"
        else:
            legend_block = legend_base + "</div>\n"
        html_text = html_text.replace('<div id="mynetwork"', legend_block + '<div id="mynetwork"')

        # 2) JS to auto-fit only; leave physics as configured
        # Also draw swimlane wireframes (rectangles) across all product-class panels for each Location row when enabled.
        try:
            _draw_wire = bool(getattr(config, 'swimlane_wireframes', True)) and ('positions' in locals()) and isinstance(positions, dict) and len(positions) > 0 and use_fixed_grid and (not enable_physics)
        except Exception:
            _draw_wire = False
        rects_js = "[]"
        style_js = "{}"
        labels_js = "[]"
        label_style_js = "{}"
        pc_labels_js = "[]"
        pc_label_style_js = "{}"
        if _draw_wire:
            import json as _json
            # Overall horizontal min/max among all nodes (canvas coords)
            try:
                all_xs = [int(xy[0]) for xy in positions.values()]
                x_min_all = min(all_xs)
                x_max_all = max(all_xs)
            except Exception:
                x_min_all = 0
                x_max_all = len(pc_to_col) * PCSEP + 3 * XSEP
            # Y band spacing
            YSEP_val = int(getattr(config, 'grid_y_sep', YSEP))
            margin_x = int(getattr(config, 'swimlane_wireframe_margin_x', 60))
            margin_y = int(getattr(config, 'swimlane_wireframe_margin_y', 40))
            # Collect unique loc_index rows present
            loc_indices = sorted({int(d.get('loc_index', 0)) for _, d in G.nodes(data=True)})
            # Build preferred label text for each loc_index (most common location name in that row)
            from collections import Counter as _Counter  # local import to avoid global dependency
            _names_by_li = {}
            for _n, _d in G.nodes(data=True):
                try:
                    _li = int(_d.get('loc_index', 0))
                except Exception:
                    _li = 0
                _locname = str(_d.get('location', '')).strip()
                if _locname:
                    _names_by_li.setdefault(_li, []).append(_locname)
            rects = []
            labels = []
            width = int((x_max_all - x_min_all) + 2 * margin_x)
            # Label config
            label_on = bool(getattr(config, 'swimlane_labels', True))
            lab_ml = int(getattr(config, 'swimlane_label_margin_left', 10))
            lab_mt = int(getattr(config, 'swimlane_label_margin_top', 6))
            for li in loc_indices:
                y_center = int(li * YSEP)
                x_left = int(x_min_all - margin_x)
                y_top = int(y_center - (YSEP_val // 2) + margin_y)
                height = int(YSEP_val - 2 * margin_y)
                rects.append({'x': x_left, 'y': y_top, 'w': width, 'h': height})
                if label_on:
                    names = _names_by_li.get(li, [])
                    label_text = _Counter(names).most_common(1)[0][0] if names else f"Lane {li}"
                    labels.append({'text': label_text, 'x': int(x_left + lab_ml), 'y': int(y_top + lab_mt)})
            rects_js = _json.dumps(rects)
            labels_js = _json.dumps(labels)
            style_js = _json.dumps({
                'color': getattr(config, 'swimlane_wireframe_color', 'rgba(0,0,0,0.25)'),
                'stroke': int(getattr(config, 'swimlane_wireframe_stroke', 2)),
                'radius': int(getattr(config, 'swimlane_wireframe_corner_radius', 10)),
            })
            label_style_js = _json.dumps({
                'enabled': bool(getattr(config, 'swimlane_labels', True)),
                'color': getattr(config, 'swimlane_label_text_color', '#222222'),
                'fontPx': int(getattr(config, 'swimlane_label_font_px', 14)),
                'fontFamily': getattr(config, 'swimlane_label_font_family', 'Arial, sans-serif'),
            })
            # Build per-product-class bottom-centered labels for each swimlane
            pc_labels = []
            try:
                pc_label_on = bool(getattr(config, 'swimlane_pc_labels', True))
            except Exception:
                pc_label_on = True
            if pc_label_on:
                # per-PC centers across all nodes (panel centers)
                pc_centers: Dict[str, int] = {}
                for pc in pc_order:
                    xs = [int(positions[n][0]) for n, d in G.nodes(data=True) if str(d.get('product_class', '')).strip() == pc and n in positions]
                    if xs:
                        pc_centers[pc] = int((min(xs) + max(xs)) / 2)
                bottom_margin = int(getattr(config, 'swimlane_pc_label_margin_bottom', 10))
                for li in loc_indices:
                    # derive y from the rect we just computed
                    y_center = int(li * YSEP)
                    y_top = int(y_center - (YSEP_val // 2) + margin_y)
                    height = int(YSEP_val - 2 * margin_y)
                    y_baseline = int(y_top + height - bottom_margin)
                    for pc in pc_order:
                        cx = pc_centers.get(pc)
                        if cx is None:
                            continue
                        pc_labels.append({'text': pc, 'x': int(cx), 'y': y_baseline})
            pc_labels_js = _json.dumps(pc_labels)
            pc_label_style_js = _json.dumps({
                'enabled': bool(getattr(config, 'swimlane_pc_labels', True)),
                'color': getattr(config, 'swimlane_pc_label_text_color', '#222222'),
                'fontPx': int(getattr(config, 'swimlane_pc_label_font_px', 16)),
                'fontFamily': getattr(config, 'swimlane_pc_label_font_family', 'Arial, sans-serif'),
            })
        inject_js = (
            "\n<script type=\"text/javascript\">\n"
            "window.addEventListener('load', function(){\n"
            "  try { if (typeof network !== 'undefined') { network.fit({animation:{duration:500, easing:'easeInOutQuad'}}); } } catch(e) { console && console.warn && console.warn('auto-fit failed', e); }\n"
            "  // Force-hide the pyvis loading overlay, which can linger when physics is disabled\n"
            "  try {\n"
            "    var lb = document.getElementById('loadingBar'); if (lb) { lb.style.display = 'none'; }\n"
            "    var txt = document.getElementById('text'); if (txt && txt.parentNode && txt.parentNode.parentNode && txt.parentNode.parentNode.parentNode && txt.parentNode.parentNode.parentNode.id === 'loadingBar') { txt.style.display = 'none'; }\n"
            "    var bar = document.getElementById('bar'); if (bar && bar.parentNode && bar.parentNode.parentNode && bar.parentNode.parentNode.parentNode && bar.parentNode.parentNode.parentNode.id === 'loadingBar') { bar.style.display = 'none'; }\n"
            "  } catch(e) { }\n"
            "  try { if (typeof network !== 'undefined') { network.once('afterDrawing', function(){ var lb2 = document.getElementById('loadingBar'); if(lb2){ lb2.style.display='none'; } }); } } catch(e) { }\n"
            + ("  try { if (typeof network !== 'undefined') {\n"
               "    var rects = " + rects_js + ";\n"
               "    var style = " + style_js + ";\n"
               "    var labels = " + labels_js + ";\n"
               "    var labelStyle = " + label_style_js + ";\n"
               "    var pcLabels = " + pc_labels_js + ";\n"
               "    var pcLabelStyle = " + pc_label_style_js + ";\n"
               "    function drawRoundRect(ctx, x, y, w, h, r){\n"
               "      if (!ctx) return; r = Math.max(0, r||0); ctx.beginPath(); ctx.moveTo(x+r, y); ctx.lineTo(x+w-r, y); ctx.arcTo(x+w, y, x+w, y+r, r); ctx.lineTo(x+w, y+h-r); ctx.arcTo(x+w, y+h, x+w-r, y+h, r); ctx.lineTo(x+r, y+h); ctx.arcTo(x, y+h, x, y+h-r, r); ctx.lineTo(x, y+r); ctx.arcTo(x, y, x+r, y, r); ctx.closePath(); }\n"
               "    network.on('afterDrawing', function(ctx){ try { if (!rects || rects.length===0) return; ctx.save(); ctx.strokeStyle = style.color || 'rgba(0,0,0,0.25)'; ctx.lineWidth = style.stroke || 2; for (var i=0;i<rects.length;i++){ var r = rects[i]; drawRoundRect(ctx, r.x, r.y, r.w, r.h, style.radius||8); ctx.stroke(); } if (labelStyle && labelStyle.enabled && labels && labels.length){ ctx.fillStyle = labelStyle.color || '#222'; ctx.font = String(labelStyle.fontPx||14) + 'px ' + (labelStyle.fontFamily || 'Arial, sans-serif'); ctx.textBaseline = 'top'; ctx.textAlign = 'left'; for (var j=0;j<labels.length;j++){ var L = labels[j]; try { ctx.fillText(String(L.text||''), L.x, L.y); } catch(e2){} } } if (pcLabelStyle && pcLabelStyle.enabled && pcLabels && pcLabels.length){ ctx.fillStyle = pcLabelStyle.color || '#222'; ctx.font = String(pcLabelStyle.fontPx||16) + 'px ' + (pcLabelStyle.fontFamily || 'Arial, sans-serif'); ctx.textBaseline = 'alphabetic'; ctx.textAlign = 'center'; for (var k=0;k<pcLabels.length;k++){ var P = pcLabels[k]; try { ctx.fillText(String(P.text||''), P.x, P.y); } catch(e3){} } } ctx.restore(); } catch(e) { console && console.warn && console.warn('wireframe draw failed', e); } });\n"
               "  } } catch(e) { }\n")
            +
            "});\n"
            "</script>\n"
        )
        if "</body>" in html_text:
            html_text = html_text.replace("</body>", inject_js + "</body>")
        else:
            html_text = html_text + inject_js
        out_html.write_text(html_text, encoding="utf-8")
    except Exception:
        # Best effort; if injection fails we still have the HTML
        pass



# ------------------------ SimPy scaffold ------------------------------------

def build_simpy_model_from_dataframe(df: pd.DataFrame, product_class: Optional[str] = None):
    """Return a callable that will build a SimPy environment from the dataframe.

    Notes:
    - This is a scaffold to be extended. It creates Resource objects for each
      Equipment@Location and stores a basic routing table derived from the edges.
    - Actual process durations, capacities, batch sizes, etc., should be provided
      via additional columns in the Excel and mapped here later.
    """
    if simpy is None:
        raise RuntimeError("simpy isn't installed. Install with 'pip install simpy'.")

    if product_class:
        df = df[df["product_class"].str.upper() == product_class.upper()].copy()
        if df.empty:
            raise ValueError(f"No rows found for product class '{product_class}'.")

    # Build basic routing table
    routes: Dict[str, list[Tuple[str, Dict[str, str]]]] = {}
    res_caps: Dict[str, int] = {}

    for _, r in df.iterrows():
        pc_row = str(r.get("product_class", "")).strip()
        src = node_id(r["location"], r["equipment_name"], pc_row, str(r.get("input", "")).strip()) if r["equipment_name"] else None
        if not src:
            continue
        res_caps.setdefault(src, 1)  # default capacity 1; to be overridden from Excel later
        next_loc = str(r["next_location"]).strip()
        next_eq = str(r["next_equipment"]).strip()
        if next_loc and next_eq:
            dst = node_id(next_loc, next_eq, pc_row, str(r.get("output", "")).strip())
            routes.setdefault(src, []).append(
                (dst, {"process": r["process"], "input": r["input"], "output": r["output"]})
            )

    def build_env():
        env = simpy.Environment()
        # Create resources
        resources: Dict[str, simpy.Resource] = {name: simpy.Resource(env, capacity=cap) for name, cap in res_caps.items()}

        # Example entity generator and processor (placeholder)
        def entity(name: str, start_node: str):
            yield env.timeout(0)
            current = start_node
            while True:
                res = resources[current]
                with res.request() as req:
                    yield req
                    # Placeholder processing time; replace with Excel-driven params
                    yield env.timeout(1)
                # Routing: pick first available edge deterministically (to be improved)
                nexts = routes.get(current, [])
                if not nexts:
                    break  # terminal
                current = nexts[0][0]

        # Start a single demo entity at the first node (lowest level)
        start_row = df.sort_values("level").iloc[0]
        pc_row = str(start_row.get("product_class", "")).strip()
        start_node = node_id(start_row["location"], start_row["equipment_name"], pc_row, str(start_row.get("input", "")).strip()) if start_row["equipment_name"] else list(resources.keys())[0]
        env.process(entity("demo", start_node))
        return env, resources, routes

    return build_env


# ------------------------ Input preparation (Excel) -------------------------
MAKE_REQUIRED_INPUT_COLS = [
    "Mean Production Rate (Tons/hr)",
    "Std Dev of Production Rate (Tons/Hr)",
    "Planned Maintenance Dates (Days of year)",
    "Unplanned downtime %",
    "Consumption %",
]
MAKE_KEY_COLS = ["Location", "Equipment Name", "Output"]
MAKE_SHEET_COLS = ["Location", "Equipment Name", "Input", "Output"] + MAKE_REQUIRED_INPUT_COLS

STORE_REQUIRED_INPUT_COLS = [
    "Silo Max Capacity",
    "Silo Opening Stock (High)",
    "Silo Opening Stock (Low)",
    "Load Rate (ton/hr)",
    "Unload Rate (ton/hr)",
]
STORE_KEY_COLS = ["Location", "Equipment Name", "Input"]
STORE_SHEET_COLS = STORE_KEY_COLS + STORE_REQUIRED_INPUT_COLS

# Deliver sheet columns: now include an optional 'Demand Store' selector after 'Input'
DELIVERY_REQUIRED_INPUT_COLS = [
]
DELIVERY_OPTIONAL_INPUT_COLS = [
    "Demand Store",
]
DELIVERY_KEY_COLS = ["Location", "Input"]
# Column set used when creating new sheet (order will be adjusted to place 'Demand Store' after 'Input')
DELIVERY_SHEET_COLS = DELIVERY_KEY_COLS + DELIVERY_OPTIONAL_INPUT_COLS + DELIVERY_REQUIRED_INPUT_COLS

# MOVE_TRAIN sheet (replaces legacy 'Move')
MOVE_TRAIN_REQUIRED_INPUT_COLS = [
    "Distance",
    "# Trains",
    "# Carraiges",
    "# Carraige Capacity (ton)",
    "Avg Speed - Loaded (km/hr)",
    "Avg Speed - Empty (km/hr)",
]
MOVE_TRAIN_KEY_COLS = ["Product Class", "Product", "Origin Location", "Destination Location"]
MOVE_TRAIN_SHEET_COLS = MOVE_TRAIN_KEY_COLS + [
    "Origin Store",
    "Destination Store",
    "Load Rate (tons/hr)",
    "Unload Rate (tons/hr)",
] + MOVE_TRAIN_REQUIRED_INPUT_COLS

# MOVE_SHIP sheet per spec
# Updated to include two additional columns: "#Hulls" and "Payload per Hull"
# placed immediately after "Route avg Speed (knots)" for clarity.
MOVE_SHIP_SHEET_COLS = [
    "Origin Location",
    "Route Group",
    "# Vessels",
    "Route avg Speed (knots)",
    "#Hulls",
    "Payload per Hull",
]

# BERTHS sheet
BERTHS_REQUIRED_INPUT_COLS = [
    "# Berths",
    "Probability Berth Occupied (%)",
]
BERTHS_KEY_COLS = ["Berth"]
BERTHS_SHEET_COLS = BERTHS_KEY_COLS + BERTHS_REQUIRED_INPUT_COLS

# New ship-related sheets
SHIP_BERTHS_SHEET_COLS = [
    "Location",
    "# Berths",
    "Probability Berth Occupied %",
    "Pilot In (Hours)",
    "Pilot Out (Hours)",
]

SHIP_ROUTES_SHEET_COLS = [
    "Route Group",
    "Route ID",
    "Origin Location",
    "Product 1 Load",
    "Product 1 Store",
    "Product 2 Load",
    "Product 2 Store",
    "Destination 1 Location",
    "Product 1 Unload",
    "Product 1 Unload Store",
    "Product 2 Unload",
    "Product 2 Unload Store",
    "Destination 2 Location",
    "Product 3 Load ",
    "Product 3 Store",
    "Destination 3 Location",
    "Destination 3 Unload",
    "Destination 3 Store",
    "Return Location",
]

SHIP_DISTANCES_SHEET_COLS = [
    "Location 1",
    "Location 2",
    "Distance (nM)",
]


def _normalize_key_triplet(df_like: pd.DataFrame, loc_col: str, eq_col: str, in_col: str) -> pd.Series:
    def _norm(s: pd.Series) -> pd.Series:
        return s.astype(str).fillna("").str.strip().str.upper()
    return _norm(df_like[loc_col]) + "|" + _norm(df_like[eq_col]) + "|" + _norm(df_like[in_col])


def _normalize_key_pair(df_like: pd.DataFrame, loc_col: str, in_col: str) -> pd.Series:
    def _norm(s: pd.Series) -> pd.Series:
        return s.astype(str).fillna("").str.strip().str.upper()
    return _norm(df_like[loc_col]) + "|" + _norm(df_like[in_col])


def _normalize_key_quad(df_like: pd.DataFrame, pc_col: str, loc_col: str, eq_col: str, next_loc_col: str) -> pd.Series:
    def _norm(s: pd.Series) -> pd.Series:
        return s.astype(str).fillna("").str.strip().str.upper()
    return _norm(df_like[pc_col]) + "|" + _norm(df_like[loc_col]) + "|" + _norm(df_like[eq_col]) + "|" + _norm(df_like[next_loc_col])


def _ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            # Create missing columns explicitly as object dtype to avoid pandas
            # inferring float64 (from all-NA) which later rejects string writes.
            df[c] = pd.Series(pd.NA, dtype="object")
    return df


def _read_network_df(xlsx_path: Path) -> pd.DataFrame:
    df = read_table(xlsx_path, sheet="Network")
    # Enforce types for keys
    for c in ["location", "equipment_name", "input", "process"]:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("").str.strip()
    return df


def _read_sheet_df(xlsx_path: Path, sheet_name: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_excel(xlsx_path, sheet_name=sheet_name)
    except Exception:
        return None


def _write_sheet_df(xlsx_path: Path, sheet_name: str, df: pd.DataFrame) -> None:
    if openpyxl is None:
        raise RuntimeError("openpyxl is required to write Excel files. Install with 'pip install openpyxl'.")
    # Use replace to keep other sheets
    with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)


# --- Action logging for generated workbook ---
ACTION_LOG: list[dict] = []

def _log_action(action: str, details: str = "") -> None:
    """Append an action entry with timestamp to the in-memory action log.
    The log is later written to the generated workbook as a 'Log' sheet.
    Also echo to console for immediate visibility.
    """
    try:
        ts = pd.Timestamp.now(tz=None)
    except Exception:
        ts = pd.Timestamp.utcnow()
    ACTION_LOG.append({"Timestamp": ts, "Action": action, "Details": details})
    try:
        print(f"[prepare_inputs] {ts}: {action} — {details}")
    except Exception:
        # Console printing is best-effort
        pass


def ensure_settings_sheet(xlsx_path: Path) -> dict:
    """Create or update the 'Settings' sheet with required settings.
    Returns a summary dict with counts of added/updated keys.
    """
    required = {
        "Number of Simulation Runs": 500,
        "Modeling Horizon (#Days)": 365,
        "Time Buckets (Days, Half Days, Hours)": "Hours",
    }
    # Try to read existing settings
    current = _read_sheet_df(xlsx_path, "Settings")
    added = 0
    updated = 0
    if current is None:
        # Fresh sheet
        out = pd.DataFrame({"Setting": list(required.keys()), "Value": list(required.values())})
        _write_sheet_df(xlsx_path, "Settings", out)
        added = len(required)
    else:
        # Normalize column presence
        cols = {c.lower(): c for c in current.columns}
        if "setting" not in cols or "value" not in cols:
            # Rebuild with just Setting/Value preserving unknown columns is complex; keep minimal
            current = pd.DataFrame(columns=["Setting", "Value"])
        else:
            # Keep only these two for simplicity
            current = current.rename(columns={cols["setting"]: "Setting", cols["value"]: "Value"})[["Setting", "Value"]]
        # Build a map
        if not current.empty:
            # Normalize setting names for matching
            norm_map = {str(r["Setting"]).strip().lower(): i for i, r in current.reset_index().iterrows()}
        else:
            norm_map = {}
        for k, v in required.items():
            key_norm = k.strip().lower()
            if key_norm in norm_map:
                i = norm_map[key_norm]
                old = current.loc[i, "Value"]
                if pd.isna(old) or str(old).strip() == "":
                    current.loc[i, "Value"] = v
                    updated += 1
            else:
                current = pd.concat([current, pd.DataFrame([[k, v]], columns=["Setting", "Value"])], ignore_index=True)
                added += 1
        _write_sheet_df(xlsx_path, "Settings", current)
    summary = {"added": added, "updated": updated}
    _log_action("ensure_settings_sheet", f"added={added}, updated={updated}")
    return summary


def ensure_make_sheet(xlsx_path: Path) -> dict:
    """Ensure the 'Make' sheet exists and includes a row for every unique
    (Location, Equipment Name, Output) triplet from the Network with Process=Make.
    Output is taken from the Network sheet's Output column for Make processes.
    Returns a summary dict with counts of rows added.
    """
    # Read Network
    net = _read_network_df(xlsx_path)
    if net.empty:
        raise ValueError("Network sheet is empty or missing required columns.")
    make_rows = net[net["process"].str.upper() == "MAKE"].copy()
    if make_rows.empty:
        # Create empty if not exists, else do nothing
        existing = _read_sheet_df(xlsx_path, "Make")
        if existing is None:
            empty_df = pd.DataFrame(columns=MAKE_SHEET_COLS)
            _write_sheet_df(xlsx_path, "Make", empty_df)
        return {"rows_added": 0}

    # Build unique rows using Network Input and Output; key by Output
    uniq = (
        make_rows[["location", "equipment_name", "input", "output"]]
        .fillna("")
        .drop_duplicates()
        .rename(columns={"location": "Location", "equipment_name": "Equipment Name", "input": "Input", "output": "Output"})
        .reset_index(drop=True)
    )
    # FILTER: Remove rows where Output is empty/nan
    uniq = uniq[uniq["Output"].astype(str).str.strip() != ""]

    uniq["KEY"] = _normalize_key_triplet(uniq, "Location", "Equipment Name", "Output")

    # Read existing Make sheet (if any)
    current = _read_sheet_df(xlsx_path, "Make")
    created_new = False
    if current is None or current.empty:
        current = pd.DataFrame(columns=MAKE_SHEET_COLS)
        created_new = True

    # Build existing key set without modifying user's headings
    if not current.empty:
        # Use 'Output' if present, otherwise fall back to legacy 'Product' for keying
        key_third_col = "Output" if "Output" in current.columns else ("Product" if "Product" in current.columns else None)
        if key_third_col is None:
            existing_key_set = set()
        else:
            for c in ["Location", "Equipment Name", key_third_col]:
                if c in current.columns:
                    current[c] = current[c].astype(str)
            if key_third_col != "Output":
                temp = current.rename(columns={key_third_col: "Output"})
                current_keys = _normalize_key_triplet(temp, "Location", "Equipment Name", "Output")
            else:
                current_keys = _normalize_key_triplet(current, "Location", "Equipment Name", "Output")
            existing_key_set = set(current_keys.tolist())
    else:
        existing_key_set = set()

    rows_to_add = uniq[~uniq["KEY"].isin(existing_key_set)].copy()

    # Defaults - use empty strings as placeholders
    defaults = {
        "Mean Production Rate (Tons/hr)": "",
        "Std Dev of Production Rate (Tons/Hr)": "",
        "Planned Maintenance Dates (Days of year)": "",
        "Unplanned downtime %": "",
        "Consumption %": "",
    }

    added_count = 0
    if not rows_to_add.empty:
        # Create new rows with defaults (only for columns that already exist to preserve headings)
        for _, r in rows_to_add.iterrows():
            new_row = {}
            for col in current.columns:
                if col in {"Location", "Equipment Name", "Input", "Output"}:
                    new_row[col] = r.get(col, r.get(col.title(), "")) if col in r else r[col]
                elif col in defaults:
                    new_row[col] = defaults[col]
                else:
                    # leave user-defined extra columns blank
                    new_row[col] = ""
            current = pd.concat([current, pd.DataFrame([new_row])], ignore_index=True)
            added_count += 1

    # Preserve existing column order (do not inject or reorder headings)
    ordered_cols = list(current.columns)
    sort_cols = [c for c in ["Location", "Equipment Name", "Output"] if c in current.columns]

    # Enforce uniqueness on the key columns before writing
    if sort_cols:
        current = current.drop_duplicates(subset=sort_cols, keep="first")
        current = current.sort_values(sort_cols)

    current = current[ordered_cols].reset_index(drop=True)

    _write_sheet_df(xlsx_path, "Make", current)
    summary = {"rows_added": added_count}
    _log_action("ensure_make_sheet", f"rows_added={added_count}")
    return summary


def ensure_store_sheet(xlsx_path: Path) -> dict:
    """Ensure the 'Store' sheet exists and includes a row for every unique
    (Location, Equipment Name, Input) triplet from the Network with Process=Store.
    """
    net = _read_network_df(xlsx_path)
    if net.empty:
        raise ValueError("Network sheet is empty or missing required columns.")
    store_rows = net[net["process"].str.upper() == "STORE"].copy()
    
    if store_rows.empty:
        existing = _read_sheet_df(xlsx_path, "Store")
        if existing is None:
            empty_df = pd.DataFrame(columns=STORE_SHEET_COLS)
            _write_sheet_df(xlsx_path, "Store", empty_df)
        return {"rows_added": 0}

    uniq = (
        store_rows[["location", "equipment_name", "input"]]
        .fillna("")
        .drop_duplicates()
        .rename(columns={"location": "Location", "equipment_name": "Equipment Name", "input": "Input"})
        .reset_index(drop=True)
    )
    # FILTER: Remove rows where Input is empty/nan (good practice)
    uniq = uniq[uniq["Input"].str.strip() != ""]

    uniq["KEY"] = _normalize_key_triplet(uniq, "Location", "Equipment Name", "Input")

    current = _read_sheet_df(xlsx_path, "Store")
    if current is None or current.empty:
        current = pd.DataFrame(columns=STORE_SHEET_COLS)

    # Build existing key set without changing user's headings
    if not current.empty and all(c in current.columns for c in ["Location", "Equipment Name", "Input"]):
        current["Location"] = current["Location"].astype(str)
        current["Equipment Name"] = current["Equipment Name"].astype(str)
        current["Input"] = current["Input"].astype(str)
        current_keys = _normalize_key_triplet(current, "Location", "Equipment Name", "Input")
        existing_key_set = set(current_keys.tolist())
    else:
        existing_key_set = set()

    rows_to_add = uniq[~uniq["KEY"].isin(existing_key_set)].copy()

    defaults = {
        "Silo Max Capacity": "",
        "Silo Opening Stock (High)": "",
        "Silo Opening Stock (Low)": "",
    }

    added_count = 0
    if not rows_to_add.empty:
        for _, r in rows_to_add.iterrows():
            new_row = {"Location": r["Location"], "Equipment Name": r["Equipment Name"], "Input": r["Input"]}
            new_row.update(defaults)
            current = pd.concat([current, pd.DataFrame([new_row])], ignore_index=True)
            added_count += 1

    # Ensure required columns exist to avoid KeyError during reordering
    current = _ensure_columns(current, STORE_SHEET_COLS)

    extra_cols = [c for c in current.columns if c not in STORE_SHEET_COLS]
    ordered_cols = STORE_SHEET_COLS + extra_cols
    sort_cols = ["Location", "Equipment Name", "Input"]
    current = current[ordered_cols].sort_values(sort_cols).reset_index(drop=True)

    _write_sheet_df(xlsx_path, "Store", current)
    summary = {"rows_added": added_count}
    _log_action("ensure_store_sheet", f"rows_added={added_count}")
    return summary


def ensure_move_train_sheet(xlsx_path: Path) -> dict:
    """Ensure the 'Move_TRAIN' sheet exists with unique rows for train moves from the Network.

    Fields:
      - Product Class, Product, Origin Location, Origin Store, Destination Location, Destination Store,
        Distance, # Trains, # Carraiges, # Carraige Capacity (ton),
        Avg Speed - Loaded (km/hr), Avg Speed - Empty (km/hr)

    New behavior:
      - Auto-populate "Origin Store" and "Destination Store" from the Network when possible.
        • Origin Store: Prefer a Store row at the origin with matching product (Input) where Next Process=Move and Next Equipment=TRAIN.
          Fallback to any Store at origin with matching product.
        • Destination Store: Prefer from the Move row (Process=Move, Equipment=TRAIN) whose Next Process=Store at the destination;
          use Next Equipment as the destination store name. Fallback to any Store at destination with matching product.
      - Existing user-entered values are preserved (only blank cells are filled).
    """
    net = _read_network_df(xlsx_path)
    if net.empty:
        raise ValueError("Network sheet is empty or missing required columns.")

    # Convenience uppercase/trimmed helpers
    def _u(s: pd.Series) -> pd.Series:
        return s.astype(str).fillna("").str.strip().str.upper()

    mv = net[_u(net["process"]) == "MOVE"].copy()
    if mv.empty:
        existing = _read_sheet_df(xlsx_path, "Move_TRAIN")
        if existing is None:
            empty_df = pd.DataFrame(columns=MOVE_TRAIN_SHEET_COLS)
            _write_sheet_df(xlsx_path, "Move_TRAIN", empty_df)
        return {"rows_added": 0}

    # Filter to TRAIN equipment if present; otherwise ensure empty header
    mv_train = mv[_u(mv["equipment_name"]) == "TRAIN"].copy()
    if mv_train.empty:
        existing = _read_sheet_df(xlsx_path, "Move_TRAIN")
        if existing is None:
            empty_df = pd.DataFrame(columns=MOVE_TRAIN_SHEET_COLS)
            _write_sheet_df(xlsx_path, "Move_TRAIN", empty_df)
        return {"rows_added": 0}

    # Build helper maps from Network for store inference
    # Destination store from Move rows where next is Store
    dest_store_map: dict[tuple[str, str, str], str] = {}
    mv_t = mv_train.copy()
    for _, r in mv_t.iterrows():
        try:
            if str(r.get("next_process", "")).strip().upper() == "STORE":
                key = (
                    str(r.get("output", "")).strip().upper(),
                    str(r.get("location", "")).strip().upper(),
                    str(r.get("next_location", "")).strip().upper(),
                )
                dst_store = str(r.get("next_equipment", "")).strip()
                if key not in dest_store_map and dst_store:
                    dest_store_map[key] = dst_store
        except Exception:
            pass

    # Origin store from Store rows that feed a TRAIN move
    origin_store_map: dict[tuple[str, str], str] = {}
    store_rows = net[_u(net["process"]) == "STORE"].copy()
    for _, r in store_rows.iterrows():
        try:
            if str(r.get("next_process", "")).strip().upper() == "MOVE" and str(r.get("next_equipment", "")).strip().upper() == "TRAIN":
                key = (
                    str(r.get("input", "")).strip().upper(),
                    str(r.get("location", "")).strip().upper(),
                )
                origin_store = str(r.get("equipment_name", "")).strip()
                if key not in origin_store_map and origin_store:
                    origin_store_map[key] = origin_store
        except Exception:
            pass

    # Fallback maps: any Store at (product, location)
    any_store_map: dict[tuple[str, str], str] = {}
    if not store_rows.empty:
        tmp = (
            store_rows[["input", "location", "equipment_name"]]
            .dropna(subset=["input", "location", "equipment_name"]).copy()
        )
        tmp["k_prod"] = _u(tmp["input"])
        tmp["k_loc"] = _u(tmp["location"])
        tmp_sorted = tmp.sort_values(["k_loc", "k_prod", "equipment_name"], kind="mergesort")
        for _, r in tmp_sorted.iterrows():
            key = (r["k_prod"], r["k_loc"])
            if key not in any_store_map:
                any_store_map[key] = str(r.get("equipment_name", "")).strip()

    # Build unique template rows from MOVE rows
    cols_map = {
        "product_class": "Product Class",
        "output": "Product",
        "location": "Origin Location",
        "next_location": "Destination Location",
    }
    uniq = (
        mv_train[["product_class", "output", "location", "next_location"]]
        .fillna("")
        .rename(columns=cols_map)
        .drop_duplicates()
        .reset_index(drop=True)
    )
    uniq["KEY"] = (
        _u(uniq["Product Class"]) + "|" + _u(uniq["Product"]) + "|" + _u(uniq["Origin Location"]) + "|" + _u(uniq["Destination Location"])
    )

    current = _read_sheet_df(xlsx_path, "Move_TRAIN")
    if current is None or current.empty:
        current = pd.DataFrame(columns=MOVE_TRAIN_SHEET_COLS)
    else:
        current = _ensure_columns(current, MOVE_TRAIN_SHEET_COLS)

    if not current.empty and all(c in current.columns for c in ["Product Class", "Product", "Origin Location", "Destination Location"]):
        cur_key = (
            _u(current["Product Class"]) + "|" + _u(current["Product"]) + "|" + _u(current["Origin Location"]) + "|" + _u(current["Destination Location"]) 
        )
        existing_key_set = set(cur_key.tolist())
    else:
        existing_key_set = set()

    rows_to_add = uniq[~uniq["KEY"].isin(existing_key_set)].copy()

    defaults = {name: "" for name in MOVE_TRAIN_REQUIRED_INPUT_COLS}

    def infer_origin_store(prod: str, origin: str) -> str:
        k = (prod.strip().upper(), origin.strip().upper())
        if k in origin_store_map:
            return origin_store_map[k]
        if k in any_store_map:
            return any_store_map[k]
        return ""

    def infer_destination_store(prod: str, origin: str, dest: str) -> str:
        k3 = (prod.strip().upper(), origin.strip().upper(), dest.strip().upper())
        if k3 in dest_store_map:
            return dest_store_map[k3]
        # fallback to any store at destination for product
        k2 = (prod.strip().upper(), dest.strip().upper())
        if k2 in any_store_map:
            return any_store_map[k2]
        return ""

    added_count = 0
    if not rows_to_add.empty:
        for _, r in rows_to_add.iterrows():
            prod = str(r.get("Product", ""))
            o_loc = str(r.get("Origin Location", ""))
            d_loc = str(r.get("Destination Location", ""))
            new_row = {col: r.get(col, "") for col in ["Product Class", "Product", "Origin Location", "Destination Location"]}
            new_row["Origin Store"] = infer_origin_store(prod, o_loc)
            new_row["Destination Store"] = infer_destination_store(prod, o_loc, d_loc)
            for col in MOVE_TRAIN_REQUIRED_INPUT_COLS:
                if col not in new_row:
                    new_row[col] = defaults[col]
            # Respect existing column order and append
            filled = {c: new_row.get(c, "") for c in current.columns}
            for c in MOVE_TRAIN_SHEET_COLS:
                if c not in filled:
                    filled[c] = new_row.get(c, "")
            current = pd.concat([current, pd.DataFrame([filled])], ignore_index=True)
            added_count += 1

    # Second pass: backfill blanks in existing rows, preserving user-entered values
    if not current.empty:
        # Ensure text dtype for store and per-move rate columns to avoid pandas FutureWarning when writing strings
        for _col in ["Origin Store", "Destination Store", "Load Rate (tons/hr)", "Unload Rate (tons/hr)"]:
            if _col in current.columns:
                try:
                    current[_col] = current[_col].astype("object")
                except Exception:
                    pass
        for idx, r in current.iterrows():
            prod = str(r.get("Product", ""))
            o_loc = str(r.get("Origin Location", ""))
            d_loc = str(r.get("Destination Location", ""))
            if (pd.isna(r.get("Origin Store")) or str(r.get("Origin Store", "")).strip() == ""):
                current.at[idx, "Origin Store"] = infer_origin_store(prod, o_loc)
            if (pd.isna(r.get("Destination Store")) or str(r.get("Destination Store", "")).strip() == ""):
                current.at[idx, "Destination Store"] = infer_destination_store(prod, o_loc, d_loc)

    # Preserve existing column order
    ordered_cols = list(current.columns)
    sort_cols = [c for c in ["Product Class", "Product", "Origin Location", "Destination Location"] if c in current.columns]
    if sort_cols:
        current = current.drop_duplicates(subset=sort_cols, keep="first")
        current = current.sort_values(sort_cols)
    current = current[ordered_cols].reset_index(drop=True)

    _write_sheet_df(xlsx_path, "Move_TRAIN", current)
    summary = {"rows_added": added_count}
    _log_action("ensure_move_train_sheet", f"rows_added={added_count}, auto_filled_origin_dest_stores")
    return summary


def ensure_move_ship_sheet(xlsx_path: Path) -> dict:
    """Ensure the 'Move_SHIP' sheet exists and is seeded with unique origin locations.

    Rules:
      - Find unique Origin Locations from Network where Process=Move and Equipment Name=Ship.
      - Create or update 'Move_SHIP' with columns MOVE_SHIP_SHEET_COLS.
      - Append missing Origin Location rows; leave other fields blank for user input.
    """
    net = _read_network_df(xlsx_path)
    if net.empty:
        # still ensure header exists
        existing = _read_sheet_df(xlsx_path, "Move_SHIP")
        if existing is None:
            _write_sheet_df(xlsx_path, "Move_SHIP", pd.DataFrame(columns=MOVE_SHIP_SHEET_COLS))
        _log_action("ensure_move_ship_sheet", "rows_added=0 (empty Network)")
        return {"rows_added": 0}

    def _u(s: pd.Series) -> pd.Series:
        return s.astype(str).fillna("").str.strip().str.upper()

    ship_moves = net[(_u(net["process"]) == "MOVE") & (_u(net["equipment_name"]) == "SHIP")]
    origins = ship_moves["location"].astype(str).fillna("").str.strip()
    origins = origins[origins != ""].drop_duplicates().sort_values().reset_index(drop=True)

    # Read or create current sheet
    current = _read_sheet_df(xlsx_path, "Move_SHIP")
    if current is None or current.empty:
        current = pd.DataFrame(columns=MOVE_SHIP_SHEET_COLS)
    else:
        current = _ensure_columns(current.copy(), MOVE_SHIP_SHEET_COLS)

    # Build existing origin set
    if "Origin Location" in current.columns:
        existing_set = set(current["Origin Location"].astype(str).fillna("").str.strip().str.upper().tolist())
    else:
        existing_set = set()

    rows_to_add = [o for o in origins if o.upper() not in existing_set]

    added_count = 0
    for o in rows_to_add:
        row = {c: "" for c in MOVE_SHIP_SHEET_COLS}
        row["Origin Location"] = o
        current = pd.concat([current, pd.DataFrame([row])], ignore_index=True)
        added_count += 1

    # Ensure required columns and desired ordering (place new columns right after speed)
    desired = [c for c in MOVE_SHIP_SHEET_COLS if c in current.columns]
    # Include any extra legacy/user columns at the end to preserve previous data
    for c in current.columns:
        if c not in desired:
            desired.append(c)
    if "Origin Location" in current.columns:
        current = current.drop_duplicates(subset=["Origin Location"], keep="first")
        current = current.sort_values(["Origin Location"]).reset_index(drop=True)
    current = current.reindex(columns=desired)

    _write_sheet_df(xlsx_path, "Move_SHIP", current)
    _log_action("ensure_move_ship_sheet", f"rows_added={added_count}")
    return {"rows_added": added_count}


def ensure_delivery_sheet(xlsx_path: Path) -> dict:
    """Ensure the 'Deliver' sheet exists and includes a row for every unique
    (Location, Input) pair from the Network with Process=Deliver. Also ensure a
    'Demand Store' column exists immediately after 'Input' and auto-populate it
    using Network mapping where Process=Store and Next Process=Deliver and
    Next Equipment=TRUCK.
    """
    net = _read_network_df(xlsx_path)
    if net.empty:
        raise ValueError("Network sheet is empty or missing required columns.")
    del_rows = net[net["process"].str.upper() == "DELIVER"].copy()

    # --- FIX: Changed sheet name from 'Delivery' to 'Deliver' ---
    SHEET_NAME = "Deliver"

    if del_rows.empty:
        existing = _read_sheet_df(xlsx_path, SHEET_NAME)
        if existing is None:
            # Create with defined columns (includes optional 'Demand Store')
            empty_df = pd.DataFrame(columns=DELIVERY_SHEET_COLS)
            # Reorder to place 'Demand Store' after 'Input'
            desired_order = []
            for c in ["Location", "Input", "Demand Store"]:
                if c in empty_df.columns:
                    desired_order.append(c)
            for c in empty_df.columns:
                if c not in desired_order:
                    desired_order.append(c)
            empty_df = empty_df.reindex(columns=desired_order)
            _write_sheet_df(xlsx_path, SHEET_NAME, empty_df)
        _log_action("ensure_delivery_sheet", "rows_added=0 (no deliver rows in Network)")
        return {"rows_added": 0}

    uniq = (
        del_rows[["location", "input"]]
        .fillna("")
        .drop_duplicates()
        .rename(columns={"location": "Location", "input": "Input"})
        .reset_index(drop=True)
    )
    # FILTER: Remove rows where Input is empty/nan
    uniq = uniq[uniq["Input"].str.strip() != ""]

    uniq["KEY"] = _normalize_key_pair(uniq, "Location", "Input")

    current = _read_sheet_df(xlsx_path, SHEET_NAME)
    if current is None or current.empty:
        current = pd.DataFrame(columns=DELIVERY_SHEET_COLS)

    # Ensure required columns exist and position 'Demand Store' right after 'Input'
    for col in ["Location", "Input", "Demand Store"]:
        if col not in current.columns:
            current[col] = ""
    # Reorder columns to have 'Location', 'Input', 'Demand Store', then others
    ordered = []
    for c in ["Location", "Input", "Demand Store"]:
        if c in current.columns:
            ordered.append(c)
    for c in current.columns:
        if c not in ordered:
            ordered.append(c)
    current = current.reindex(columns=ordered)

    if not current.empty and all(c in current.columns for c in ["Location", "Input"]):
        current["Location"] = current["Location"].astype(str)
        current["Input"] = current["Input"].astype(str)
        current_keys = _normalize_key_pair(current, "Location", "Input")
        existing_key_set = set(current_keys.tolist())
    else:
        existing_key_set = set()

    rows_to_add = uniq[~uniq["KEY"].isin(existing_key_set)].copy()

    defaults = {
        "Demand per Location": "",
        "Demand Store": "",
    }

    added_count = 0
    if not rows_to_add.empty:
        for _, r in rows_to_add.iterrows():
            new_row = {}
            for col in current.columns:
                if col in {"Location", "Input"}:
                    new_row[col] = r.get(col, "")
                elif col in defaults:
                    new_row[col] = defaults[col]
                else:
                    new_row[col] = ""
            current = pd.concat([current, pd.DataFrame([new_row])], ignore_index=True)
            added_count += 1

    # Build mapping from Network: Store -> Deliver/TRUCK per (Location, Input)
    def _u(s: pd.Series) -> pd.Series:
        return s.astype(str).fillna("").str.strip().str.upper()

    store_to_deliver = net[( _u(net["process"]) == "STORE") & (_u(net["next_process"]) == "DELIVER") & (_u(net["next_equipment"]) == "TRUCK")]
    # Key: (Location, Input), Value: list of Equipment Name
    mapping: dict[tuple[str, str], list[str]] = {}
    for _, r in store_to_deliver.iterrows():
        loc = str(r.get("location", "")).strip()
        inp = str(r.get("input", "")).strip()
        eq = str(r.get("equipment_name", "")).strip()
        if not loc or not inp or not eq:
            continue
        key = (loc.upper(), inp.upper())
        mapping.setdefault(key, [])
        if eq not in mapping[key]:
            mapping[key].append(eq)

    # Optional: if Store sheet exists, build a set of valid (loc, eq, input)
    store_df = _read_sheet_df(xlsx_path, "Store")
    valid_store = set()
    if store_df is not None and not store_df.empty and all(c in store_df.columns for c in ["Location", "Equipment Name", "Input"]):
        tmp = store_df[["Location", "Equipment Name", "Input"]].astype(str).fillna("")
        for _, r in tmp.iterrows():
            valid_store.add((r["Location"].strip().upper(), r["Equipment Name"].strip().upper(), r["Input"].strip().upper()))

    # Populate or correct 'Demand Store'
    filled, ambiguous, missing, corrected_invalid = 0, 0, 0, 0
    if not current.empty:
        for idx, r in current.iterrows():
            loc = str(r.get("Location", "")).strip()
            inp = str(r.get("Input", "")).strip()
            cur = str(r.get("Demand Store", "")).strip()
            if not loc or not inp:
                continue
            key_pair = (loc.upper(), inp.upper())
            candidates = mapping.get(key_pair, [])

            # Helper to choose best candidate deterministically (valid stores only)
            def _choose(loc_u: str, inp_u: str, cand_list: list[str]) -> str:
                if not cand_list:
                    return ""
                valid = [eq for eq in cand_list if (loc_u, eq.upper(), inp_u) in valid_store]
                if not valid:
                    return ""
                if len(valid) == 1:
                    return valid[0]
                # Stable deterministic choice among valid candidates for repeatability
                return sorted(valid, key=lambda s: s.upper())[0]

            if cur == "":
                # Fill when empty
                chosen = _choose(key_pair[0], key_pair[1], candidates)
                if chosen:
                    current.at[idx, "Demand Store"] = chosen
                    filled += 1
                else:
                    if candidates:
                        ambiguous += 1  # multiple but could not choose uniquely
                    else:
                        missing += 1
            else:
                # Validate existing value; correct if invalid and we can determine a unique/preferred candidate
                is_valid = (key_pair[0], cur.upper(), key_pair[1]) in valid_store
                if not is_valid:
                    chosen = _choose(key_pair[0], key_pair[1], candidates)
                    if chosen:
                        current.at[idx, "Demand Store"] = chosen
                        corrected_invalid += 1
                    else:
                        # leave as-is but count as ambiguous/missing as appropriate
                        if candidates:
                            ambiguous += 1
                        else:
                            missing += 1

    ordered_cols = list(current.columns)
    sort_cols = [c for c in ["Location", "Input"] if c in current.columns]

    # Deduplicate/sort only if key cols present
    if sort_cols:
        current = current.drop_duplicates(subset=sort_cols, keep="first")
        current = current.sort_values(sort_cols)

    current = current[ordered_cols].reset_index(drop=True)

    _write_sheet_df(xlsx_path, SHEET_NAME, current)
    summary = {"rows_added": added_count, "demand_store_filled": filled, "demand_store_ambiguous": ambiguous, "demand_store_missing": missing}
    _log_action("ensure_delivery_sheet", f"rows_added={added_count}, demand_store_filled={filled}, ambiguous={ambiguous}, missing={missing}")
    return summary


def ensure_ship_berths_sheet(xlsx_path: Path) -> dict:
    """Ensure the 'SHIP_BERTHS' sheet exists and includes a row for every unique
    berth Location inferred from the Network using ship movement rules.

    We infer candidate Location values using the same logic as the legacy 'Berths' sheet:
      - If Process=Store AND Next Process=Move AND Next Equipment=SHIP, then Location := current Location
      - If Process=Move AND Equipment Name=SHIP, then Location := Next Location

    Existing rows are preserved; only new locations are appended with blank parameters.
    """
    net = _read_network_df(xlsx_path)
    if net.empty:
        # Ensure header exists
        current = _read_sheet_df(xlsx_path, "SHIP_BERTHS")
        if current is None:
            _write_sheet_df(xlsx_path, "SHIP_BERTHS", pd.DataFrame(columns=SHIP_BERTHS_SHEET_COLS))
        _log_action("ensure_ship_berths_sheet", "rows_added=0 (empty Network)")
        return {"rows_added": 0}

    def _u(s: pd.Series) -> pd.Series:
        return s.astype(str).fillna("").str.strip().str.upper()

    case_a = net[( _u(net["process"]) == "STORE") & (_u(net["next_process"]) == "MOVE") & (_u(net["next_equipment"]) == "SHIP")]
    loc_a = case_a["location"].astype(str).fillna("").str.strip()
    case_b = net[( _u(net["process"]) == "MOVE") & (_u(net["equipment_name"]) == "SHIP")]
    loc_b = case_b["next_location"].astype(str).fillna("").str.strip()

    locations = pd.Series(pd.concat([loc_a, loc_b], ignore_index=True)).replace({"nan": ""})
    locations = locations[locations != ""].drop_duplicates().sort_values().reset_index(drop=True)

    current = _read_sheet_df(xlsx_path, "SHIP_BERTHS")
    if current is None or current.empty:
        current = pd.DataFrame(columns=SHIP_BERTHS_SHEET_COLS)
    else:
        current = _ensure_columns(current.copy(), SHIP_BERTHS_SHEET_COLS)

    existing = set(current.get("Location", pd.Series(dtype=str)).astype(str).fillna("").str.strip().str.upper().tolist())

    added = 0
    for loc in locations:
        if loc.upper() in existing:
            continue
        row = {c: "" for c in SHIP_BERTHS_SHEET_COLS}
        row["Location"] = loc
        current = pd.concat([current, pd.DataFrame([row])], ignore_index=True)
        added += 1

    ordered_cols = list(current.columns)
    if "Location" in ordered_cols:
        current = current.drop_duplicates(subset=["Location"], keep="first").sort_values(["Location"]).reset_index(drop=True)
    current = current[ordered_cols]

    _write_sheet_df(xlsx_path, "SHIP_BERTHS", current)
    summary = {"rows_added": added}
    _log_action("ensure_ship_berths_sheet", f"rows_added={added}")
    return summary


def _is_routes_transposed(df: pd.DataFrame) -> bool:
    try:
        if df is None or df.empty:
            return False
        cols = [c for c in df.columns]
        if len(cols) == 0:
            return False
        # If first column is named 'Field' or contains many of our expected field names, treat as transposed
        first_col = cols[0]
        if str(first_col).strip().lower() == "field":
            return True
        # Heuristic: count overlaps between first column values and known fields
        values = set(str(v).strip() for v in df[first_col].dropna().tolist())
        overlap = len(values.intersection(set(SHIP_ROUTES_SHEET_COLS)))
        return overlap >= max(3, len(SHIP_ROUTES_SHEET_COLS)//4)
    except Exception:
        return False


def read_ship_routes_normalized(xlsx_path: Path) -> pd.DataFrame:
    """Read SHIP_ROUTES sheet and return a normalized wide dataframe with columns SHIP_ROUTES_SHEET_COLS.

    Supports two formats:
      1) Wide (columns are the field names) — returned as-is with ensured columns.
      2) Transposed (first column lists field names, subsequent columns are routes) — converted to wide rows.
    """
    df = _read_sheet_df(xlsx_path, "SHIP_ROUTES")
    if df is None or df.empty:
        return pd.DataFrame(columns=SHIP_ROUTES_SHEET_COLS)

    # If already wide
    if not _is_routes_transposed(df) and all(col in df.columns for col in ["Route Group", "Route ID", "Origin Location"]):
        df = _ensure_columns(df.copy(), SHIP_ROUTES_SHEET_COLS)
        return df

    # Transposed format
    cols = list(df.columns)
    field_col = cols[0]
    # Build one row per subsequent column
    rows: list[dict] = []
    field_names = df[field_col].astype(str).fillna("").str.strip()
    for col in cols[1:]:
        route_vals = {}
        series = df[col]
        for i, fname in enumerate(field_names):
            if fname in SHIP_ROUTES_SHEET_COLS:
                route_vals[fname] = series.iloc[i] if i < len(series) else pd.NA
        # Only append non-empty routes (must have at least Route Group or Route ID or Origin Location)
        if any(str(route_vals.get(k, "")).strip() for k in ["Route Group", "Route ID", "Origin Location"]):
            rows.append(route_vals)
    out = pd.DataFrame(rows, columns=SHIP_ROUTES_SHEET_COLS)
    out = _ensure_columns(out, SHIP_ROUTES_SHEET_COLS)
    return out


def ensure_ship_routes_sheet(xlsx_path: Path) -> dict:
    """Ensure the 'SHIP_ROUTES' sheet exists in TRANSPOSED format going forward.

    Behavior:
      - If sheet is missing: create a transposed template with a 'Field' column listing SHIP_ROUTES_SHEET_COLS
        and an example blank route column named 'Route 1'.
      - If sheet exists in old WIDE format: migrate to transposed, preserving data as separate route columns
        (column headers are Route Group-Route ID when available, otherwise 'Route N').
      - If already transposed: preserve as-is, only ensure the 'Field' list contains all required entries.
    """
    current = _read_sheet_df(xlsx_path, "SHIP_ROUTES")
    # Create fresh transposed template if missing or empty
    if current is None or current.empty:
        tdf = pd.DataFrame({
            "Field": SHIP_ROUTES_SHEET_COLS,
            "Route 1": ["" for _ in SHIP_ROUTES_SHEET_COLS],
        })
        _write_sheet_df(xlsx_path, "SHIP_ROUTES", tdf)
        _log_action("ensure_ship_routes_sheet", "initialized transposed template with 'Route 1'")
        return {"rows_added": 0}

    # If already transposed, just ensure all fields are present in the first column
    if _is_routes_transposed(current):
        field_col = current.columns[0]
        existing_fields = set(str(v).strip() for v in current[field_col].dropna().tolist())
        missing = [f for f in SHIP_ROUTES_SHEET_COLS if f not in existing_fields]
        if missing:
            # Append missing rows at the bottom
            add_df = pd.DataFrame({field_col: missing})
            # Create blank cells for all route columns
            for c in current.columns[1:]:
                add_df[c] = ""
            current = pd.concat([current, add_df], ignore_index=True)
        # Preserve order according to SHIP_ROUTES_SHEET_COLS when possible
        sorter = {name: i for i, name in enumerate(SHIP_ROUTES_SHEET_COLS)}
        try:
            current[field_col] = pd.Categorical(current[field_col], categories=SHIP_ROUTES_SHEET_COLS, ordered=True)
            current = current.sort_values(field_col)
            current[field_col] = current[field_col].astype(str)
        except Exception:
            pass
        _write_sheet_df(xlsx_path, "SHIP_ROUTES", current)
        _log_action("ensure_ship_routes_sheet", "kept transposed; ensured fields")
        return {"rows_added": 0}

    # Otherwise migrate from wide to transposed
    wide = _ensure_columns(current.copy(), SHIP_ROUTES_SHEET_COLS)
    cols = ["Field"]
    route_cols: list[str] = []
    # Build one column per route (row)
    for i, (_, r) in enumerate(wide.iterrows()):
        label_parts = []
        rg = str(r.get("Route Group", "")).strip()
        rid = str(r.get("Route ID", "")).strip()
        if rg:
            label_parts.append(rg)
        if rid:
            label_parts.append(rid)
        label = "-".join(label_parts) if label_parts else f"Route {i+1}"
        # Avoid duplicate column names
        base_label = label or f"Route {i+1}"
        label = base_label
        suffix = 2
        while label in route_cols:
            label = f"{base_label} ({suffix})"
            suffix += 1
        route_cols.append(label)
    cols += route_cols

    # Create transposed DataFrame
    tdf = pd.DataFrame({"Field": SHIP_ROUTES_SHEET_COLS})
    for label, (_, r) in zip(route_cols, wide.iterrows()):
        tdf[label] = [r.get(f, "") for f in SHIP_ROUTES_SHEET_COLS]

    _write_sheet_df(xlsx_path, "SHIP_ROUTES", tdf)
    _log_action("ensure_ship_routes_sheet", f"migrated from wide to transposed with {len(route_cols)} routes")
    return {"rows_added": 0}


def ensure_ship_distances_sheet(xlsx_path: Path) -> dict:
    """Ensure the 'SHIP_DISTANCES' sheet exists, deriving unique undirected Location 1/Location 2 pairs from SHIP_ROUTES.
    Existing distances are preserved; new pairs are appended with blank Distance (nM).
    Supports both wide and transposed SHIP_ROUTES by normalizing via read_ship_routes_normalized().

    Additionally enforces: no rows with blank endpoints and no NaN values in any column.
    Also migrates legacy columns 'From'/'To' to 'Location 1'/'Location 2'.
    """
    routes = read_ship_routes_normalized(xlsx_path)

    # Normalizer for location cells: treat None/NaN/"nan"/"none"/blank as empty string; otherwise trimmed text
    def _norm_loc(val) -> str:
        if pd.isna(val):
            return ""
        s = str(val).strip()
        if s.lower() in {"", "nan", "none"}:
            return ""
        return s

    # Start with existing distances
    dist_df = _read_sheet_df(xlsx_path, "SHIP_DISTANCES")
    migrated = False
    if dist_df is None or dist_df.empty:
        dist_df = pd.DataFrame(columns=SHIP_DISTANCES_SHEET_COLS)
    else:
        # If legacy columns present, rename them
        cols_lower = {c.lower(): c for c in dist_df.columns}
        if "from" in cols_lower and "to" in cols_lower:
            dist_df = dist_df.rename(columns={cols_lower["from"]: "Location 1", cols_lower["to"]: "Location 2"})
            migrated = True
        # Ensure required columns present
        dist_df = _ensure_columns(dist_df.copy(), SHIP_DISTANCES_SHEET_COLS)
        # Clean existing rows: drop any with blank endpoints; replace NaN in Distance with empty string
        dist_df["Location 1"] = dist_df["Location 1"].apply(_norm_loc).astype("object")
        dist_df["Location 2"] = dist_df["Location 2"].apply(_norm_loc).astype("object")
        # Drop rows where endpoints are empty
        dist_df = dist_df[(dist_df["Location 1"] != "") & (dist_df["Location 2"] != "")].copy()
        # Normalize Distance column to avoid NaN
        if "Distance (nM)" in dist_df.columns:
            dist_df["Distance (nM)"] = dist_df["Distance (nM)"].apply(lambda v: "" if pd.isna(v) else v)

    if migrated:
        _log_action("migrate_ship_distances_columns", "Renamed 'From'/'To' to 'Location 1'/'Location 2'")

    # Build set of existing undirected keys to avoid duplicates (Location 1/2 treated as undirected pair)
    def _pair_key(a: str, b: str) -> tuple[str, str]:
        a_ = _norm_loc(a)
        b_ = _norm_loc(b)
        return tuple(sorted([a_, b_], key=lambda x: x.upper()))

    existing_pairs = set()
    if not dist_df.empty:
        for _, r in dist_df.iterrows():
            f = _norm_loc(r.get("Location 1", ""))
            t = _norm_loc(r.get("Location 2", ""))
            if f and t:
                existing_pairs.add(_pair_key(f, t))

    new_pairs: list[tuple[str, str]] = []

    def _add_pair(a, b):
        a = _norm_loc(a)
        b = _norm_loc(b)
        if not a or not b:
            return
        k = _pair_key(a, b)
        if k not in existing_pairs and k not in new_pairs:
            new_pairs.append(k)

    if routes is not None and not routes.empty:
        # Extract combos as specified using a COMPRESSED sequence of non-empty locations
        # Sequence per route: Origin -> Dest1 -> Dest2 -> Dest3 -> Return
        # If any intermediate field is empty, we skip it and still connect the previous
        # non-empty location to the next non-empty location.
        for _, r in routes.iterrows():
            seq_raw = [
                r.get("Origin Location", ""),
                r.get("Destination 1 Location", ""),
                r.get("Destination 2 Location", ""),
                r.get("Destination 3 Location", ""),
                r.get("Return Location", ""),
            ]
            # Normalize and drop empties
            seq = [loc for loc in (_norm_loc(x) for x in seq_raw) if loc != ""]
            # Remove successive duplicates (case-insensitive)
            compressed: list[str] = []
            for loc in seq:
                if not compressed or compressed[-1].strip().upper() != loc.strip().upper():
                    compressed.append(loc)
            # Add consecutive pairs
            for i in range(len(compressed) - 1):
                _add_pair(compressed[i], compressed[i + 1])

    # Append new rows with blank Distance (nM)
    added = 0
    for a, b in new_pairs:
        row = {"Location 1": a, "Location 2": b, "Distance (nM)": ""}
        dist_df = pd.concat([dist_df, pd.DataFrame([row])], ignore_index=True)
        added += 1

    # Final cleanup: enforce no blanks/NaN and stable ordering
    if not dist_df.empty:
        dist_df["Location 1"] = dist_df["Location 1"].apply(_norm_loc).astype("object")
        dist_df["Location 2"] = dist_df["Location 2"].apply(_norm_loc).astype("object")
        if "Distance (nM)" in dist_df.columns:
            dist_df["Distance (nM)"] = dist_df["Distance (nM)"].apply(lambda v: "" if pd.isna(v) else v)
        # Drop rows with blank endpoints
        dist_df = dist_df[(dist_df["Location 1"] != "") & (dist_df["Location 2"] != "")]
        # Drop exact duplicates
        dist_df = dist_df.drop_duplicates(subset=["Location 1", "Location 2"], keep="first")
        # Sort
        dist_df = dist_df.sort_values(["Location 1", "Location 2"]).reset_index(drop=True)

    _write_sheet_df(xlsx_path, "SHIP_DISTANCES", dist_df)
    _log_action("ensure_ship_distances_sheet", f"rows_added={added}")
    return {"rows_added": added}


def prepare_inputs_excel(xlsx_path: Path) -> dict:
    """High-level entry to prepare Excel inputs: Settings, Make, Store, Move, Deliver, and Ship sheets.
    Returns a combined summary.
    NOTE: This function updates the provided workbook IN-PLACE. For the new
    non-destructive workflow that writes to a separate file, use
    `prepare_inputs_generate(src_xlsx, out_xlsx)`.
    """
    if not xlsx_path.exists():
        raise FileNotFoundError(xlsx_path)
    # Ensure it's an Excel file
    if xlsx_path.suffix.lower() not in {".xlsx", ".xlsm", ".xls"}:
        raise ValueError("prepare_inputs_excel requires an Excel workbook (.xlsx/.xlsm/.xls)")

    settings_summary = ensure_settings_sheet(xlsx_path)
    make_summary = ensure_make_sheet(xlsx_path)
    store_summary = ensure_store_sheet(xlsx_path)
    move_train_summary = ensure_move_train_sheet(xlsx_path)
    move_ship_summary = ensure_move_ship_sheet(xlsx_path)
    delivery_summary = ensure_delivery_sheet(xlsx_path)
    ship_berths_summary = ensure_ship_berths_sheet(xlsx_path)
    ship_routes_summary = ensure_ship_routes_sheet(xlsx_path)
    ship_distances_summary = ensure_ship_distances_sheet(xlsx_path)
    
    return {
        "settings": settings_summary,
        "make": make_summary,
        "store": store_summary,
        "move_train": move_train_summary,
        "move_ship": move_ship_summary,
        "deliver": delivery_summary,
        "ship_berths": ship_berths_summary,
        "ship_routes": ship_routes_summary,
        "ship_distances": ship_distances_summary,
    }


def prepare_inputs_generate(src_xlsx: Path, out_xlsx: Path) -> dict:
    """Generate a new workbook from the source model inputs and normalize sheets.

    Behavior:
    - Reads the Network sheet from `src_xlsx` and writes it to `out_xlsx`.
    - Copies existing Settings/Make/Store/Move/Deliver (or Delivery) and Berths from source into
      the generated file (sheet name unified as 'Deliver').
    - Runs ensure_... functions on `out_xlsx` so required rows/columns exist, including Berths.
    - Writes a 'Log' sheet listing all actions performed with timestamps.

    Returns a summary dict, same structure as `prepare_inputs_excel`.
    """
    if not src_xlsx.exists():
        raise FileNotFoundError(src_xlsx)
    if src_xlsx.suffix.lower() not in {".xlsx", ".xlsm", ".xls"}:
        raise ValueError("prepare_inputs requires an Excel source workbook (.xlsx/.xlsm/.xls)")
    if openpyxl is None:
        raise RuntimeError("openpyxl is required to write Excel files. Install with 'pip install openpyxl'.")

    # Reset action log for this run
    global ACTION_LOG
    ACTION_LOG = []
    _log_action("start_prepare_inputs_generate", f"src={src_xlsx}, out={out_xlsx}")

    # 1) Read Network from source
    try:
        net_df = pd.read_excel(src_xlsx, sheet_name="Network")
        _log_action("read_network", f"rows={len(net_df)} from {src_xlsx}")
    except Exception as e:
        _log_action("read_network_failed", str(e))
        raise RuntimeError(f"Failed to read 'Network' sheet from {src_xlsx}: {e}")

    # 2) Collect optional sheets to copy over
    copy_sheet_names = [
        "Settings", "Make", "Store", "Deliver", "Delivery",
        "Berths",  # legacy, preserved if present
        "Move_TRAIN", "Move_SHIP",
        "SHIP_BERTHS", "SHIP_ROUTES", "SHIP_DISTANCES",
    ]
    copied_dfs: dict[str, pd.DataFrame] = {}
    for nm in copy_sheet_names:
        df = _read_sheet_df(src_xlsx, nm)
        if df is not None:
            out_name = "Deliver" if nm.lower().startswith("deliver") else nm
            copied_dfs[out_name] = df
            _log_action("copy_sheet_found", f"{nm} -> {out_name}, rows={len(df)}")
        else:
            _log_action("copy_sheet_missing", f"{nm} not present; skipping")

    # 3) Write initial generated workbook (overwrite if exists)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl", mode="w") as writer:
        net_df.to_excel(writer, index=False, sheet_name="Network")
        _log_action("write_sheet", f"Network rows={len(net_df)} -> {out_xlsx}")
        for nm, df in copied_dfs.items():
            try:
                df.to_excel(writer, index=False, sheet_name=nm)
                _log_action("write_sheet", f"{nm} rows={len(df)} -> {out_xlsx}")
            except Exception as ex:
                _log_action("write_sheet_failed", f"{nm}: {ex}")
                # Best effort: skip problematic sheet copy
                pass

    # 4) Normalize/ensure sheets on the generated workbook
    settings_summary = ensure_settings_sheet(out_xlsx)
    make_summary = ensure_make_sheet(out_xlsx)
    store_summary = ensure_store_sheet(out_xlsx)
    move_train_summary = ensure_move_train_sheet(out_xlsx)
    move_ship_summary = ensure_move_ship_sheet(out_xlsx)
    delivery_summary = ensure_delivery_sheet(out_xlsx)
    ship_berths_summary = ensure_ship_berths_sheet(out_xlsx)
    ship_routes_summary = ensure_ship_routes_sheet(out_xlsx)
    ship_distances_summary = ensure_ship_distances_sheet(out_xlsx)

    # 5) Write the action log to a 'Log' sheet in the generated workbook
    try:
        log_df = pd.DataFrame(ACTION_LOG, columns=["Timestamp", "Action", "Details"])
        with pd.ExcelWriter(out_xlsx, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            log_df.to_excel(writer, index=False, sheet_name="Log")
        _log_action("write_log", f"entries={len(log_df)}")
    except Exception as ex:
        # If writing log fails, we still return summaries
        _log_action("write_log_failed", str(ex))

    # 6) Safety: remove legacy 'Move' sheet from the generated workbook if present
    try:
        wb = openpyxl.load_workbook(out_xlsx)
        if "Move" in wb.sheetnames:
            ws = wb["Move"]
            wb.remove(ws)
            wb.save(out_xlsx)
            _log_action("remove_legacy_move_sheet", "Deleted 'Move' from generated workbook")
    except Exception as ex:
        _log_action("remove_legacy_move_sheet_failed", str(ex))

    return {
        "generated_path": str(out_xlsx),
        "settings": settings_summary,
        "make": make_summary,
        "store": store_summary,
        "move_train": move_train_summary,
        "move_ship": move_ship_summary,
        "deliver": delivery_summary,
        "ship_berths": ship_berths_summary,
        "ship_routes": ship_routes_summary,
        "ship_distances": ship_distances_summary,
        "log_entries": len(ACTION_LOG),
    }

# ------------------------ CLI ----------------------------------------------

def write_sample_csv(out_path: Path) -> None:
    """Write the sample data (from the user's screenshot) as an Excel (.xlsx, sheet 'Network')
    or CSV depending on the provided extension."""
    rows = [
        [1, "GP", "Gladstone", "K1", "None", "Make", "CL", "Store", "Gladstone", "CL_STORE"],
        [2, "GP", "Gladstone", "CL_STORE", "CL", "Store", "CL", "Move", "Gladstone", "SHIP"],
        [2, "GP", "Gladstone", "CL_STORE", "CL", "Store", "CL", "Make", "Gladstone", "CM1"],
        [2, "GP", "Gladstone", "CL_STORE", "CL", "Store", "CL", "Make", "Gladstone", "CM2"],
        [3, "GP", "Gladstone", "SHIP", "CL", "Move", "CL", "Store", "Bulwer Island", "CL_STORE"],
        [3, "GP", "Gladstone", "CM1", "CL", "Make", "GP", "Store", "Gladstone", "GP_STORE"],
        [3, "GP", "Gladstone", "CM2", "CL", "Make", "GP", "Store", "Gladstone", "GP_STORE"],
        [4, "GP", "Gladstone", "GP_STORE", "GP", "Store", "GP", "Deliver", "Gladstone", "TRUCK"],
        [4, "GP", "Gladstone", "GP_STORE", "GP", "Store", "GP", "Move", "Gladstone", "TRAIN"],
        [4, "GP", "Gladstone", "GP_STORE", "GP", "Store", "GP", "Move", "Townsville", "SHIP"],
        [4, "GP", "Gladstone", "GP_STORE", "GP", "Store", "GP", "Move", "Port Kembla", "SHIP"],
        [4, "GP", "Gladstone", "GP_STORE", "GP", "Store", "GP", "Move", "Newcastle", "SHIP"],
        [4, "GP", "Gladstone", "GP_STORE", "GP", "Store", "GP", "Move", "Bulwer Island", "SHIP"],
        [5, "GP", "Gladstone", "ROAD", "GP", "Deliver", "GP", "", "", ""],
        [5, "GP", "Gladstone", "RAIL", "GP", "Move", "GP", "Store", "Mackay", "GP_STORE"],
        [5, "GP", "Townsville", "SHIP", "GP", "Move", "GP", "Store", "Townsville", "GP_STORE"],
        [5, "GP", "Port Kembla", "SHIP", "GP", "Move", "GP", "Store", "Port Kembla", "GP_STORE"],
        [5, "GP", "Newcastle", "SHIP", "GP", "Move", "GP", "Store", "Newcastle", "GP_STORE"],
        [5, "GP", "Bulwer Island", "SHIP", "GP", "Move", "GP", "Store", "Bulwer Island", "GP_STORE"],
        [6, "GP", "Mackay", "GP_STORE", "GP", "Store", "GP", "Deliver", "Mackay", "TRUCK"],
        [6, "GP", "Townsville", "GP_STORE", "GP", "Store", "GP", "Deliver", "Townsville", "TRUCK"],
        [6, "GP", "Port Kembla", "GP_STORE", "GP", "Store", "GP", "Deliver", "Port Kembla", "TRUCK"],
        [6, "GP", "Newcastle", "GP_STORE", "GP", "Store", "GP", "Deliver", "Newcastle", "TRUCK"],
        [6, "GP", "Bulwer Island", "GP_STORE", "GP", "Store", "GP", "Deliver", "Bulwer Island", "TRUCK"],
        [7, "GP", "Mackay", "TRUCK", "GP", "Deliver", "GP", "None", "", ""],
        [7, "GP", "Townsville", "TRUCK", "GP", "Deliver", "GP", "None", "", ""],
        [7, "GP", "Port Kembla", "TRUCK", "GP", "Deliver", "GP", "None", "", ""],
        [7, "GP", "Newcastle", "TRUCK", "GP", "Deliver", "GP", "None", "", ""],
        [7, "GP", "Bulwer Island", "TRUCK", "GP", "Deliver", "GP", "None", "", ""],
    ]
    cols = [
        "Level",
        "Product Class",
        "Location",
        "Equipment Name",
        "Input",
        "Process",
        "Output",
        "Next Process",
        "Next Location",
        "Next Equipment",
    ]
    df = pd.DataFrame(rows, columns=cols)
    if out_path.suffix.lower() == ".xlsx":
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Network")
    else:
        df.to_csv(out_path, index=False)


def _load_config_from_module(module_obj):
    # Support both module-level 'Config' class and pre-instantiated 'config' object
    cfg = None
    if hasattr(module_obj, 'Config'):
        try:
            cfg = module_obj.Config()  # type: ignore
        except Exception:
            pass
    if cfg is None and hasattr(module_obj, 'config'):
        cfg = getattr(module_obj, 'config')
    return cfg


def _load_config(config_path: Optional[str | Path] = None):
    """Load Config either from a provided path (.py) or from 'supply_chain_viz_config' in CWD.
    If not found, create a commented template and instruct the user.
    """
    import importlib.util, runpy
    if config_path:
        p = Path(config_path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        # Execute file to a dict and instantiate Config if present
        ns = runpy.run_path(str(p))
        if 'Config' in ns:
            return ns['Config']()
        elif 'config' in ns:
            return ns['config']
        else:
            raise RuntimeError("The provided config file does not define 'Config' or 'config'.")
    # Try import supply_chain_viz_config from CWD
    try:
        import importlib
        vc = importlib.import_module('supply_chain_viz_config')
        cfg = _load_config_from_module(vc)
        if cfg is not None:
            return cfg
    except Exception:
        pass
    # If not present, create a template
    template = Path('supply_chain_viz_config.py')
    if not template.exists():
        try:
            template.write_text(
                "# Auto-generated configuration template. Edit values and re-run.\n"
                "from dataclasses import dataclass\n\n"
                "@dataclass\n"
                "class Config:\n"
                "    in_path: str = 'Model Inputs.xlsx'\n"
                "    generated_inputs_path: str = 'generated_model_inputs.xlsx'\n"
                "    use_generated_inputs: bool = True\n"
                "    sheet: str | int | None = None\n"
                "    out_html: str = 'my_supply_chain.html'\n"
                "    open_after: bool = True\n"
                "    product_class: str | None = None\n"
                "    height: str = '90vh'\n"
                "    physics: bool = False\n"
                "    simplify_move: bool = True\n"
                "    grid_x_sep: int = 120\n"
                "    grid_y_sep: int = 160\n"
                "    cell_stack_sep: int = 140\n"
                "    fork_junction_dx: int = 45\n"
                "    fork_leaf_dx: int = 120\n"
                "    fork_prong_dy: int = 18\n"
                "    move_color: str = '#ff7f0e'\n"
                "    move_label_font_size: int = 8\n"
                "    move_label_pad: int = 6\n"
                "    move_label_bg: bool = True\n"
                "    move_label_text_color: str = '#333333'\n"
                "    move_label_bg_rgba: str = 'rgba(255,255,255,0.85)'\n"
                "    move_label_max_per_side: int | None = None\n"
                "    prepare_inputs: bool = False\n"
                "    show_legend: bool = True\n",
                encoding='utf-8'
            )
        except Exception:
            pass
    raise RuntimeError("No configuration found. A 'supply_chain_viz_config.py' template has been created in the current folder. Edit it and re-run.")


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    p = argparse.ArgumentParser(description="Supply chain visualization and SimPy scaffold (config-driven)")
    p.add_argument("--config", type=str, default=None, help="Path to a Python config file (supply_chain_viz_config.py). If omitted, the script loads supply_chain_viz_config from the current folder.")
    p.add_argument("--write-sample", type=str, default=None, help="Write sample file (XLSX preferred) to this path and exit unless a config points to another input.")
    args = p.parse_args(argv)

    if args.write_sample:
        out_file = Path(args.write_sample)
        write_sample_csv(out_file)
        print(f"Sample file written to: {out_file}")

    # Load configuration
    try:
        config = _load_config(args.config)
    except Exception as e:
        print(e)
        return 1

    in_path = Path(getattr(config, 'in_path', 'Model Inputs.xlsx'))
    out_html = Path(getattr(config, 'out_html', 'my_supply_chain.html'))
    gen_path = Path(getattr(config, 'generated_inputs_path', 'generated_model_inputs.xlsx'))
    use_generated = bool(getattr(config, 'use_generated_inputs', True))

    # Optional: prepare inputs and exit (non-destructive; writes a new workbook)
    if bool(getattr(config, 'prepare_inputs', False)):
        try:
            summary = prepare_inputs_generate(in_path, gen_path)
            print("Generated model inputs created:")
            print(f"  File: {Path(summary['generated_path']).resolve()}")
            print(f"  Settings: added={summary['settings'].get('added', 0)}, updated={summary['settings'].get('updated', 0)}")
            print(f"  Make: rows_added={summary['make'].get('rows_added', 0)}")
            print(f"  Store: rows_added={summary['store'].get('rows_added', 0)}")
            print(f"  Move_TRAIN: rows_added={summary.get('move_train', {}).get('rows_added', 0)}")
            print(f"  Move_SHIP: rows_added={summary.get('move_ship', {}).get('rows_added', 0)}")
            print(f"  Deliver: rows_added={summary.get('deliver', {}).get('rows_added', 0)}")
            print(f"  SHIP_BERTHS: rows_added={summary.get('ship_berths', {}).get('rows_added', 0)}")
            print(f"  SHIP_ROUTES: rows_added={summary.get('ship_routes', {}).get('rows_added', 0)}")
            print(f"  SHIP_DISTANCES: rows_added={summary.get('ship_distances', {}).get('rows_added', 0)}")
        except Exception as e:
            print(f"Input preparation failed: {e}")
            return 1
        # If prepare_inputs only, stop here
        print("prepare_inputs=True was set in config; exiting after preparation.")
        return 0

    # Choose input for visualization
    input_used = in_path
    if use_generated and gen_path.exists():
        input_used = gen_path
        print(f"Using generated inputs workbook: {input_used.resolve()}")
    else:
        if use_generated:
            print(f"Generated inputs file not found at {gen_path.resolve()}; falling back to source: {in_path.resolve()}")
        else:
            print(f"Using source inputs workbook: {in_path.resolve()}")

    # Build graph
    df = read_table(input_used, sheet=getattr(config, 'sheet', None))
    G = build_graph(df, product_class=getattr(config, 'product_class', None))
    try:
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
    except Exception:
        n_nodes = n_edges = -1
    print(f"Graph built: {n_nodes} nodes, {n_edges} edges.")
    if n_nodes == 0 or n_edges == 0:
        print("Warning: The graph appears empty. Check input and product_class filter in supply_chain_viz_config.py.")

    export_pyvis(G, out_html, config)

    print(f"HTML visualization saved to: {out_html.resolve()}")

    # Optional auto-open
    if bool(getattr(config, 'open_after', False)):
        try:
            webbrowser.open(out_html.resolve().as_uri())
        except Exception:
            webbrowser.open(str(out_html.resolve()))

    # SimPy scaffold summary (still runs, no flags required)
    try:
        builder = build_simpy_model_from_dataframe(df, product_class=getattr(config, 'product_class', None))
        env, resources, routes = builder()
        print(f"SimPy scaffold created. Resources: {len(resources)}; Routes: {sum(len(v) for v in routes.values())}")
    except Exception as e:
        print(f"SimPy scaffold note: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())