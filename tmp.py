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


def node_id(location: str, equipment: str, product_class: Optional[str] = None) -> str:
    """Return a stable node identifier.

    If a product_class is provided, include it in the node id so that
    nodes for different product classes do not overlap/merge. This keeps
    backward compatibility when product_class is omitted.
    """
    if product_class is None or str(product_class).strip() == "":
        return f"{equipment}@{location}"
    return f"{equipment}@{location}#{str(product_class).strip()}"


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
            lvl = int(r["level"]) if pd.notna(r["level"]) else 10 ** 9
        except Exception:
            lvl = 10 ** 9
        cur_min = loc_min_level.get(loc, 10 ** 9)
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
        return (loc_min_level.get(loc, 10 ** 9), loc_first_idx.get(loc, 10 ** 9), loc)

    ordered_locations = sorted(all_locs, key=_loc_rank)
    loc_to_row_index: Dict[str, int] = {loc: i for i, loc in enumerate(ordered_locations)}

    # Add nodes and infer levels for destinations
    for _, row in df.iterrows():
        src_level_val = int(row["level"]) if pd.notna(row["level"]) else 0
        pc = str(row.get("product_class", "")).strip()
        # Node id includes product class to avoid overlap; assign row index by Location
        src_node = node_id(row["location"], row["equipment_name"], pc) if row["equipment_name"] else None
        if src_node:
            G.add_node(
                src_node,
                title=f"Product Class: {pc}<br>Location: {row['location']}<br>Equipment: {row['equipment_name']}",
                level=src_level_val,
                product_class=pc,
                location=str(row["location"]).strip(),
                equipment=row["equipment_name"],
                loc_index=loc_to_row_index.get(str(row["location"]).strip(), 0),
            )
        # destination node might be blank (terminal)
        if str(row["next_location"]).strip() and str(row["next_equipment"]).strip():
            dst_node = node_id(row["next_location"], row["next_equipment"], pc)
            # infer destination level from source (place to the right)
            dst_level = src_level_val + 1
            if dst_node not in G:
                G.add_node(
                    dst_node,
                    title=f"Product Class: {pc}<br>Location: {row['next_location']}<br>Equipment: {row['next_equipment']}",
                    product_class=pc,
                    location=str(row["next_location"]).strip(),
                    equipment=row["next_equipment"],
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
        src = node_id(row["location"], row["equipment_name"], pc_row) if row["equipment_name"] else None
        if not src or src not in G:
            continue
        next_loc = str(row["next_location"]).strip()
        next_eq = str(row["next_equipment"]).strip()
        if not (next_loc and next_eq):
            # terminal step — no outgoing edge
            continue
        dst = node_id(next_loc, next_eq, pc_row)
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
                dst_loc_idx = int(G.nodes.get(v, {}).get("loc_index", 10 ** 9))
                out_lbl = str(data.get("label", "")).strip()
                outgoing_moves.setdefault(u, []).append((dst_loc_idx, dst_loc, dst_eq, out_lbl))

                src_loc = G.nodes.get(u, {}).get("location", "")
                src_eq = G.nodes.get(u, {}).get("equipment", "")
                src_loc_idx = int(G.nodes.get(u, {}).get("loc_index", 10 ** 9))
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
            color = color_override if (color_override is not None) else getattr(config, 'move_label_text_color',
                                                                                '#333333')
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
            net.add_node(node_id, label="", image=data_url, shape="image", size=1, physics=False, fixed=True,
                         x=int(x_center), y=int(y_center))
        except Exception:
            # Fallback: simple label node
            net.add_node(node_id, label=str(text), shape="box", physics=False, fixed=True, x=int(x_center),
                         y=int(y_center))

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
            eq_col = cols_lower.get('equipment name',
                                    'Equipment Name' if 'Equipment Name' in store_df.columns else None)
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
        src = node_id(r["location"], r["equipment_name"], pc_row) if r["equipment_name"] else None
        if not src:
            continue
        res_caps.setdefault(src, 1)  # default capacity 1; to be overridden from Excel later
        next_loc = str(r["next_location"]).strip()
        next_eq = str(r["next_equipment"]).strip()
        if next_loc and next_eq:
            dst = node_id(next_loc, next_eq, pc_row)
            routes.setdefault(src, []).append(
                (dst, {"process": r["process"], "input": r["input"], "output": r["output"]})
            )

    def build_env():
        env = simpy.Environment()
        # Create resources
        resources: Dict[str, simpy.Resource] = {name: simpy.Resource(env, capacity=cap) for name, cap in
                                                res_caps.items()}

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
        start_node = node_id(start_row["location"], start_row["equipment_name"], pc_row) if start_row[
            "equipment_name"] else list(resources.keys())[0]
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
MAKE_KEY_COLS = ["Location", "Equipment Name", "Input"]
MAKE_SHEET_COLS = MAKE_KEY_COLS + MAKE_REQUIRED_INPUT_COLS

STORE_REQUIRED_INPUT_COLS = [
    "Silo Max Capacity",
    "Silo Opening Stock (High)",
    "Silo Opening Stock (Low)",
]
STORE_KEY_COLS = ["Location", "Equipment Name", "Input"]
STORE_SHEET_COLS = STORE_KEY_COLS + STORE_REQUIRED_INPUT_COLS

DELIVERY_REQUIRED_INPUT_COLS = [
    "Demand per Location",
]
DELIVERY_KEY_COLS = ["Location", "Input"]
DELIVERY_SHEET_COLS = DELIVERY_KEY_COLS + DELIVERY_REQUIRED_INPUT_COLS

# MOVE sheet
MOVE_REQUIRED_INPUT_COLS = [
    "#Equipment (99-unlimited)",
    "#Parcels",
    "Capacity Per Parcel",
    "Load Rate (Ton/hr)",
    "Travel to Time (Min)",
    "Unload Rate (Ton/Hr)",
    "Travel back Time (Min)",
]
MOVE_KEY_COLS = ["Product Class", "Location", "Equipment Name", "Next Location"]
MOVE_SHEET_COLS = MOVE_KEY_COLS + MOVE_REQUIRED_INPUT_COLS


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

    return _norm(df_like[pc_col]) + "|" + _norm(df_like[loc_col]) + "|" + _norm(df_like[eq_col]) + "|" + _norm(
        df_like[next_loc_col])


def _ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
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
    return {"added": added, "updated": updated}


def ensure_make_sheet(xlsx_path: Path) -> dict:
    """Ensure the 'Make' sheet exists and includes a row for every unique
    (Location, Equipment Name, Input) triplet from the Network with Process=Make.
    Returns a summary dict with counts of rows added and fields filled.
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
        return {"rows_added": 0, "fields_filled": 0}

    uniq = (
        make_rows[["location", "equipment_name", "input"]]
        .fillna("")
        .drop_duplicates()
        .rename(columns={"location": "Location", "equipment_name": "Equipment Name", "input": "Input"})
        .reset_index(drop=True)
    )
    # FILTER: Remove rows where Input is empty/nan
    uniq = uniq[uniq["Input"].str.strip() != ""]

    uniq["KEY"] = _normalize_key_triplet(uniq, "Location", "Equipment Name", "Input")

    # Read existing Make sheet (if any)
    current = _read_sheet_df(xlsx_path, "Make")
    if current is None or current.empty:
        current = pd.DataFrame(columns=MAKE_SHEET_COLS)

    current = _ensure_columns(current, MAKE_SHEET_COLS)

    if not current.empty:
        current["Location"] = current["Location"].astype(str)
        current["Equipment Name"] = current["Equipment Name"].astype(str)
        current["Input"] = current["Input"].astype(str)
        current_keys = _normalize_key_triplet(current, "Location", "Equipment Name", "Input")
        existing_key_set = set(current_keys.tolist())
    else:
        existing_key_set = set()

    rows_to_add = uniq[~uniq["KEY"].isin(existing_key_set)].copy()

    # Defaults - now using empty strings instead of 0.0
    defaults = {
        "Mean Production Rate (Tons/hr)": "",
        "Std Dev of Production Rate (Tons/Hr)": "",
        "Planned Maintenance Dates (Days of year)": "",
        "Unplanned downtime %": "",
        "Consumption %": "",
    }

    added_count = 0
    if not rows_to_add.empty:
        # Create new rows with defaults
        for _, r in rows_to_add.iterrows():
            new_row = {"Location": r["Location"], "Equipment Name": r["Equipment Name"], "Input": r["Input"]}
            new_row.update(defaults)
            current = pd.concat([current, pd.DataFrame([new_row])], ignore_index=True)
            added_count += 1

    # Order columns
    extra_cols = [c for c in current.columns if c not in MAKE_SHEET_COLS]
    ordered_cols = MAKE_SHEET_COLS + extra_cols
    sort_cols = ["Location", "Equipment Name", "Input"]

    # --- FIX: STRICT DEDUPLICATION ---
    # Enforce uniqueness on the key columns before writing.
    # This cleans up any existing duplicates and ensures new rows didn't create conflicts.
    current = current.drop_duplicates(subset=sort_cols, keep="first")

    current = current[ordered_cols].sort_values(sort_cols).reset_index(drop=True)

    _write_sheet_df(xlsx_path, "Make", current)
    return {"rows_added": added_count}


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

    current = _ensure_columns(current, STORE_SHEET_COLS)

    if not current.empty:
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

    extra_cols = [c for c in current.columns if c not in STORE_SHEET_COLS]
    ordered_cols = STORE_SHEET_COLS + extra_cols
    sort_cols = ["Location", "Equipment Name", "Input"]
    current = current[ordered_cols].sort_values(sort_cols).reset_index(drop=True)

    _write_sheet_df(xlsx_path, "Store", current)
    return {"rows_added": added_count}


def ensure_move_sheet(xlsx_path: Path) -> dict:
    """Ensure the 'Move' sheet exists and includes a row for every unique
    (Product Class, Location, Equipment Name, Next Location) quad from the Network with Process=Move.

    The following parameter fields are created and left blank for user input per route:
      - #Equipment (99-unlimited)
      - #Parcels
      - Capacity Per Parcel
      - Load Rate (Ton/hr)
      - Travel to Time (Min)
      - Unload Rate (Ton/Hr)
      - Travel back Time (Min)
    """
    net = _read_network_df(xlsx_path)
    if net.empty:
        raise ValueError("Network sheet is empty or missing required columns.")

    move_rows = net[net["process"].str.upper() == "MOVE"].copy()

    if move_rows.empty:
        existing = _read_sheet_df(xlsx_path, "Move")
        if existing is None:
            empty_df = pd.DataFrame(columns=MOVE_SHEET_COLS)
            _write_sheet_df(xlsx_path, "Move", empty_df)
        return {"rows_added": 0}

    # Build unique key tuples
    uniq = (
        move_rows[["product_class", "location", "equipment_name", "next_location"]]
        .fillna("")
        .drop_duplicates()
        .rename(
            columns={
                "product_class": "Product Class",
                "location": "Location",
                "equipment_name": "Equipment Name",
                "next_location": "Next Location",
            }
        )
        .reset_index(drop=True)
    )

    # If Product Class is entirely blank, we still allow the row, using empty string as the class.
    uniq["KEY"] = _normalize_key_quad(uniq, "Product Class", "Location", "Equipment Name", "Next Location")

    # Read current Move sheet
    current = _read_sheet_df(xlsx_path, "Move")
    if current is None or current.empty:
        current = pd.DataFrame(columns=MOVE_SHEET_COLS)

    current = _ensure_columns(current, MOVE_SHEET_COLS)

    # Build existing key set from current
    if not current.empty:
        # Ensure type consistency
        for c in ["Product Class", "Location", "Equipment Name", "Next Location"]:
            current[c] = current[c].astype(str)
        current_keys = _normalize_key_quad(current, "Product Class", "Location", "Equipment Name", "Next Location")
        existing_key_set = set(current_keys.tolist())
    else:
        existing_key_set = set()

    rows_to_add = uniq[~uniq["KEY"].isin(existing_key_set)].copy()

    defaults = {name: "" for name in MOVE_REQUIRED_INPUT_COLS}

    added_count = 0
    if not rows_to_add.empty:
        for _, r in rows_to_add.iterrows():
            new_row = {
                "Product Class": r["Product Class"],
                "Location": r["Location"],
                "Equipment Name": r["Equipment Name"],
                "Next Location": r["Next Location"],
            }
            new_row.update(defaults)
            current = pd.concat([current, pd.DataFrame([new_row])], ignore_index=True)
            added_count += 1

    # Order columns and sort for stability; enforce uniqueness on key cols
    extra_cols = [c for c in current.columns if c not in MOVE_SHEET_COLS]
    ordered_cols = MOVE_SHEET_COLS + extra_cols
    sort_cols = ["Product Class", "Location", "Equipment Name", "Next Location"]

    current = current.drop_duplicates(subset=sort_cols, keep="first")
    current = current[ordered_cols].sort_values(sort_cols).reset_index(drop=True)

    _write_sheet_df(xlsx_path, "Move", current)
    return {"rows_added": added_count}


def ensure_delivery_sheet(xlsx_path: Path) -> dict:
    """Ensure the 'Deliver' sheet exists and includes a row for every unique
    (Location, Input) pair from the Network with Process=Deliver.
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
            empty_df = pd.DataFrame(columns=DELIVERY_SHEET_COLS)
            _write_sheet_df(xlsx_path, SHEET_NAME, empty_df)
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

    current = _ensure_columns(current, DELIVERY_SHEET_COLS)

    if not current.empty:
        current["Location"] = current["Location"].astype(str)
        current["Input"] = current["Input"].astype(str)
        current_keys = _normalize_key_pair(current, "Location", "Input")
        existing_key_set = set(current_keys.tolist())
    else:
        existing_key_set = set()

    rows_to_add = uniq[~uniq["KEY"].isin(existing_key_set)].copy()

    defaults = {
        "Demand per Location": "",
    }

    added_count = 0
    if not rows_to_add.empty:
        for _, r in rows_to_add.iterrows():
            new_row = {"Location": r["Location"], "Input": r["Input"]}
            new_row.update(defaults)
            current = pd.concat([current, pd.DataFrame([new_row])], ignore_index=True)
            added_count += 1

    extra_cols = [c for c in current.columns if c not in DELIVERY_SHEET_COLS]
    ordered_cols = DELIVERY_SHEET_COLS + extra_cols
    sort_cols = ["Location", "Input"]

    # Also good practice to deduplicate here as well
    current = current.drop_duplicates(subset=sort_cols, keep="first")

    current = current[ordered_cols].sort_values(sort_cols).reset_index(drop=True)

    _write_sheet_df(xlsx_path, SHEET_NAME, current)
    return {"rows_added": added_count}


def ensure_berths_sheet(xlsx_path: Path) -> dict:
    """Ensure the 'Berths' sheet exists and includes a row for every unique
    berth Location inferred from the Network using ship movement rules.

    Rules for inferring a Berth name (string):
      - If Process=Store AND Next Process=Move AND Next Equipment=SHIP, then Berth := Location
      - If Process=Move AND Equipment Name=SHIP, then Berth := Next Location

    Only non-empty berth names are considered. Existing rows are preserved and
    only new berth names are appended, with required fields left blank for user input.
    """
    net = _read_network_df(xlsx_path)
    if net.empty:
        raise ValueError("Network sheet is empty or missing required columns.")

    # Normalize strings used in conditions
    def _u(s: pd.Series) -> pd.Series:
        return s.astype(str).fillna("").str.strip().str.upper()

    # Case A: Store -> Next Move by SHIP => current Location is a Berth
    case_a = net[
        (_u(net["process"]) == "STORE")
        & (_u(net["next_process"]) == "MOVE")
        & (_u(net["next_equipment"]) == "SHIP")
        ]
    berths_a = case_a["location"].astype(str).fillna("").str.strip()

    # Case B: Move by SHIP => Next Location is a Berth
    case_b = net[
        (_u(net["process"]) == "MOVE")
        & (_u(net["equipment_name"]) == "SHIP")
        ]
    berths_b = case_b["next_location"].astype(str).fillna("").str.strip()

    berth_names = pd.Series(pd.concat([berths_a, berths_b], ignore_index=True)).replace({"nan": ""})
    berth_names = berth_names[berth_names != ""]
    berth_names = berth_names.drop_duplicates().sort_values().reset_index(drop=True)

    # Build the unique df for candidate rows
    uniq = pd.DataFrame({"Berth": berth_names})
    if uniq.empty:
        # Ensure sheet exists even if no berths inferred
        existing = _read_sheet_df(xlsx_path, "Berths")
        if existing is None:
            empty_df = pd.DataFrame(columns=BERTHS_SHEET_COLS)
            _write_sheet_df(xlsx_path, "Berths", empty_df)
        return {"rows_added": 0}

    # Read current Berths sheet
    current = _read_sheet_df(xlsx_path, "Berths")
    if current is None or current.empty:
        current = pd.DataFrame(columns=BERTHS_SHEET_COLS)

    current = _ensure_columns(current, BERTHS_SHEET_COLS)

    # Build existing key set (normalized berth name)
    if not current.empty:
        cur_keys = current["Berth"].astype(str).fillna("").str.strip().str.upper()
        existing_key_set = set(cur_keys.tolist())
    else:
        existing_key_set = set()

    uniq["KEY"] = uniq["Berth"].astype(str).fillna("").str.strip().str.upper()
    rows_to_add = uniq[~uniq["KEY"].isin(existing_key_set)].copy()

    defaults = {name: "" for name in BERTHS_REQUIRED_INPUT_COLS}

    added_count = 0
    if not rows_to_add.empty:
        for _, r in rows_to_add.iterrows():
            new_row = {"Berth": r["Berth"]}
            new_row.update(defaults)
            current = pd.concat([current, pd.DataFrame([new_row])], ignore_index=True)
            added_count += 1

    # Preserve extra columns if any and sort
    extra_cols = [c for c in current.columns if c not in BERTHS_SHEET_COLS]
    ordered_cols = BERTHS_SHEET_COLS + extra_cols
    sort_cols = ["Berth"]

    current = current.drop_duplicates(subset=sort_cols, keep="first")
    current = current[ordered_cols].sort_values(sort_cols).reset_index(drop=True)

    _write_sheet_df(xlsx_path, "Berths", current)
    return {"rows_added": added_count}


def prepare_inputs_excel(xlsx_path: Path) -> dict:
    """High-level entry to prepare Excel inputs: Settings, Make, Store, Move, Deliver, and Berths sheets.
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
    move_summary = ensure_move_sheet(xlsx_path)
    delivery_summary = ensure_delivery_sheet(xlsx_path)
    berths_summary = ensure_berths_sheet(xlsx_path)

    return {
        "settings": settings_summary,
        "make": make_summary,
        "store": store_summary,
        "move": move_summary,
        "deliver": delivery_summary,
        "berths": berths_summary,
    }


def prepare_inputs_generate(src_xlsx: Path, out_xlsx: Path) -> dict:
    """Generate a new workbook from the source model inputs and normalize sheets.

    Behavior:
    - Reads the Network sheet from `src_xlsx` and writes it to `out_xlsx`.
    - Copies existing Settings/Make/Store/Move/Deliver (or Delivery) and Berths from source into
      the generated file (sheet name unified as 'Deliver').
    - Runs ensure_... functions on `out_xlsx` so required rows/columns exist, including Berths.

    Returns a summary dict, same structure as `prepare_inputs_excel`.
    """
    if not src_xlsx.exists():
        raise FileNotFoundError(src_xlsx)
    if src_xlsx.suffix.lower() not in {".xlsx", ".xlsm", ".xls"}:
        raise ValueError("prepare_inputs requires an Excel source workbook (.xlsx/.xlsm/.xls)")
    if openpyxl is None:
        raise RuntimeError("openpyxl is required to write Excel files. Install with 'pip install openpyxl'.")

    # 1) Read Network from source
    try:
        net_df = pd.read_excel(src_xlsx, sheet_name="Network")
    except Exception as e:
        raise RuntimeError(f"Failed to read 'Network' sheet from {src_xlsx}: {e}")

    # 2) Collect optional sheets to copy over
    copy_sheet_names = ["Settings", "Make", "Store", "Move", "Deliver", "Delivery", "Berths"]
    copied_dfs: dict[str, pd.DataFrame] = {}
    for nm in copy_sheet_names:
        df = _read_sheet_df(src_xlsx, nm)
        if df is not None:
            out_name = "Deliver" if nm.lower().startswith("deliver") else nm
            copied_dfs[out_name] = df

    # 3) Write initial generated workbook (overwrite if exists)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl", mode="w") as writer:
        net_df.to_excel(writer, index=False, sheet_name="Network")
        for nm, df in copied_dfs.items():
            try:
                df.to_excel(writer, index=False, sheet_name=nm)
            except Exception:
                # Best effort: skip problematic sheet copy
                pass

    # 4) Normalize/ensure sheets on the generated workbook
    settings_summary = ensure_settings_sheet(out_xlsx)
    make_summary = ensure_make_sheet(out_xlsx)
    store_summary = ensure_store_sheet(out_xlsx)
    move_summary = ensure_move_sheet(out_xlsx)
    delivery_summary = ensure_delivery_sheet(out_xlsx)

    return {
        "generated_path": str(out_xlsx),
        "settings": settings_summary,
        "make": make_summary,
        "store": store_summary,
        "move": move_summary,
        "deliver": delivery_summary,
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
    raise RuntimeError(
        "No configuration found. A 'supply_chain_viz_config.py' template has been created in the current folder. Edit it and re-run.")


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    p = argparse.ArgumentParser(description="Supply chain visualization and SimPy scaffold (config-driven)")
    p.add_argument("--config", type=str, default=None,
                   help="Path to a Python config file (viz_config.py). If omitted, the script loads viz_config from the current folder.")
    p.add_argument("--write-sample", type=str, default=None,
                   help="Write sample file (XLSX preferred) to this path and exit unless a config points to another input.")
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
            print(
                f"  Settings: added={summary['settings'].get('added', 0)}, updated={summary['settings'].get('updated', 0)}")
            print(f"  Make: rows_added={summary['make'].get('rows_added', 0)}")
            print(f"  Store: rows_added={summary['store'].get('rows_added', 0)}")
            print(f"  Move: rows_added={summary['move'].get('rows_added', 0)}")
            print(f"  Deliver: rows_added={summary['deliver'].get('rows_added', 0)}")
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
            print(
                f"Generated inputs file not found at {gen_path.resolve()}; falling back to source: {in_path.resolve()}")
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