"""
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


def node_id(location: str, equipment: str) -> str:
    return f"{equipment}@{location}"


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

    # Compute swimlane order (one lane per Location), preferring locations that first appear
    # at lower Levels (higher rank). Tie-break by first-appearance index in the Location column,
    # then by first appearance anywhere (Location or Next Location).
    df_idx = df.reset_index().rename(columns={"index": "_row_idx"})
    df_idx["location"] = df_idx["location"].astype(str).str.strip()
    df_idx["next_location"] = df_idx["next_location"].astype(str).str.strip()
    # First-appearance in Location column
    first_loc_idx: Dict[str, int] = {}
    min_loc_level: Dict[str, int] = {}
    for i, r in df_idx.iterrows():
        loc = r["location"]
        try:
            lvl = int(r["level"]) if pd.notna(r["level"]) else 10**9
        except Exception:
            lvl = 10**9
        if loc and loc not in first_loc_idx:
            first_loc_idx[loc] = int(r["_row_idx"]) if "_row_idx" in r else i
        if loc:
            cur_min = min_loc_level.get(loc, 10**9)
            if lvl < cur_min:
                min_loc_level[loc] = lvl
    # First appearance anywhere (Location or Next Location)
    first_any_idx: Dict[str, int] = {}
    for i, r in df_idx.iterrows():
        for col in ("location", "next_location"):
            loc = r[col]
            if loc and loc not in first_any_idx:
                first_any_idx[loc] = int(r["_row_idx"]) if "_row_idx" in r else i
    # All locations observed
    all_locs_raw = set([str(x).strip() for x in df_idx["location"].tolist()]) | set([str(x).strip() for x in df_idx["next_location"].tolist()])
    all_locs = [loc for loc in all_locs_raw if loc]
    def _rank_tuple(loc: str) -> tuple:
        lvl = min_loc_level.get(loc, 10**9)
        first_loc = first_loc_idx.get(loc, 10**9)
        any_idx = first_any_idx.get(loc, 10**9)
        return (lvl, first_loc, any_idx, loc)
    ordered_locs = sorted(all_locs, key=_rank_tuple)
    loc_to_rank: Dict[str, int] = {loc: i for i, loc in enumerate(ordered_locs)}

    # Add nodes and infer levels for destinations
    for _, row in df.iterrows():
        src_level_val = int(row["level"]) if pd.notna(row["level"]) else 0
        src_node = node_id(row["location"], row["equipment_name"]) if row["equipment_name"] else None
        if src_node:
            G.add_node(
                src_node,
                title=f"Location: {row['location']}<br>Equipment: {row['equipment_name']}",
                level=src_level_val,
                location=row["location"],
                equipment=row["equipment_name"],
                loc_index=loc_to_rank.get(str(row["location"]).strip(), 0),
            )
        # destination node might be blank (terminal)
        if str(row["next_location"]).strip() and str(row["next_equipment"]).strip():
            dst_node = node_id(row["next_location"], row["next_equipment"])
            # infer destination level from source (place to the right)
            dst_level = src_level_val + 1
            if dst_node not in G:
                G.add_node(
                    dst_node,
                    title=f"Location: {row['next_location']}<br>Equipment: {row['next_equipment']}",
                    location=row["next_location"],
                    equipment=row["next_equipment"],
                    level=dst_level,
                    loc_index=loc_to_rank.get(str(row["next_location"]).strip(), 0),
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
        src = node_id(row["location"], row["equipment_name"]) if row["equipment_name"] else None
        if not src or src not in G:
            continue
        next_loc = str(row["next_location"]).strip()
        next_eq = str(row["next_equipment"]).strip()
        if not (next_loc and next_eq):
            # terminal step — no outgoing edge
            continue
        dst = node_id(next_loc, next_eq)
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

    if use_fixed_grid and not enable_physics:
        # Disable physics and hierarchical engine; we will pin nodes at grid coordinates.
        net.toggle_physics(False)
        net.set_options(
            '{"interaction": {"navigationButtons": true, "keyboard": true, "zoomView": true, "dragView": true, "dragNodes": false, "multiselect": true}, "edges": {"smooth": false, "arrows": {"to": {"enabled": true, "scaleFactor": 0.7}}, "font": {"size": 10, "align": "middle", "vadjust": 0, "strokeWidth": 3, "strokeColor": "#ffffff", "background": "rgba(255,255,255,0.6)"}}, "nodes": {"font": {"size": 0}, "shapeProperties": {"useImageSize": true, "useBorderWithImage": false}, "scaling": {"min": 1, "max": 1}}}'
        )
        # Pre-compute positions: rows by location index, columns by level (levels left->right within each location)
        positions: Dict[str, Tuple[int, int]] = {}
        for n, d in G.nodes(data=True):
            positions[n] = (int(d.get("level", 0)) * XSEP, int(d.get("loc_index", 0)) * YSEP)
        # De-overlap and reduce crossings within each (location, level) cell using a barycentric heuristic
        groups: Dict[Tuple[int, int], list[str]] = {}
        base_y_cache: Dict[str, int] = {}
        for n, d in G.nodes(data=True):
            key = (int(d.get("loc_index", 0)), int(d.get("level", 0)))
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
        for (loc_i, lvl), ns in groups.items():
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
                positions[n] = (int(d.get("level", 0)) * XSEP, int(d.get("loc_index", 0)) * YSEP)
            # Apply initial de-overlap within each (location, level) cell so nodes start spaced apart
            groups: Dict[Tuple[int, int], list[str]] = {}
            base_y_cache: Dict[str, int] = {}
            for n, d in G.nodes(data=True):
                key = (int(d.get("loc_index", 0)), int(d.get("level", 0)))
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
    def _add_text_sprite(node_id: str, text: str, x_center: int, y_center: int):
        try:
            import urllib.parse as _urlparse  # noqa
            def _esc(s: str) -> str:
                return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            font = int(getattr(config, 'move_label_font_size', 10))
            pad = int(getattr(config, 'move_label_pad', 6))
            color = getattr(config, 'move_label_text_color', '#333333')
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
        src = node_id(r["location"], r["equipment_name"]) if r["equipment_name"] else None
        if not src:
            continue
        res_caps.setdefault(src, 1)  # default capacity 1; to be overridden from Excel later
        next_loc = str(r["next_location"]).strip()
        next_eq = str(r["next_equipment"]).strip()
        if next_loc and next_eq:
            dst = node_id(next_loc, next_eq)
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
        start_node = node_id(start_row["location"], start_row["equipment_name"]) if start_row["equipment_name"] else list(resources.keys())[0]
        env.process(entity("demo", start_node))
        return env, resources, routes

    return build_env


# ------------------------ Input preparation (Excel) -------------------------
MAKE_REQUIRED_INPUT_COLS = [
    "Mean Production Rate (Tons/hr)",
    "Std Dev of Production Rate (Tons/Hr)",
    "Planned Maintenance Dates (Days of year)",
    "Unplanned downtime %",
]
MAKE_KEY_COLS = ["Location", "Equipment Name", "Input"]
MAKE_SHEET_COLS = MAKE_KEY_COLS + MAKE_REQUIRED_INPUT_COLS


def _normalize_key_triplet(df_like: pd.DataFrame, loc_col: str, eq_col: str, in_col: str) -> pd.Series:
    def _norm(s: pd.Series) -> pd.Series:
        return s.astype(str).fillna("").str.strip().str.upper()
    return _norm(df_like[loc_col]) + "|" + _norm(df_like[eq_col]) + "|" + _norm(df_like[in_col])


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

    Missing required input fields will be filled with default placeholders.
    Returns a summary dict with counts of rows added and fields filled.
    """
    # Read Network
    net = _read_network_df(xlsx_path)
    if net.empty:
        raise ValueError("Network sheet is empty or missing required columns.")
    make_rows = net[net["process"].str.upper() == "MAKE"].copy()
    if make_rows.empty:
        # Nothing to do
        existing = _read_sheet_df(xlsx_path, "Make")
        if existing is None:
            # Create an empty Make sheet with headers
            empty_df = pd.DataFrame(columns=MAKE_SHEET_COLS)
            _write_sheet_df(xlsx_path, "Make", empty_df)
            return {"rows_added": 0, "fields_filled": 0}
        else:
            return {"rows_added": 0, "fields_filled": 0}

    uniq = (
        make_rows[["location", "equipment_name", "input"]]
        .fillna("")
        .drop_duplicates()
        .rename(columns={"location": "Location", "equipment_name": "Equipment Name", "input": "Input"})
        .reset_index(drop=True)
    )
    uniq["KEY"] = _normalize_key_triplet(uniq, "Location", "Equipment Name", "Input")

    # Read existing Make sheet (if any)
    current = _read_sheet_df(xlsx_path, "Make")
    if current is None or current.empty:
        current = pd.DataFrame(columns=MAKE_SHEET_COLS)
    # Ensure required columns present
    current = _ensure_columns(current, MAKE_SHEET_COLS)
    # Build normalized key in current
    if not current.empty:
        current["Location"] = current["Location"].astype(str)
        current["Equipment Name"] = current["Equipment Name"].astype(str)
        current["Input"] = current["Input"].astype(str)
        current_keys = _normalize_key_triplet(current, "Location", "Equipment Name", "Input")
        existing_key_set = set(current_keys.tolist())
    else:
        existing_key_set = set()

    rows_to_add = uniq[~uniq["KEY"].isin(existing_key_set)].copy()

    # Defaults
    defaults = {
        "Mean Production Rate (Tons/hr)": 0.0,
        "Std Dev of Production Rate (Tons/Hr)": 0.0,
        "Planned Maintenance Dates (Days of year)": "",
        "Unplanned downtime %": 0.0,
    }

    added_count = 0
    filled_count = 0

    if not rows_to_add.empty:
        # Create new rows with defaults
        for _, r in rows_to_add.iterrows():
            new_row = {"Location": r["Location"], "Equipment Name": r["Equipment Name"], "Input": r["Input"]}
            new_row.update(defaults)
            current = pd.concat([current, pd.DataFrame([new_row])], ignore_index=True)
            added_count += 1

    # Fill missing fields in existing rows
    for col, def_val in defaults.items():
        if col not in current.columns:
            current[col] = def_val
            filled_count += len(current)
        else:
            mask = current[col].isna() | (current[col].astype(str).str.strip() == "")
            n = int(mask.sum())
            if n > 0:
                current.loc[mask, col] = def_val
                filled_count += n

    # Order columns: required first, then any extras
    extra_cols = [c for c in current.columns if c not in MAKE_SHEET_COLS]
    ordered_cols = MAKE_SHEET_COLS + extra_cols
    # Sort for neatness by key triplet
    sort_cols = ["Location", "Equipment Name", "Input"]
    current = current[ordered_cols].sort_values(sort_cols).reset_index(drop=True)

    _write_sheet_df(xlsx_path, "Make", current)
    return {"rows_added": added_count, "fields_filled": filled_count}


def prepare_inputs_excel(xlsx_path: Path) -> dict:
    """High-level entry to prepare Excel inputs: Settings and Make sheets.
    Returns a combined summary.
    """
    if not xlsx_path.exists():
        raise FileNotFoundError(xlsx_path)
    # Ensure it's an Excel file
    if xlsx_path.suffix.lower() not in {".xlsx", ".xlsm", ".xls"}:
        raise ValueError("prepare_inputs_excel requires an Excel workbook (.xlsx/.xlsm/.xls)")

    settings_summary = ensure_settings_sheet(xlsx_path)
    make_summary = ensure_make_sheet(xlsx_path)
    return {"settings": settings_summary, "make": make_summary}

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
    """Load Config either from a provided path (.py) or from 'viz_config' in CWD.
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
    # Try import viz_config from CWD
    try:
        import importlib
        vc = importlib.import_module('viz_config')
        cfg = _load_config_from_module(vc)
        if cfg is not None:
            return cfg
    except Exception:
        pass
    # If not present, create a template
    template = Path('viz_config.py')
    if not template.exists():
        try:
            template.write_text(
                "# Auto-generated configuration template. Edit values and re-run.\n"
                "from dataclasses import dataclass\n\n"
                "@dataclass\n"
                "class Config:\n"
                "    in_path: str = 'Model Inputs.xlsx'\n"
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
    raise RuntimeError("No configuration found. A 'viz_config.py' template has been created in the current folder. Edit it and re-run.")


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    p = argparse.ArgumentParser(description="Supply chain visualization and SimPy scaffold (config-driven)")
    p.add_argument("--config", type=str, default=None, help="Path to a Python config file (viz_config.py). If omitted, the script loads viz_config from the current folder.")
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

    # Optional: prepare inputs and exit
    if bool(getattr(config, 'prepare_inputs', False)):
        try:
            summary = prepare_inputs_excel(in_path)
            print("Input preparation completed:")
            print(f"  Settings: added={summary['settings'].get('added', 0)}, updated={summary['settings'].get('updated', 0)}")
            print(f"  Make: rows_added={summary['make'].get('rows_added', 0)}, fields_filled={summary['make'].get('fields_filled', 0)}")
        except Exception as e:
            print(f"Input preparation failed: {e}")
            return 1
        # If prepare_inputs only, stop here
        print("prepare_inputs=True was set in config; exiting after preparation.")
        return 0

    # Build graph
    df = read_table(in_path, sheet=getattr(config, 'sheet', None))
    G = build_graph(df, product_class=getattr(config, 'product_class', None))
    try:
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
    except Exception:
        n_nodes = n_edges = -1
    print(f"Graph built: {n_nodes} nodes, {n_edges} edges.")
    if n_nodes == 0 or n_edges == 0:
        print("Warning: The graph appears empty. Check input and product_class filter in viz_config.py.")

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

