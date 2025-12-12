"""
Generate a stand‑alone HTML animation of the supply chain network that matches the layout
of supply_chain_viz.py and replays inventory + transporter events from CSV logs.

What it does
- Builds the network from your workbook using the same grid layout as supply_chain_viz.py
  (fixed swimlanes: rows by Location, columns by Product Class panels; uses Config spacing).
- Reads inventory snapshots (sim_outputs/inventory_daily.csv) to update tank fills over time.
- Reads the detailed action log (sim_outputs/sim_log.csv) and reconstructs vehicle trips
  from each Loaded→Unloaded pair to animate moving dots along edges with exact event timing.
- Exports a single HTML file (my_supply_chain_live.html) with embedded JSON data and
  a vis-network canvas + lightweight DOM overlay for vehicle dots and HUD controls.

How to use
  python supply_chain_live_viz.py

Requirements
- pandas, jinja2 (already listed in requirements.txt)
- No web server required. The output HTML references the local lib/vis-9.1.2 assets in this repo.

Notes
- This first version expects you to run sim_run.py with write_daily_snapshots=True and write_log=True.
  That produces the CSVs we consume:
    sim_outputs/inventory_daily.csv  # per store_key per snapshot hour
    sim_outputs/sim_log.csv          # events with time_h, Loaded/Unloaded, src/dst, qty
- Vehicle reconstruction: loads are paired to the next matching unload with same product and
  same src→dst (by location+equipment). If exact counts differ, excess events are ignored.
- Layout fidelity: we compute the same coordinates as export_pyvis() uses when physics=False.
  If you change spacing in supply_chain_viz_config.Config, rerun this script to regenerate.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import webbrowser

import supply_chain_viz as scv
from supply_chain_viz_config import Config as VizConfig

# Static library paths used by the HTML (relative to project root)
VIS_CSS = "lib/vis-9.1.2/vis-network.css"
VIS_JS = "lib/vis-9.1.2/vis-network.min.js"

ROOT = Path(__file__).parent
SIM_OUT = ROOT / "sim_outputs"
OUT_HTML = ROOT / "my_supply_chain_live.html"


@dataclass
class NodePos:
    id: str
    x: int
    y: int
    product_class: str
    location: str
    level: int


@dataclass
class EdgeDef:
    id: str
    src: str
    dst: str
    product_class: str


@dataclass
class Trip:
    edge_id: str
    t_load: float   # start of loading (h)
    t_depart: float # end loading; begin travel (h)
    t_arrive: float # arrival; begin unload (h)
    t_done: float   # end unload (h)
    qty: float


def _compute_positions(G, config) -> Dict[str, NodePos]:
    """Recompute the exact fixed positions used in export_pyvis when physics=False."""
    XSEP = int(getattr(config, 'grid_x_sep', 120))
    YSEP = int(getattr(config, 'grid_y_sep', 160))
    PCSEP = int(getattr(config, 'grid_pc_sep', 560))
    STAG = int(getattr(config, 'cell_stack_sep', 140))

    # Product-class columns present (and order)
    pcs_present: List[str] = []
    for _n, _d in G.nodes(data=True):
        pcv = str(_d.get('product_class', '')).strip()
        if pcv and pcv not in pcs_present:
            pcs_present.append(pcv)
    pc_order_cfg = getattr(config, 'pc_order', None)
    if isinstance(pc_order_cfg, (list, tuple)):
        pc_order = [str(x).strip() for x in pc_order_cfg if str(x).strip() in pcs_present]
        for pc in pcs_present:
            if pc not in pc_order:
                pc_order.append(pc)
    else:
        pc_order = sorted(pcs_present)
    pc_to_col: Dict[str, int] = {pc: i for i, pc in enumerate(pc_order)}

    # Base positions
    positions: Dict[str, Tuple[int, int]] = {}
    for n, d in G.nodes(data=True):
        pcv = str(d.get("product_class", "")).strip()
        pc_col = pc_to_col.get(pcv, 0)
        x = pc_col * PCSEP + int(d.get("level", 0)) * XSEP
        y = int(d.get("loc_index", 0)) * YSEP
        positions[n] = (x, y)

    # Group by (loc_index, pc_col, level) and apply vertical stagger to de-overlap
    from collections import defaultdict
    groups: Dict[Tuple[int, int, int], List[str]] = defaultdict(list)
    base_y_cache: Dict[str, int] = {}
    for n, d in G.nodes(data=True):
        pcv = str(d.get("product_class", "")).strip()
        pc_col = pc_to_col.get(pcv, 0)
        key = (int(d.get("loc_index", 0)), pc_col, int(d.get("level", 0)))
        groups[key].append(n)
        base_y_cache[n] = int(d.get("loc_index", 0)) * YSEP

    def _avg_neighbor_y(nodes_list: List[str], lvl: int) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for nn in nodes_list:
            ys: List[int] = []
            # prefer neighbors at next level, then previous level
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

    for (loc_i, pc_col, lvl), ns in groups.items():
        if len(ns) > 1:
            scores = _avg_neighbor_y(ns, lvl)
            ns_sorted = sorted(ns, key=lambda n: (scores.get(n, 0), n))
            k = len(ns_sorted)
            for i, nn in enumerate(ns_sorted):
                x, y = positions[nn]
                y_off = (i - (k - 1) / 2) * STAG
                positions[nn] = (x, y + int(y_off))

    nodepos: Dict[str, NodePos] = {}
    for n, d in G.nodes(data=True):
        x, y = positions[n]
        nodepos[n] = NodePos(
            id=n,
            x=int(x),
            y=int(y),
            product_class=str(d.get("product_class", "")),
            location=str(d.get("location", "")),
            level=int(d.get("level", 0)),
        )
    return nodepos


def _build_edges(G) -> List[EdgeDef]:
    edges: List[EdgeDef] = []
    # Use a deterministic synthetic id for each unique src→dst across multi-edges
    seen = set()
    for u, v, edata in G.edges(data=True):
        key = (u, v)
        if key in seen:
            continue
        seen.add(key)
        edges.append(EdgeDef(id=f"{u}__TO__{v}", src=u, dst=v, product_class=str(edata.get("product_class", ""))))
    return edges


def _read_inventory_snapshots(path: Path) -> pd.DataFrame:
    """Return DataFrame with columns: time_h, store_key, level, capacity."""
    if not path.exists():
        # As a fallback, try store_levels.csv (single snapshot only)
        alt = SIM_OUT / "store_levels.csv"
        if not alt.exists():
            raise FileNotFoundError(
                "Inventory snapshots not found. Run sim_run.py with write_daily_snapshots=True, or provide store_levels.csv."
            )
        df = pd.read_csv(alt)
        df["time_h"] = 0.0
        df.rename(columns={"Store": "store_key", "Level": "level"}, inplace=True)
        df["capacity"] = None
        return df[["time_h", "store_key", "level", "capacity"]]
    df = pd.read_csv(path)
    # keep necessary columns
    cols = ["time_h", "store_key", "level", "capacity"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"inventory_daily.csv missing columns: {missing}")
    return df[cols].copy()


def _read_sim_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError("sim_log.csv not found. Ensure sim_config.write_log=True and rerun sim_run.py")
    df = pd.read_csv(path)
    # Expected columns include: time_h, event, product, src_location, src_equipment, dst_location, dst_equipment, qty_t
    need = ["time_h", "event", "product", "src_location", "src_equipment", "dst_location", "dst_equipment", "qty_t"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"sim_log.csv missing required column: {c}")
    return df


def _store_key_from_row(row) -> str:
    # Construct the same store_key shape as in the frozen model: "{PC}|{LOCATION}|{EQUIPMENT}|{INPUT}".
    # The log may not contain 'input'; we can match nodes by a triple (pc, location, equipment) for animation edges.
    pc = str(row.get("product", "")).upper()
    loc = str(row.get("location", ""))
    eq = str(row.get("equipment", ""))
    return f"{pc}|{loc}|{eq}"


def _node_lookup_triplet(G) -> Dict[Tuple[str, str, str], str]:
    """Map (product_class, location, equipment) -> node_id used in the graph."""
    m: Dict[Tuple[str, str, str], str] = {}
    for n, d in G.nodes(data=True):
        pc = str(d.get("product_class", ""))
        loc = str(d.get("location", ""))
        eq = str(d.get("equipment", ""))
        m[(pc, loc, eq)] = n
    return m


def _pair_trips(log_df: pd.DataFrame, node_id_by_triplet: Dict[Tuple[str, str, str], str]) -> List[Trip]:
    """Reconstruct trips by pairing loaded events to the next matching unloaded event for the same src→dst and product.
    We use a simple FIFO queue per (pc, src_loc, src_eq, dst_loc, dst_eq).
    """
    # Normalize and filter
    df = log_df.copy()
    df["product"] = df["product"].astype(str).str.upper()
    df.sort_values(["time_h"], inplace=True)

    from collections import defaultdict, deque
    loads: Dict[Tuple[str, str, str, str, str], deque] = defaultdict(deque)
    trips: List[Trip] = []

    for _, row in df.iterrows():
        ev = str(row.get("event", ""))
        pc = str(row.get("product", ""))
        src_loc = str(row.get("src_location", ""))
        src_eq = str(row.get("src_equipment", ""))
        dst_loc = str(row.get("dst_location", ""))
        dst_eq = str(row.get("dst_equipment", ""))
        qty = float(row.get("qty_t", 0.0) or 0.0)
        t = float(row.get("time_h", 0.0) or 0.0)

        if ev == "Loaded":
            loads[(pc, src_loc, src_eq, dst_loc, dst_eq)].append((t, qty))
        elif ev == "Unloaded":
            key = (pc, src_loc, src_eq, dst_loc, dst_eq)
            if not loads[key]:
                # try reversed src/dst; some logs may flip columns
                key2 = (pc, dst_loc, dst_eq, src_loc, src_eq)
                if loads[key2]:
                    t_load, q = loads[key2].popleft()
                    src_loc, src_eq, dst_loc, dst_eq = dst_loc, dst_eq, src_loc, src_eq
                else:
                    continue
            else:
                t_load, q = loads[key].popleft()
            # Resolve node ids
            src_id = node_id_by_triplet.get((pc, src_loc, src_eq))
            dst_id = node_id_by_triplet.get((pc, dst_loc, dst_eq))
            if not src_id or not dst_id:
                continue
            edge_id = f"{src_id}__TO__{dst_id}"
            # We don’t know exact loading/unloading durations from the log; estimate as 0.5h if timestamps equal
            t_depart = max(t_load, t_load + 0.5)
            t_arrive = max(t, t_depart)  # arrival at unload timestamp
            t_done = t_arrive + 0.5
            trips.append(Trip(edge_id=edge_id, t_load=t_load, t_depart=t_depart, t_arrive=t_arrive, t_done=t_done, qty=q))

    return trips


def _levels_series(inv_df: pd.DataFrame) -> Tuple[List[float], Dict[str, List[float]], Dict[str, float]]:
    """Return timeline hours, per-store level arrays, and per-store capacity (if provided)."""
    inv_df = inv_df.copy()
    inv_df["time_h"] = inv_df["time_h"].astype(float)
    times = sorted(inv_df["time_h"].unique().tolist())
    by_store: Dict[str, List[float]] = {}
    cap: Dict[str, float] = {}
    for sk, grp in inv_df.groupby("store_key"):
        levels = [None] * len(times)
        m = {float(r["time_h"]): float(r["level"]) for _, r in grp.iterrows()}
        for i, t in enumerate(times):
            levels[i] = m.get(float(t), None)
        by_store[sk] = levels
        # capacity optional
        cvals = grp["capacity"].dropna().astype(float).tolist()
        if cvals:
            cap[sk] = cvals[-1]
    return times, by_store, cap


def _render_html(out_path: Path, payload: dict) -> None:
    """Render an HTML file with embedded JSON and animation JS using a Jinja2 template.

    Using Jinja2 with raw blocks prevents Python from interpreting JavaScript braces.
    """
    # Prepare variables for the template
    json_data = json.dumps(payload, separators=(",", ":"))
    height = payload.get("height", "85vh")
    max_frames = max(0, len(payload.get("timeline", {}).get("t", [])))

    try:
        from jinja2 import Template  # type: ignore
    except Exception:
        Template = None

    if Template is not None:
        tpl = Template(
            """
<!DOCTYPE html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Supply Chain — Live HTML Animation</title>
  <link rel=\"stylesheet\" href=\"{{ vis_css }}\" />
  <style>
  {% raw %}
    html, body { height: 100%; margin: 0; }
    #container { position: relative; border: 1px solid #ddd; }
    #network { position: relative; width: 100%; height: 100%; z-index: 1; }
    #hud { position: absolute; right: 12px; bottom: 10px; z-index: 5; background: rgba(255,255,255,0.85); padding: 6px 8px; border-radius: 6px; font-family: Inter, Arial, sans-serif; font-size: 12px; color: #333; }
    #controls { position: absolute; left: 12px; bottom: 10px; z-index: 5; background: rgba(255,255,255,0.9); padding: 6px 8px; border-radius: 6px; font-family: Inter, Arial, sans-serif; font-size: 12px; color: #333; display: flex; gap: 8px; align-items: center; }
    #filters { position: absolute; left: 12px; top: 10px; z-index: 5; background: rgba(255,255,255,0.9); padding: 6px 8px; border-radius: 6px; font-family: Inter, Arial, sans-serif; font-size: 12px; color: #333; display: flex; gap: 8px; align-items: center; }
    #overlay { position: absolute; left: 0; top: 0; right: 0; bottom: 0; pointer-events: none; z-index: 4; }
    .vehicle { position: absolute; width: 10px; height: 10px; border-radius: 50%; border: 1px solid #fff; box-shadow: 0 0 2px rgba(0,0,0,0.5); }
    .rail { background: #ff3333; }
    .ship { background: #00c3ff; }
    .tank { position: absolute; width: 24px; height: 60px; border: 1px solid #444; background: #eee; border-radius: 3px; overflow: hidden; transform: translate(-50%, -50%); z-index: 6; }
    .fill { position: absolute; left: 1px; right: 1px; bottom: 1px; height: 0; background: #ffb347; }
    .label { position: absolute; top: -28px; left: 50%; transform: translateX(-50%); font-size: 10px; color: #222; text-align: center; white-space: nowrap; }
  {% endraw %}
  </style>
</head>
<body>
  <div id=\"container\" style=\"height: {{ height }};\">
    <div id=\"filters\"> 
      <label>Product Class: <select id=\"pcFilter\"><option value=\"\">All</option></select></label>
      <label>Location: <select id=\"locFilter\"><option value=\"\">All</option></select></label>
    </div>
    <div id=\"controls\">
      <button id=\"playPause\">Play</button>
      <label>Speed <input id=\"speed\" type=\"range\" min=\"0.25\" max=\"8\" step=\"0.25\" value=\"1\"/></label>
      <label>Time <input id=\"scrub\" type=\"range\" min=\"0\" max=\"{{ max_frames }}\" step=\"1\" value=\"0\"/></label>
    </div>
    <div id=\"hud\">t=0 h</div>
    <div id=\"network\"></div>
    <div id=\"overlay\"></div>
  </div>
  <script>const DATA = {{ json_data | safe }};</script>
  <script src=\"{{ vis_js }}\"></script>
  <script>
  {% raw %}
  (function(){
    const container = document.getElementById('network');
    const overlay = document.getElementById('overlay');
    const hud = document.getElementById('hud');
    const playBtn = document.getElementById('playPause');
    const speedEl = document.getElementById('speed');
    const scrub = document.getElementById('scrub');
    const pcFilter = document.getElementById('pcFilter');
    const locFilter = document.getElementById('locFilter');

    const nodes = new vis.DataSet(DATA.nodes.map(n => ({id: n.id, label: '', title: n.labelCompact, x: n.x, y: n.y, fixed: true, physics: false, shape: 'dot', size: 1})));
    const edges = new vis.DataSet(DATA.edges.map(e => ({id: e.id, from: e.src, to: e.dst, arrows: 'to', smooth: false, color: {"color": "#888"} })));
    const network = new vis.Network(container, { nodes, edges }, { physics: false, interaction: { dragNodes: false } });

    // Edge geometry cache (must exist before any computeEdgeGeom calls)
    var edgeGeom = new Map();

    // Position tank overlays over node positions (DOM coordinates)
    function placeTanks(){
      DATA.nodes.forEach(n => {
        const pos = network.getPositions([n.id])[n.id];
        if (!pos) return;
        const p = network.canvasToDOM({x: pos.x, y: pos.y});
        const t = tankByNode.get(n.id);
        if (!t) return;
        t.root.style.left = p.x + 'px';
        t.root.style.top  = p.y + 'px';
      });
    }

    // Recompute edge geometry and tank positions after draw/zoom/drag/resize
    // Defer the initial draw to the next tick so all const declarations (e.g., `times`) are initialized.
    network.once('afterDrawing', () => { setTimeout(() => { computeEdgeGeom(); placeTanks(); drawLevels(0); drawVehicles(0); }, 0); });
    network.on('zoom', () => { computeEdgeGeom(); placeTanks(); });
    network.on('dragEnd', () => { computeEdgeGeom(); placeTanks(); });
    window.addEventListener('resize', () => { computeEdgeGeom(); placeTanks(); });

    // Build overlays (tanks + labels) so we can animate fills without re-rendering nodes
    const tankByNode = new Map();
    function createTank(n) {
      const div = document.createElement('div');
      div.className = 'tank';
      div.style.width = '26px';
      div.style.height = '70px';
      const fill = document.createElement('div');
      fill.className = 'fill';
      const label = document.createElement('div');
      label.className = 'label';
      label.textContent = n.labelCompact;
      div.appendChild(fill); div.appendChild(label);
      overlay.appendChild(div);
      tankByNode.set(n.id, {root: div, fill, label, pc: n.product_class, loc: n.location});
    }
    DATA.nodes.forEach(n => createTank(n));

    // Vehicle pool
    const vehiclePool = new Map();
    function vehicleEl(edgeId, idx, cls) {
      const key = edgeId + '::' + idx;
      let el = vehiclePool.get(key);
      if (!el) {
        el = document.createElement('div');
        el.className = 'vehicle ' + cls;
        overlay.appendChild(el);
        vehiclePool.set(key, el);
      }
      return el;
    }

    // Filters
    const pcs = Array.from(new Set(DATA.nodes.map(n => n.product_class))).sort();
    pcs.forEach(pc => { const o = document.createElement('option'); o.value = pc; o.textContent = pc; pcFilter.appendChild(o); });
    const locs = Array.from(new Set(DATA.nodes.map(n => n.location))).sort();
    locs.forEach(loc => { const o = document.createElement('option'); o.value = loc; o.textContent = loc; locFilter.appendChild(o); });

    function applyFilters() {
      const pcv = pcFilter.value; const locv = locFilter.value;
      DATA.nodes.forEach(n => {
        const show = (!pcv || n.product_class === pcv) && (!locv || n.location === locv);
        const t = tankByNode.get(n.id);
        if (t) t.root.style.display = show ? 'block' : 'none';
      });
      // Edges and trips visibility
      const nodeVisible = new Set(DATA.nodes.filter(n => {
        return (!pcv || n.product_class === pcv) && (!locv || n.location === locv);
      }).map(n => n.id));
      DATA.edges.forEach(e => {
        const vis = nodeVisible.has(e.src) && nodeVisible.has(e.dst);
        const ed = edges.get(e.id);
        if (ed) edges.update({id: e.id, hidden: !vis});
      });
    }
    pcFilter.onchange = applyFilters;
    locFilter.onchange = applyFilters;
    applyFilters();

    function toPx(p) { const pos = network.canvasToDOM({x: p.x, y: p.y}); return {x: pos.x, y: pos.y}; }
    // edgeGeom declared above
    function computeEdgeGeom(){
      edgeGeom.clear();
      DATA.edges.forEach(e => {
        const from = network.getPositions([e.src])[e.src];
        const to = network.getPositions([e.dst])[e.dst];
        if (!from || !to) return;
        const a = toPx(from); const b = toPx(to); const dx=b.x-a.x, dy=b.y-a.y; const dist=Math.hypot(dx,dy)||1; const nx=dx/dist, ny=dy/dist; const px=-ny, py=nx;
        edgeGeom.set(e.id, {a, b, nx, ny, px, py});
      });
    }

    function setTankLevel(nid, pct){
      const t=tankByNode.get(nid); if (!t) return;
      pct=Math.max(0,Math.min(1,pct)); const h=68;
      t.fill.style.height = (h*pct)+'px';
      t.fill.style.background = pct<0.1 ? '#ff4d4d' : (pct>0.9 ? '#44c767' : '#ffb347');
    }

    const times = DATA.timeline.t; const storeLevels = DATA.timeline.levels; const storeCaps = DATA.timeline.capacity || {};
    // Build per-node series by summing all mapped store series
    const nodeSeries = new Map();
    const nodeCaps = new Map();
    (function buildNodeSeries(){
      const nFrames = times.length;
      for (const [sk, series] of Object.entries(storeLevels)){
        const nid = DATA.storeToNode[sk]; if (!nid) continue;
        let arr = nodeSeries.get(nid);
        if (!arr) { arr = new Array(nFrames).fill(0); nodeSeries.set(nid, arr); }
        for (let i=0;i<nFrames;i++){
          const v = series[i]; arr[i] = (arr[i]||0) + (v==null?0: v);
        }
        const c = storeCaps[sk];
        if (c!=null){ nodeCaps.set(nid, (nodeCaps.get(nid)||0) + c); }
      }
      // Fallback: if a node has no series, create zeros
      DATA.nodes.forEach(n => { if (!nodeSeries.has(n.id)) nodeSeries.set(n.id, new Array(times.length).fill(0)); });
    })();

    function drawLevels(frame){
      const ts = times[frame]; if (ts === undefined) return;
      for (const [nid, series] of nodeSeries.entries()){
        const v = series[frame];
        const cap = nodeCaps.get(nid) || (DATA.nodeCap && DATA.nodeCap[nid]) || Math.max(1, v||0);
        const pct = v!=null ? (v/cap) : 0;
        setTankLevel(nid, pct);
      }
      hud.textContent = `t=${Math.round(ts)} h (day ${(ts/24).toFixed(1)})`;
    }

    let playing=false; let speed=1.0; let frame=0; let lastTs=null;
    playBtn.onclick = function(){ playing = !playing; playBtn.textContent = playing ? 'Pause' : 'Play'; if (playing) requestAnimationFrame(tick); };
    speedEl.oninput = function(){ speed = parseFloat(speedEl.value)||1.0; };
    scrub.oninput = function(){ frame = parseInt(scrub.value)||0; drawLevels(frame); drawVehicles(frame); };

    function tick(ts){
      if (!playing) return;
      if (lastTs==null) lastTs=ts;
      const dt=(ts-lastTs)/1000; lastTs=ts;
      const step = Math.max(1, Math.floor(speed*dt*30));
      frame = Math.min(times.length-1, frame + step);
      scrub.value = String(frame);
      drawLevels(frame);
      drawVehicles(frame);
      if (frame < times.length-1) requestAnimationFrame(tick); else playing=false;
    }

    // Trips rendering (event-paired)
    const laneCache = new Map();
    function laneOffset(edgeId, idx){ let lane = laneCache.get(edgeId+':'+idx); if (lane==null){ lane = (idx%5)-2; laneCache.set(edgeId+':'+idx, lane);} return lane; }

    function drawVehicles(frame){
      const ts = times[frame]; if (ts===undefined) return;
      DATA.trips.forEach((tr, idx) => {
        const g = edgeGeom.get(tr.edge_id); if (!g) return;
        let x=g.a.x, y=g.a.y;
        if (ts < tr.t_load){ x=g.a.x; y=g.a.y; }
        else if (ts < tr.t_depart){ x=g.a.x + g.nx*8; y=g.a.y + g.ny*8; }
        else if (ts < tr.t_arrive){ const prog = (ts - tr.t_depart)/Math.max(1e-9, (tr.t_arrive - tr.t_depart)); x = g.a.x + (g.b.x - g.a.x)*prog; y = g.a.y + (g.b.y - g.a.y)*prog; }
        else if (ts < tr.t_done){ x=g.b.x - g.nx*8; y=g.b.y - g.ny*8; }
        else { x=g.a.x; y=g.a.y; }
        const lane = laneOffset(tr.edge_id, idx); x += g.px*lane*6; y += g.py*lane*6;
        const cls = tr.mode === 'rail' ? 'rail' : 'ship';
        const el = vehicleEl(tr.edge_id, idx, cls); el.style.transform = `translate(${x-5}px, ${y-5}px)`;
      });
    }

    drawLevels(0); drawVehicles(0);
  })();
  {% endraw %}
  </script>
</body>
</html>
            """
        )
        html = tpl.render(
            vis_css=VIS_CSS,
            vis_js=VIS_JS,
            height=height,
            max_frames=max_frames,
            json_data=json_data,
        )
    else:
        # Fallback: concatenate literal strings to avoid brace interpolation
        html = (
            "<!DOCTYPE html>\n"
            "<html>\n<head>\n<meta charset=\"utf-8\" />\n<title>Supply Chain — Live HTML Animation</title>\n"
            f"<link rel=\"stylesheet\" href=\"{VIS_CSS}\" />\n"
            "<style>\nhtml, body { height: 100%; margin: 0; }\n#container { position: relative; border: 1px solid #ddd; }\n#network { width: 100%; height: 100%; }\n#hud { position: absolute; right: 12px; bottom: 10px; z-index: 5; background: rgba(255,255,255,0.85); padding: 6px 8px; border-radius: 6px; font-family: Inter, Arial, sans-serif; font-size: 12px; color: #333; }\n#controls { position: absolute; left: 12px; bottom: 10px; z-index: 5; background: rgba(255,255,255,0.9); padding: 6px 8px; border-radius: 6px; font-family: Inter, Arial, sans-serif; font-size: 12px; color: #333; display: flex; gap: 8px; align-items: center; }\n#filters { position: absolute; left: 12px; top: 10px; z-index: 5; background: rgba(255,255,255,0.9); padding: 6px 8px; border-radius: 6px; font-family: Inter, Arial, sans-serif; font-size: 12px; color: #333; display: flex; gap: 8px; align-items: center; }\n#overlay { position: absolute; left: 0; top: 0; right: 0; bottom: 0; pointer-events: none; }\n.vehicle { position: absolute; width: 10px; height: 10px; border-radius: 50%; border: 1px solid #fff; box-shadow: 0 0 2px rgba(0,0,0,0.5); }\n.rail { background: #ff3333; }\n.ship { background: #00c3ff; }\n.tank { position: absolute; width: 24px; height: 60px; border: 1px solid #444; background: #eee; border-radius: 3px; overflow: hidden; transform: translate(-50%, -50%); }\n.fill { position: absolute; left: 1px; right: 1px; bottom: 1px; height: 0; background: #ffb347; }\n.label { position: absolute; top: -28px; left: 50%; transform: translateX(-50%); font-size: 10px; color: #222; text-align: center; white-space: nowrap; }\n</style>\n"
            "</head>\n<body>\n"
            f"<div id=\"container\" style=\"height: {height};\">\n"
            "<div id=\"filters\">\n<label>Product Class: <select id=\"pcFilter\"><option value=\"\">All</option></select></label>\n<label>Location: <select id=\"locFilter\"><option value=\"\">All</option></select></label>\n</div>\n"
            "<div id=\"controls\">\n<button id=\"playPause\">Play</button>\n<label>Speed <input id=\"speed\" type=\"range\" min=\"0.25\" max=\"8\" step=\"0.25\" value=\"1\"/></label>\n"
            f"<label>Time <input id=\"scrub\" type=\"range\" min=\"0\" max=\"{max_frames}\" step=\"1\" value=\"0\"/></label>\n</div>\n"
            "<div id=\"hud\">t=0 h</div>\n<div id=\"network\"></div>\n<div id=\"overlay\"></div>\n</div>\n"
            f"<script>const DATA = {json_data};</script>\n"
            f"<script src=\"{VIS_JS}\"></script>\n"
            "<script>\n// Minimal fallback JS omitted for brevity in this branch.\n</script>\n"
            "</body>\n</html>\n"
        )

    out_path.write_text(html, encoding="utf-8")


def main():
    # Load visualization config (so layout matches your existing HTML)
    cfg = VizConfig()

    # Build graph from workbook using the same helpers; honors product_class filter
    df = scv.read_table(cfg.resolve_in_path(), sheet=getattr(cfg, 'sheet', None))
    G = scv.build_graph(df, product_class=getattr(cfg, 'product_class', None))

    # Compute fixed positions equivalent to export_pyvis physics=False branch
    nodepos = _compute_positions(G, cfg)
    edges = _build_edges(G)

    # Build mapping from frozen store keys (PC|Location|Equipment|Input) or triples -> node id
    # We use triple matching for logs and inventory (which provide store_key).
    triplet_to_node = _node_lookup_triplet(G)

    # Inventory snapshots
    inv_df = _read_inventory_snapshots(SIM_OUT / "inventory_daily.csv")
    times, level_series, capacity = _levels_series(inv_df)

    # Map store_key values in inventory to node ids; tolerate 3-part keys (PC|Location|Equipment)
    store_to_node: Dict[str, str] = {}
    for sk in level_series.keys():
        parts = sk.split('|')
        node_id = None
        if len(parts) >= 3:
            node_id = triplet_to_node.get((parts[0], parts[1], parts[2]))
        if node_id:
            store_to_node[sk] = node_id
        else:
            # best-effort: find any node at location with same pc
            for (pc, loc, eq), nid in triplet_to_node.items():
                if pc == parts[0] and loc == parts[1]:
                    store_to_node[sk] = nid
                    break

    # Vehicle trips from sim_log
    log_df = _read_sim_log(SIM_OUT / "sim_log.csv")
    trips_basic = _pair_trips(log_df, triplet_to_node)

    # Enrich trips with src/dst ids + infer mode by edge pc or equipment names in node id
    edge_map = {e.id: e for e in edges}
    trips_out = []
    for tr in trips_basic:
        e = edge_map.get(tr.edge_id)
        if not e:
            # try to parse
            try:
                src, dst = tr.edge_id.split("__TO__")
            except Exception:
                continue
        else:
            src = e.src; dst = e.dst
        # mode heuristic
        src_eq = G.nodes[src].get('equipment', '').lower()
        dst_eq = G.nodes[dst].get('equipment', '').lower()
        mode = 'ship' if ('ship' in src_eq or 'ship' in dst_eq or 'port' in src_eq or 'port' in dst_eq) else 'rail'
        trips_out.append({
            'edge_id': tr.edge_id,
            'src': src,
            'dst': dst,
            'mode': mode,
            't_load': tr.t_load,
            't_depart': tr.t_depart,
            't_arrive': tr.t_arrive,
            't_done': tr.t_done,
            'qty': tr.qty,
        })

    # Prepare node meta for HTML
    nodes_payload = []
    node_labels = {}
    node_cap = {}
    for nid, np in nodepos.items():
        label_compact = f"{G.nodes[nid].get('location','')}, {G.nodes[nid].get('equipment','')}\n{np.product_class}"
        nodes_payload.append({
            'id': nid,
            'x': np.x,
            'y': np.y,
            'product_class': np.product_class,
            'location': np.location,
            'labelCompact': label_compact,
        })
        # try to map a capacity from inventory store mapping if any
        node_labels[nid] = label_compact
        # no per-node capacity in graph; we’ll rely on per-store capacity, else leave unset
        # node_cap can be filled from capacity of a mapped store if unique
    # heuristic: if exactly one store_key maps to a node, use its capacity
    node_to_store = {}
    for sk, nid in store_to_node.items():
        node_to_store.setdefault(nid, []).append(sk)
    for nid, sks in node_to_store.items():
        if len(sks) == 1 and sks[0] in capacity:
            node_cap[nid] = capacity[sks[0]]

    edges_payload = [{'id': e.id, 'src': e.src, 'dst': e.dst, 'product_class': e.product_class} for e in edges]

    payload = {
        'height': getattr(VizConfig(), 'height', '85vh'),
        'nodes': nodes_payload,
        'edges': edges_payload,
        'storeToNode': store_to_node,
        'nodeLabels': node_labels,
        'nodeCap': node_cap,
        'timeline': {
            't': times,
            'levels': level_series,
            'capacity': capacity,
            'step_h': float(times[1]-times[0]) if len(times) > 1 else 24.0,
        },
        'trips': trips_out,
    }

    _render_html(OUT_HTML, payload)
    try:
        webbrowser.open(OUT_HTML.as_uri())
    except Exception:
        pass
    print(f"Wrote {OUT_HTML}")


if __name__ == '__main__':
    main()
