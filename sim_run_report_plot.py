# sim_run_report_plot.py
from typing import List, Tuple
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
import json
import re
from sim_run_config import config


def _extract_prev_data_from_html(html_path: Path) -> tuple:
    """Extract Level (t) data and runtime from existing HTML report."""
    if not html_path.exists():
        return {}, 30  # Default 30s
    
    try:
        html_content = html_path.read_text(encoding='utf-8')
        prev_data = {}
        prev_runtime = 30
        
        # Extract runtime from summary (e.g., "run time 13s")
        runtime_match = re.search(r'run time (\d+)s', html_content)
        if runtime_match:
            prev_runtime = int(runtime_match.group(1))
        
        # Find all inventory plot divs and their corresponding Plotly.newPlot calls
        pattern = r'<div id="(plot-\d+)" class="plot" data-category="inventory" data-product="([^"]*)" data-location="([^"]*)"[^>]*></div><script>Plotly\.newPlot\("[^"]+", (\{.*?\})\);</script>'
        
        for match in re.finditer(pattern, html_content):
            plot_id, product, location, fig_json = match.groups()
            key = f"{location}|{product}"
            
            try:
                fig_data = json.loads(fig_json)
                # Find Level (t) trace
                for trace in fig_data.get('data', []):
                    if trace.get('name') == 'Level (t)':
                        x_data = trace.get('x', [])
                        y_data = trace.get('y', [])
                        if x_data and y_data:
                            prev_data[key] = {'x': x_data, 'y': y_data}
                        break
            except json.JSONDecodeError:
                continue
        
        return prev_data, prev_runtime
    except Exception:
        return {}, 30


def _merge_days_to_intervals(days: list) -> list:
    """Merge consecutive days into (start, end) intervals to reduce vrect count."""
    if not days:
        return []
    sorted_days = sorted(days)
    intervals = []
    start = end = sorted_days[0]
    for d in sorted_days[1:]:
        if d == end + 1:
            end = d
        else:
            intervals.append((start, end))
            start = end = d
    intervals.append((start, end))
    return intervals


def plot_results(sim, out_dir: Path, routes: list | None = None, makes: list | None = None, graph_sequence: list | None = None, report_data: dict | None = None, elapsed_seconds: int = 0):
    if not config.write_plots: return
    
    import time
    t0 = time.time()
    
    makes = makes or []
    graph_sequence = graph_sequence or []  # List of (Location, Equipment, Process) tuples

    # Extract previous inventory data and runtime from existing HTML before overwriting
    html_path = out_dir / "sim_outputs_plots_all.html"
    prev_inventory, prev_runtime = _extract_prev_data_from_html(html_path)
    
    # Use current elapsed time for display, prev_runtime for countdown
    current_runtime = elapsed_seconds if elapsed_seconds > 0 else prev_runtime

    # Use precomputed data if available, otherwise compute from sim
    if report_data:
        df_inv = report_data.get("df_inv", pd.DataFrame())
        df_log = report_data.get("df_log", pd.DataFrame())
        flows = report_data.get("flows", {})
        downtime_by_equipment = report_data.get("downtime_by_equipment", {})
        equipment_to_stores = report_data.get("equipment_to_stores", {})
        store_to_equipment = report_data.get("store_to_equipment", {})
    else:
        # Legacy path: compute from sim directly
        if sim.inventory_snapshots:
            df_inv = pd.DataFrame(sim.inventory_snapshots)
            df_inv["time_h"] = pd.to_numeric(df_inv["time_h"], errors="coerce")
            df_inv = df_inv.dropna(subset=["time_h"])
            df_inv["day"] = pd.to_numeric(df_inv["day"], errors="coerce").fillna(0).astype(int)
            df_inv = df_inv.sort_values(["store_key", "day"])
            df_inv = df_inv.groupby(["store_key", "day"]).last().reset_index()
            df_inv["level"] = df_inv["level"].round(0).astype(int)
            try:
                rate_map = getattr(sim, 'demand_rate_map', {}) or {}
            except:
                rate_map = {}
            df_inv["demand_per_day"] = df_inv["store_key"].map(lambda k: float(rate_map.get(str(k), 0.0)) * 24.0).round(1)
        else:
            df_inv = pd.DataFrame()

        if sim.action_log:
            df_log = pd.DataFrame(sim.action_log)
            df_log["day"] = (pd.to_numeric(df_log["time_h"], errors="coerce") / 24.0).astype(int) + 1
            if "qty_t" not in df_log.columns and "qty" in df_log.columns:
                df_log["qty_t"] = pd.to_numeric(df_log["qty"], errors="coerce").fillna(0)
        else:
            df_log = pd.DataFrame()
        
        flows = {}
        downtime_by_equipment = {}
        equipment_to_stores = {}
        store_to_equipment = {}

    # 3. Transport Timelines (separate for Train and Ship)
    train_transport_fig = None
    ship_route_group_figs = []
    vessel_state_fig = None
    fleet_util_fig = None
    route_summary_fig = None
    manufacturing_figs = {}
    if not df_log.empty:
        t1 = time.time()
        train_transport_fig = _generate_transport_plot(df_log, equipment_type="Train")
        print(f"  [timing] Train transport: {time.time()-t1:.1f}s")
        t1 = time.time()
        ship_route_group_figs = _generate_ship_timeline_by_route_group(df_log)
        print(f"  [timing] Ship timelines: {time.time()-t1:.1f}s")
        t1 = time.time()
        vessel_state_fig = _generate_vessel_state_chart(df_log)
        print(f"  [timing] Vessel state: {time.time()-t1:.1f}s")
        t1 = time.time()
        fleet_util_fig = _generate_fleet_utilisation_chart(df_log)
        print(f"  [timing] Fleet util: {time.time()-t1:.1f}s")
        t1 = time.time()
        route_summary_fig = _generate_route_summary_chart(df_log)
        print(f"  [timing] Route summary: {time.time()-t1:.1f}s")
        t1 = time.time()
        manufacturing_figs = _generate_manufacturing_charts(df_log)
        print(f"  [timing] Manufacturing: {time.time()-t1:.1f}s")

    # 4. Store Figures
    t1 = time.time()
    store_figs = {}
    if not df_inv.empty:
        supplier_map = {}
        if routes:
            try:
                tmp = {}
                for r in routes:
                    dests = getattr(r, 'dest_stores', []) or []
                    origins = getattr(r, 'origin_stores', []) or []
                    for d in dests:
                        s = tmp.setdefault(str(d), set())
                        for o in origins: s.add(str(o))
                for d, s in tmp.items(): supplier_map[d] = sorted(list(s))
            except:
                pass

        for store_key in df_inv["store_key"].unique():
            data = df_inv[df_inv["store_key"] == store_key].copy()

            # Merge Flow Data
            if store_key in flows:
                flow_data = []
                for d in data["day"]:
                    row = flows[store_key].get(d, {})
                    flow_data.append(row)
                flow_df = pd.DataFrame(flow_data, index=data.index)
                data = pd.concat([data, flow_df], axis=1)

            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # === BACKGROUND TRACES (added first, drawn behind) ===
            
            # Production Output (bars behind everything else - very transparent)
            if "Production_in" in data.columns and data["Production_in"].sum() > 0:
                fig.add_trace(go.Bar(x=data["day"], y=data["Production_in"], name="Production (t)",
                                     marker=dict(color="rgba(255, 127, 14, 0.2)", line=dict(width=0)), 
                                     width=0.8, opacity=0.5), secondary_y=True)

            # Demand
            if data["demand_per_day"].sum() > 0:
                fig.add_trace(go.Scatter(x=data["day"], y=data["demand_per_day"], name="Demand (t/d)",
                                         line=dict(color="green", dash="dot", width=1)), secondary_y=True)

            # Consumption (material consumed from this store for production)
            if "Consumption_out" in data.columns and data["Consumption_out"].sum() > 0:
                fig.add_trace(go.Scatter(x=data["day"], y=data["Consumption_out"], name="Consumption (t)",
                                         line=dict(color="#d62728", dash="dashdot", width=1.5)), secondary_y=True)

            # Train In
            if "Train_in" in data.columns and data["Train_in"].sum() > 0:
                fig.add_trace(go.Scatter(x=data["day"], y=data["Train_in"], name="Rail In (t)", mode='markers',
                                         marker=dict(symbol='triangle-down', size=8, color='#6a3d9a')),
                              secondary_y=True)

            # Ship In
            if "Ship_in" in data.columns and data["Ship_in"].sum() > 0:
                fig.add_trace(go.Scatter(x=data["day"], y=data["Ship_in"], name="Ship In (t)", mode='markers',
                                         marker=dict(symbol='triangle-down', size=10, color='#1f78b4')),
                              secondary_y=True)

            # Train Out
            if "Train_out" in data.columns and data["Train_out"].sum() > 0:
                fig.add_trace(go.Scatter(x=data["day"], y=data["Train_out"], name="Rail Out (t)", mode='markers',
                                         marker=dict(symbol='triangle-up', size=8, color='#e377c2')), secondary_y=True)

            # === FOREGROUND TRACES (added last, drawn on top) ===
            
            # Previous Level (behind current level for comparison)
            parts = str(store_key).split("|")
            if len(parts) >= 4:
                loc = parts[1]
                product = parts[3]
                prev_key = f"{loc}|{product}"
                if prev_key in prev_inventory:
                    prev = prev_inventory[prev_key]
                    fig.add_trace(
                        go.Scatter(x=prev['x'], y=prev['y'], name="Prev Level (t)",
                                   line=dict(color="rgba(150,150,150,0.6)", width=1.5, dash="dot"),
                                   hovertemplate="Day %{x}: %{y:,.0f} tons<extra>Previous</extra>"),
                        secondary_y=False)
            
            # Level (on top so it's always visible - thicker line)
            fig.add_trace(
                go.Scatter(x=data["day"], y=data["level"], name="Level (t)", line=dict(color="blue", width=2.5)),
                secondary_y=False)

            # Ship Out (on top so markers are visible)
            if "Ship_out" in data.columns and data["Ship_out"].sum() > 0:
                fig.add_trace(go.Scatter(x=data["day"], y=data["Ship_out"], name="Ship Out (t)", mode='markers',
                                         marker=dict(symbol='triangle-up', size=10, color='#17becf')), secondary_y=False)

            # Downtime Markers - only show downtime from equipment that produces TO this store
            supplier_equipment = store_to_equipment.get(store_key, set())
            
            # Aggregate downtime from all supplier equipment for this store
            store_downtime = {}  # day -> {'Maintenance': hours, 'Breakdown': hours}
            for equip_key in supplier_equipment:
                if equip_key in downtime_by_equipment:
                    for day, events in downtime_by_equipment[equip_key].items():
                        if day not in store_downtime:
                            store_downtime[day] = {"Maintenance": 0, "Breakdown": 0}
                        store_downtime[day]["Maintenance"] += events.get("Maintenance", 0)
                        store_downtime[day]["Breakdown"] += events.get("Breakdown", 0)
            
            if store_downtime:
                # Maintenance days - merge consecutive days into intervals
                maint_days = [d for d in store_downtime if store_downtime[d].get("Maintenance", 0) > 0]
                maint_intervals = _merge_days_to_intervals(maint_days)
                for start, end in maint_intervals:
                    fig.add_vrect(
                        x0=start - 0.5, x1=end + 0.5,
                        fillcolor="rgba(255, 152, 0, 0.15)",
                        layer="below",
                        line_width=0,
                    )
                if maint_days:
                    fig.add_trace(go.Scatter(
                        x=[None], y=[None],
                        name="Maintenance",
                        mode='markers',
                        marker=dict(symbol='square', size=12, color='rgba(255, 152, 0, 0.4)'),
                        showlegend=True
                    ), secondary_y=False)
                
                # Breakdown days - merge consecutive days into intervals
                breakdown_days = [d for d in store_downtime if store_downtime[d].get("Breakdown", 0) >= 24]
                breakdown_intervals = _merge_days_to_intervals(breakdown_days)
                for start, end in breakdown_intervals:
                    fig.add_vrect(
                        x0=start - 0.5, x1=end + 0.5,
                        fillcolor="rgba(244, 67, 54, 0.15)",
                        layer="below",
                        line_width=0,
                    )
                if breakdown_days:
                    fig.add_trace(go.Scatter(
                        x=[None], y=[None],
                        name="Breakdown (Full Day)",
                        mode='markers',
                        marker=dict(symbol='square', size=12, color='rgba(244, 67, 54, 0.4)'),
                        showlegend=True
                    ), secondary_y=False)
                

            cap = data["capacity"].iloc[0]
            suppliers = supplier_map.get(store_key, [])
            supplier_txt = f"<br><sub>Supplied From: {'; '.join(suppliers)}</sub>" if suppliers else ""

            fig.update_layout(
                title=f"Inventory: {store_key}{supplier_txt}<br><sub>Capacity: {cap:,.0f} tons</sub>",
                xaxis_title="Day",
                yaxis_title="Inventory (Tons)",
                yaxis2_title="Flow / Demand (Tons)",
                template="plotly_white",
                height=450,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            if config.autoscale_default:
                fig.update_yaxes(range=[0, max(cap * 1.1, data["level"].max() * 1.1)], secondary_y=False)

            store_figs[store_key] = fig

    print(f"  [timing] Store figures: {time.time()-t1:.1f}s")
    
    # 5. Collect all products from inventory data for filters
    all_products = set()
    if not df_inv.empty:
        for sk in df_inv["store_key"].unique():
            parts = str(sk).split("|")
            if len(parts) >= 4:
                all_products.add(parts[3])

    # 6. HTML Generation
    t1 = time.time()
    content = []

    if train_transport_fig:
        content.append({"type": "fig", "fig": train_transport_fig, "title": "Rail Transport Timeline", "category": "shipping"})
    
    if fleet_util_fig:
        content.append({"type": "fig", "fig": fleet_util_fig, "title": "Ship Fleet Utilization", "category": "shipping"})
    
    if vessel_state_fig:
        content.append({"type": "fig", "fig": vessel_state_fig, "title": "Fleet State Over Time", "category": "shipping"})
    
    if route_summary_fig:
        content.append({"type": "fig", "fig": route_summary_fig, "title": "Route Summary", "category": "shipping"})
    
    for route_group, fig in ship_route_group_figs:
        content.append({"type": "fig", "fig": fig, "title": f"Ship Route: {route_group}", "category": "shipping"})

    # Group manufacturing by location
    mfg_by_location = {}
    for unit_key in manufacturing_figs.keys():
        loc = unit_key.split("|")[0] if "|" in unit_key else "Other"
        if loc not in mfg_by_location:
            mfg_by_location[loc] = []
        mfg_by_location[loc].append(unit_key)

    # Build a lookup to find store_figs by (location, store_name)
    store_by_loc_name = {}  # (location, store_name) -> [(store_key, product), ...]
    if not df_inv.empty:
        for sk in df_inv["store_key"].unique():
            parts = str(sk).split("|")
            if len(parts) >= 3:
                loc = parts[1]
                store_name = parts[2]
                product = parts[3] if len(parts) >= 4 else "Other"
                key = (loc, store_name)
                if key not in store_by_loc_name:
                    store_by_loc_name[key] = []
                if sk in store_figs:
                    store_by_loc_name[key].append((sk, product))
    
    # Build lookup for manufacturing figs by (location, equipment)
    mfg_by_loc_equip = {}  # (location, equipment) -> unit_key
    for unit_key in manufacturing_figs.keys():
        parts = unit_key.split("|")
        if len(parts) >= 2:
            loc, equip = parts[0], parts[1]
            mfg_by_loc_equip[(loc, equip)] = unit_key
    
    # Group graph_sequence items by location (preserving order within each location)
    location_order = []  # List of unique locations in order of first appearance
    items_by_location = {}  # location -> list of (equip, proc) in sequence order
    for (loc, equip, proc) in graph_sequence:
        if loc not in items_by_location:
            location_order.append(loc)
            items_by_location[loc] = []
        items_by_location[loc].append((equip, proc))
    
    # Track what we've added
    added_stores = set()
    added_mfg = set()
    
    # Add graphs grouped by location, with all products under each location
    for loc in location_order:
        content.append({"type": "header", "location": loc, "category": "inventory"})
        
        # Process all items for this location in sequence order
        for (equip, proc) in items_by_location[loc]:
            if proc == 'Make':
                unit_key = mfg_by_loc_equip.get((loc, equip))
                if unit_key and unit_key not in added_mfg:
                    fig = manufacturing_figs.get(unit_key)
                    if fig:
                        content.append({"type": "fig", "fig": fig, "title": f"Manufacturing: {unit_key}", "category": "manufacturing", "location": loc})
                        added_mfg.add(unit_key)
            
            elif proc == 'Store':
                key = (loc, equip)
                if key in store_by_loc_name:
                    for (sk, product) in store_by_loc_name[key]:
                        if sk not in added_stores:
                            content.append({"type": "fig", "fig": store_figs[sk], "category": "inventory", "product": product, "location": loc})
                            added_stores.add(sk)
    
    # Add any remaining manufacturing/store graphs not in the sequence
    remaining_stores = [(sk, prod) for items in store_by_loc_name.values() for (sk, prod) in items if sk not in added_stores]
    remaining_mfg = [uk for uk in manufacturing_figs.keys() if uk not in added_mfg]
    
    if remaining_stores or remaining_mfg:
        location_order.append("Other")
        content.append({"type": "header", "location": "Other", "category": "inventory"})
        for unit_key in sorted(remaining_mfg):
            fig = manufacturing_figs[unit_key]
            content.append({"type": "fig", "fig": fig, "title": f"Manufacturing: {unit_key}", "category": "manufacturing", "location": "Other"})
        for (sk, product) in sorted(remaining_stores):
            content.append({"type": "fig", "fig": store_figs[sk], "category": "inventory", "product": product, "location": "Other"})

    print(f"  [timing] Content assembly: {time.time()-t1:.1f}s")
    t1 = time.time()
    _generate_html_report(sim, out_dir, content, sorted(all_products), location_order, current_runtime, prev_runtime)
    print(f"  [timing] HTML write: {time.time()-t1:.1f}s")
    print(f"  [timing] TOTAL plot_results: {time.time()-t0:.1f}s")


def _generate_transport_plot(df_log: pd.DataFrame, equipment_type: str = None) -> go.Figure:
    if df_log.empty: return None

    moves = df_log[df_log['process'] == 'Move'].copy()
    if moves.empty: return None

    if equipment_type:
        moves = moves[moves['equipment'] == equipment_type]
        if moves.empty: return None

    moves['day'] = pd.to_numeric(moves['time_h'], errors='coerce') / 24.0
    if 'qty_t' not in moves.columns:
        moves['qty_t'] = pd.to_numeric(moves.get('qty', 0), errors='coerce').fillna(0)

    def make_route_name(row):
        try:
            if row.get('route_id') and str(row['route_id']) != 'nan':
                return f"{row['route_id']} ({row.get('product', '')})"
            if row.get('route_group'):
                return f"{row['route_group']}"
            return "Unknown Route"
        except:
            return "Unknown"

    moves['Route'] = moves.apply(make_route_name, axis=1)

    loads = moves[moves['event'].isin(['Load', 'ShipLoad'])]
    unloads = moves[moves['event'].isin(['Unload', 'ShipUnload'])]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=loads['day'], y=loads['Route'], mode='markers',
                             marker=dict(symbol='triangle-right', size=8, color='green', opacity=0.7), name='Load',
                             text=loads['qty_t'].apply(lambda x: f"{x:,.0f} t"),
                             hovertemplate="<b>LOAD</b><br>Day: %{x:.1f}<br>Route: %{y}<br>Qty: %{text}<extra></extra>"))
    fig.add_trace(go.Scatter(x=unloads['day'], y=unloads['Route'], mode='markers',
                             marker=dict(symbol='triangle-left', size=8, color='red', opacity=0.7), name='Unload',
                             text=unloads['qty_t'].apply(lambda x: f"{x:,.0f} t"),
                             hovertemplate="<b>UNLOAD</b><br>Day: %{x:.1f}<br>Route: %{y}<br>Qty: %{text}<extra></extra>"))

    title = "Transport Activity Timeline"
    if equipment_type == "Train":
        title = "Rail Transport Timeline"
    elif equipment_type == "Ship":
        title = "Ship Transport Timeline"

    fig.update_layout(title=title, xaxis_title="Day of Year", yaxis_title="Route",
                      height=max(400, len(moves['Route'].unique()) * 30), template="plotly_white",
                      yaxis=dict(autorange="reversed"))
    return fig


def _generate_ship_timeline_by_route_group(df_log: pd.DataFrame) -> List[go.Figure]:
    """Generate separate ship timeline charts for each route group, showing loads/unloads by location|product."""
    if df_log.empty: return []

    moves = df_log[(df_log['process'] == 'Move') & (df_log['equipment'] == 'Ship')].copy()
    if moves.empty: return []

    moves['day'] = pd.to_numeric(moves['time_h'], errors='coerce') / 24.0
    if 'qty_t' not in moves.columns:
        moves['qty_t'] = pd.to_numeric(moves.get('qty', 0), errors='coerce').fillna(0)
    
    # Create Location|Product label for Y-axis
    moves['loc_product'] = moves['location'].fillna('') + '|' + moves['product'].fillna('')

    # Map numeric route_ids to route groups based on integer prefix
    # 1.x or 2 (integer) = North, 2.x = South, 3.x = Import_CL, 4.x = Import_GBFS
    def get_route_group(route_id):
        if pd.isna(route_id):
            return 'Unknown'
        route_str = str(route_id).strip()
        # Extract integer prefix (before decimal)
        if '.' in route_str:
            prefix = route_str.split('.')[0]
        else:
            prefix = route_str
        try:
            prefix_int = int(float(prefix))
            if prefix_int == 1:
                return 'North'
            elif prefix_int == 2:
                # Check if it's 2.x (South) or just 2 (North - Route 10)
                if '.' in route_str:
                    return 'South'
                else:
                    return 'North'  # Route ID "2" without decimal is North (Route 10)
            elif prefix_int == 3:
                return 'Import_CL'
            elif prefix_int == 4:
                return 'Import_GBFS'
            else:
                return f'Route_{prefix_int}'
        except:
            return 'Unknown'
    
    moves['route_group'] = moves['route_id'].apply(get_route_group)
    route_groups = moves['route_group'].dropna().unique()
    figs = []

    for rg in sorted(route_groups):
        rg_moves = moves[moves['route_group'] == rg].copy()
        if rg_moves.empty: continue

        loads = rg_moves[rg_moves['event'].isin(['Load', 'ShipLoad'])]
        unloads = rg_moves[rg_moves['event'].isin(['Unload', 'ShipUnload'])]

        fig = go.Figure()

        if not loads.empty:
            # Format payload in kT and show specific route ID for tooltip
            load_texts = loads.apply(
                lambda r: f"Route ID: {r['route_id']}<br>Payload: {r['qty_t']/1000:.1f}kT", axis=1
            )
            fig.add_trace(go.Scatter(
                x=loads['day'], y=loads['loc_product'], mode='markers',
                marker=dict(symbol='triangle-right', size=10, color='green', opacity=0.7),
                name='Load',
                text=load_texts,
                hovertemplate="<b>LOAD</b><br>Day: %{x:.1f}<br>%{y}<br>%{text}<extra></extra>"
            ))

        if not unloads.empty:
            # Format payload in kT and show specific route ID for tooltip
            unload_texts = unloads.apply(
                lambda r: f"Route ID: {r['route_id']}<br>Payload: {r['qty_t']/1000:.1f}kT", axis=1
            )
            fig.add_trace(go.Scatter(
                x=unloads['day'], y=unloads['loc_product'], mode='markers',
                marker=dict(symbol='triangle-left', size=10, color='red', opacity=0.7),
                name='Unload',
                text=unload_texts,
                hovertemplate="<b>UNLOAD</b><br>Day: %{x:.1f}<br>%{y}<br>%{text}<extra></extra>"
            ))

        loc_products = rg_moves['loc_product'].dropna().unique()
        fig.update_layout(
            title=f"Ship Route: {rg}",
            xaxis_title="Day of Year",
            yaxis_title="Location|Product",
            height=max(300, len(loc_products) * 40),
            template="plotly_white",
            yaxis=dict(autorange="reversed")
        )
        figs.append((rg, fig))

    return figs


def _generate_manufacturing_charts(df_log: pd.DataFrame) -> dict:
    """Generate production charts for each manufacturing unit with downtime markers."""
    if df_log.empty: return {}

    df_log['day'] = (pd.to_numeric(df_log['time_h'], errors='coerce') / 24.0).astype(int) + 1

    production = df_log[(df_log['process'] == 'Make') & (df_log['event'] == 'Produce')].copy()
    downtime = df_log[df_log['process'] == 'Downtime'].copy()

    if production.empty:
        return {}

    production['qty_t'] = pd.to_numeric(production['qty'], errors='coerce').fillna(0)

    daily_prod = production.groupby(['location', 'equipment', 'product', 'day'])['qty_t'].sum().reset_index()

    downtime_by_unit = {}
    downtime_hours_by_unit = {}
    if not downtime.empty:
        for _, row in downtime.iterrows():
            loc = row.get('location', '')
            equip = row.get('equipment', '')
            d = int(row['day'])
            event_type = row.get('event', 'Unknown')
            key = f"{loc}|{equip}"
            if key not in downtime_by_unit:
                downtime_by_unit[key] = {}
                downtime_hours_by_unit[key] = {}
            if d not in downtime_by_unit[key]:
                downtime_by_unit[key][d] = {'Maintenance': 0, 'Breakdown': 0}
                downtime_hours_by_unit[key][d] = 0
            if event_type in downtime_by_unit[key][d]:
                downtime_by_unit[key][d][event_type] += 1
            downtime_hours_by_unit[key][d] += 1

    figs = {}
    for (loc, equip), group in daily_prod.groupby(['location', 'equipment']):
        fig = go.Figure()

        for product in group['product'].unique():
            prod_data = group[group['product'] == product]
            fig.add_trace(go.Bar(
                x=prod_data['day'],
                y=prod_data['qty_t'],
                name=f"{product}",
                hovertemplate=f"{product}: %{{y:,.0f}}t on Day %{{x}}<extra></extra>"
            ))

        unit_key = f"{loc}|{equip}"
        if unit_key in downtime_by_unit:
            unit_dt = downtime_by_unit[unit_key]
            max_prod = group['qty_t'].max() if not group.empty else 100

            maint_days = [d for d in unit_dt if unit_dt[d].get('Maintenance', 0) > 0]
            maint_intervals = _merge_days_to_intervals(maint_days)
            for start, end in maint_intervals:
                fig.add_vrect(
                    x0=start - 0.5, x1=end + 0.5,
                    fillcolor="rgba(255, 152, 0, 0.2)",
                    layer="below",
                    line_width=0,
                )
            if maint_days:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    name="Maintenance",
                    mode='markers',
                    marker=dict(symbol='square', size=12, color='rgba(255, 152, 0, 0.5)'),
                    showlegend=True
                ))
            
            breakdown_days = [d for d in unit_dt if unit_dt[d].get('Breakdown', 0) >= 24]
            breakdown_intervals = _merge_days_to_intervals(breakdown_days)
            for start, end in breakdown_intervals:
                fig.add_vrect(
                    x0=start - 0.5, x1=end + 0.5,
                    fillcolor="rgba(244, 67, 54, 0.2)",
                    layer="below",
                    line_width=0,
                )
            if breakdown_days:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    name="Breakdown (Full Day)",
                    mode='markers',
                    marker=dict(symbol='square', size=12, color='rgba(244, 67, 54, 0.5)'),
                    showlegend=True
                ))


        if unit_key in downtime_hours_by_unit:
            unit_hours = downtime_hours_by_unit[unit_key]
            if unit_hours:
                max_day = max(max(group['day']), max(unit_hours.keys()))
                cumulative = 0
                cum_days = []
                cum_hours = []
                for d in range(1, max_day + 1):
                    cumulative += unit_hours.get(d, 0)
                    cum_days.append(d)
                    cum_hours.append(cumulative)
                
                if cum_hours and max(cum_hours) > 0:
                    fig.add_trace(go.Scatter(
                        x=cum_days, y=cum_hours,
                        name="Cumulative Downtime",
                        mode='lines',
                        line=dict(color='#9c27b0', width=2, dash='dot'),
                        yaxis='y2',
                        hovertemplate="Day %{x}: %{y:.0f} hours total downtime<extra></extra>"
                    ))

        fig.update_layout(
            title=f"Production: {equip} @ {loc}",
            xaxis_title="Day",
            yaxis_title="Production (Tons)",
            yaxis2=dict(
                title="Cumulative Downtime (Hours)",
                overlaying='y',
                side='right',
                showgrid=False
            ),
            template="plotly_white",
            height=350,
            barmode='stack',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        figs[f"{loc}|{equip}"] = fig

    return figs


def _generate_fleet_utilisation_chart(df_log: pd.DataFrame) -> go.Figure:
    """Generate daily fleet utilization chart - IDLE = not utilized, else utilized."""
    if df_log.empty: return None

    state_changes = df_log[df_log['process'] == 'ShipState'].copy()
    if state_changes.empty: return None

    state_changes['time_h'] = pd.to_numeric(state_changes['time_h'], errors='coerce')
    state_changes = state_changes.sort_values('time_h')
    state_changes['day'] = (state_changes['time_h'] / 24).astype(int) + 1
    
    max_day = state_changes['day'].max()
    
    # Process events once, tracking state at end of each day
    vessel_states = {}
    daily_snapshots = {}  # day -> snapshot of vessel_states at end of day
    
    for _, row in state_changes.iterrows():
        vid = row.get('vessel_id')
        new_state = row.get('ship_state')
        day = row['day']
        if vid is not None and new_state is not None:
            vessel_states[vid] = new_state
            daily_snapshots[day] = dict(vessel_states)  # snapshot current state
    
    # Build daily utilization from snapshots (forward-fill gaps)
    daily_util = []
    last_snapshot = {}
    for day in range(1, max_day + 1):
        if day in daily_snapshots:
            last_snapshot = daily_snapshots[day]
        
        total_vessels = len(last_snapshot)
        if total_vessels > 0:
            idle_count = sum(1 for st in last_snapshot.values() if st == 'IDLE')
            util_pct = ((total_vessels - idle_count) / total_vessels) * 100
        else:
            util_pct = 0
        daily_util.append({'day': day, 'utilization': util_pct})

    if not daily_util:
        return None

    df_util = pd.DataFrame(daily_util)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_util['day'],
        y=df_util['utilization'],
        mode='lines',
        name='Fleet Utilization',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))

    fig.update_layout(
        title="Ship Fleet Utilization by Day",
        xaxis_title="Day of Year",
        yaxis_title="Utilization (%)",
        template="plotly_white",
        height=350,
        yaxis=dict(range=[0, 105])
    )

    return fig


def _generate_vessel_state_chart(df_log: pd.DataFrame) -> go.Figure:
    """Generate stacked area chart showing vessel count by state over time."""
    if df_log.empty: return None

    state_changes = df_log[df_log['process'] == 'ShipState'].copy()
    if state_changes.empty: return None

    state_changes['day'] = pd.to_numeric(state_changes['time_h'], errors='coerce') / 24.0
    state_changes = state_changes.sort_values('time_h')

    states = ['IDLE', 'LOADING', 'IN_TRANSIT', 'WAITING_FOR_BERTH', 'UNLOADING']
    state_colors = {
        'IDLE': '#2ca02c',
        'LOADING': '#1f77b4',
        'IN_TRANSIT': '#ff7f0e',
        'WAITING_FOR_BERTH': '#d62728',
        'UNLOADING': '#9467bd'
    }

    vessel_states = {}
    timeline_data = []

    for _, row in state_changes.iterrows():
        vid = row.get('vessel_id')
        new_state = row.get('ship_state')
        day = row['day']

        if vid is not None and new_state is not None:
            vessel_states[vid] = new_state

            counts = {s: 0 for s in states}
            for v, st in vessel_states.items():
                if st in counts:
                    counts[st] += 1

            timeline_data.append({'day': day, **counts})

    if not timeline_data:
        return None

    df_timeline = pd.DataFrame(timeline_data)

    fig = go.Figure()
    for state in states:
        if state in df_timeline.columns:
            fig.add_trace(go.Scatter(
                x=df_timeline['day'],
                y=df_timeline[state],
                mode='lines',
                name=state.replace('_', ' ').title(),
                fill='tonexty' if state != 'IDLE' else 'tozeroy',
                line=dict(width=0.5, color=state_colors.get(state, '#999')),
                stackgroup='one'
            ))

    fig.update_layout(
        title="Fleet State Over Time",
        xaxis_title="Day of Year",
        yaxis_title="Number of Vessels",
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(tickformat='d')

    return fig


def _generate_route_summary_chart(df_log: pd.DataFrame) -> go.Figure:
    """Generate stacked bar chart showing avg time per status for each route, with trip count labels."""
    if df_log.empty:
        return None

    state_changes = df_log[df_log['process'] == 'ShipState'].copy()
    if state_changes.empty:
        return None

    state_changes['time_h'] = pd.to_numeric(state_changes['time_h'], errors='coerce')
    state_changes = state_changes.sort_values(['vessel_id', 'time_h'])

    states = ['LOADING', 'IN_TRANSIT', 'WAITING_FOR_BERTH', 'UNLOADING', 'IDLE']
    state_colors = {
        'IDLE': '#2ca02c',
        'LOADING': '#1f77b4',
        'IN_TRANSIT': '#ff7f0e',
        'WAITING_FOR_BERTH': '#d62728',
        'UNLOADING': '#9467bd'
    }

    # Track time spent in each state per route trip
    route_trips = {}  # route_id -> list of {state: hours}
    
    # Process by vessel to track state durations
    for vessel_id in state_changes['vessel_id'].dropna().unique():
        vessel_data = state_changes[state_changes['vessel_id'] == vessel_id].copy()
        vessel_data = vessel_data.sort_values('time_h')
        
        current_route = None
        current_trip = {s: 0.0 for s in states}
        prev_time = None
        prev_state = None
        
        for _, row in vessel_data.iterrows():
            route_id = row.get('route_id')
            new_state = row.get('ship_state')
            time_h = row['time_h']
            
            # Calculate duration for previous state
            if prev_time is not None and prev_state is not None:
                duration = time_h - prev_time
                if prev_state in current_trip:
                    current_trip[prev_state] += duration
            
            # Detect route change (new trip starts when transitioning to LOADING with a route)
            if new_state == 'LOADING' and route_id and str(route_id) != 'nan':
                # Save previous trip if it had activity
                if current_route and sum(current_trip.values()) > 0:
                    if current_route not in route_trips:
                        route_trips[current_route] = []
                    route_trips[current_route].append(current_trip.copy())
                
                # Start new trip
                current_route = str(route_id)
                current_trip = {s: 0.0 for s in states}
            
            # If route changed mid-trip, update current route
            if route_id and str(route_id) != 'nan':
                current_route = str(route_id)
            
            prev_time = time_h
            prev_state = new_state
        
        # Save final trip
        if current_route and sum(current_trip.values()) > 0:
            if current_route not in route_trips:
                route_trips[current_route] = []
            route_trips[current_route].append(current_trip.copy())

    if not route_trips:
        return None

    # Calculate averages per route
    route_summaries = []
    for route_id, trips in route_trips.items():
        n_trips = len(trips)
        if n_trips == 0:
            continue
        avg_times = {s: sum(t[s] for t in trips) / n_trips for s in states}
        route_summaries.append({
            'route_id': route_id,
            'n_trips': n_trips,
            **avg_times
        })

    if not route_summaries:
        return None

    # Sort by route_id numerically (ascending: 1.01, 1.02, 1.03...)
    def sort_key(x):
        try:
            return float(x['route_id'])
        except:
            return float('inf')
    route_summaries.sort(key=sort_key)
    
    route_ids = [r['route_id'] for r in route_summaries]
    trip_counts = [r['n_trips'] for r in route_summaries]
    n_routes = len(route_ids)

    fig = go.Figure()

    # Add stacked bars for each state with text labels showing hours
    display_states = ['LOADING', 'IN_TRANSIT', 'WAITING_FOR_BERTH', 'UNLOADING']
    for state in display_states:
        values = [r[state] for r in route_summaries]
        # Only show text if value is significant (> 10 hours to avoid clutter)
        text_labels = [f"{v:.0f}h" if v >= 10 else "" for v in values]
        fig.add_trace(go.Bar(
            y=list(range(n_routes)),  # Use numeric indices
            x=values,
            name=state.replace('_', ' ').title(),
            orientation='h',
            marker_color=state_colors.get(state, '#999'),
            text=text_labels,
            textposition='inside',
            insidetextanchor='middle',
            textfont=dict(size=10, color='white'),
            hovertemplate=f"<b>{state.replace('_', ' ').title()}</b><br>Route: %{{customdata}}<br>Avg: %{{x:.1f}} hrs<extra></extra>",
            customdata=route_ids
        ))

    # Calculate total time for annotations (position at end of bar)
    totals = [sum(r[s] for s in display_states) for r in route_summaries]
    max_total = max(totals) if totals else 1

    # Add trip count annotations at end of each bar
    annotations = []
    for i, (route_id, total, count) in enumerate(zip(route_ids, totals, trip_counts)):
        annotations.append(dict(
            x=total + max_total * 0.02,
            y=i,
            text=f"{count} trips",
            showarrow=False,
            font=dict(size=10, color='#555'),
            xanchor='left',
            yanchor='middle'
        ))

    fig.update_layout(
        title="Route Summary: Avg Time by Status",
        xaxis_title="Average Hours per Trip",
        yaxis_title="Route ID",
        template="plotly_white",
        height=max(350, n_routes * 35 + 100),
        barmode='stack',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        annotations=annotations,
        margin=dict(r=100),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(n_routes)),
            ticktext=route_ids,
            autorange=False,
            range=[-0.5, n_routes - 0.5]
        )
    )

    return fig


def _generate_html_report(sim, out_dir: Path, content: list, products: list = None, locations: list = None, current_runtime: int = 0, prev_runtime: int = 30):
    html_path = out_dir / "sim_outputs_plots_all.html"
    total_unmet = sum(sim.unmet.values())
    run_time_utc = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    products = products or []
    locations = locations or []
    # Use previous runtime for countdown (current run's time will be used next time)
    countdown_seconds = prev_runtime if prev_runtime > 0 else 30
    # Display current runtime in summary
    display_runtime = current_runtime if current_runtime > 0 else prev_runtime

    product_buttons = ''.join([f'<button class="filter-btn product-btn" data-product="{p}" onclick="toggleProduct(\'{p}\')">{p}</button>' for p in products])
    location_checkboxes = ''.join([f'<label class="loc-checkbox"><input type="checkbox" checked data-location="{loc}" onchange="toggleLocation(\'{loc}\')"><span>{loc}</span></label>' for loc in locations])

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Supply Chain Report</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }}
        body {{ margin: 0; padding-top: 120px; background: #f0f4f8; min-height: 100vh; color: #2d3748; }}
        .sticky-header {{ position: fixed; top: 0; left: 0; right: 0; background: rgba(30, 35, 50, 0.98); backdrop-filter: blur(20px); padding: 12px 40px; z-index: 1000; box-shadow: 0 4px 20px rgba(0,0,0,0.15); border-bottom: 1px solid rgba(255,255,255,0.08); }}
        .header-top {{ display: flex; justify-content: space-between; align-items: center; }}
        .sticky-header h1 {{ color: #4fc3f7; margin: 0; font-size: 1.4em; font-weight: 600; text-shadow: 0 0 40px rgba(79,195,247,0.3); }}
        .run-btn {{ background: linear-gradient(135deg, #4fc3f7 0%, #00b4d8 50%, #0096c7 100%); color: #1a1a2e; border: none; padding: 12px 24px; font-size: 0.85em; font-weight: 600; border-radius: 12px; cursor: pointer; transition: all 0.3s ease; text-transform: uppercase; letter-spacing: 0.5px; box-shadow: 0 4px 20px rgba(79,195,247,0.3); }}
        .run-btn:hover {{ background: linear-gradient(135deg, #00b4d8 0%, #0096c7 50%, #0077b6 100%); transform: translateY(-2px); box-shadow: 0 6px 30px rgba(79,195,247,0.4); }}
        .run-btn:disabled {{ cursor: not-allowed; transform: none; }}
        .run-btn.running-badge {{ background: linear-gradient(135deg, #f57c00, #e65100); color: #fff; box-shadow: 0 2px 12px rgba(245,124,0,0.4); animation: pulse 1s infinite; }}
        .run-btn.finishing-badge {{ background: linear-gradient(135deg, #7c4dff, #651fff); color: #fff; box-shadow: 0 2px 12px rgba(124,77,255,0.4); animation: pulse 1s infinite; }}
        .run-btn.complete-badge {{ background: linear-gradient(135deg, #2e7d32, #1b5e20); color: #a5d6a7; box-shadow: 0 2px 12px rgba(46,125,50,0.4); animation: none; }}
        @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.85; }} }}
        #status {{ color: #a0aec0; font-size: 0.9em; margin-left: 15px; }}
        .filter-bar {{ display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap; align-items: center; }}
        .filter-label {{ color: #a0aec0; font-size: 0.85em; margin-right: 5px; font-weight: 500; }}
        .filter-btn {{ background: rgba(255,255,255,0.1); color: #b0bec5; border: 1px solid rgba(255,255,255,0.15); padding: 6px 14px; font-size: 0.85em; border-radius: 20px; cursor: pointer; transition: all 0.2s ease; }}
        .filter-btn:hover {{ background: rgba(79,195,247,0.2); border-color: #4fc3f7; color: #4fc3f7; }}
        .filter-btn.active {{ background: #4fc3f7; color: #1a1a2e; border-color: #4fc3f7; font-weight: 600; }}
        .filter-divider {{ border-left: 1px solid rgba(255,255,255,0.15); height: 24px; margin: 0 10px; }}
        .dropdown {{ position: relative; display: inline-block; }}
        .dropdown-btn {{ background: rgba(255,255,255,0.1); color: #b0bec5; border: 1px solid rgba(255,255,255,0.15); padding: 6px 14px; font-size: 0.85em; border-radius: 20px; cursor: pointer; transition: all 0.2s ease; }}
        .dropdown-btn:hover {{ background: rgba(79,195,247,0.2); border-color: #4fc3f7; color: #4fc3f7; }}
        .dropdown-content {{ display: none; position: absolute; top: 100%; left: 0; background: rgba(30, 35, 50, 0.98); border: 1px solid rgba(255,255,255,0.15); border-radius: 8px; min-width: 180px; max-height: 300px; overflow-y: auto; z-index: 1001; margin-top: 4px; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }}
        .dropdown-content.show {{ display: block; }}
        .loc-checkbox {{ display: flex; align-items: center; padding: 8px 12px; cursor: pointer; color: #b0bec5; font-size: 0.85em; transition: background 0.2s; }}
        .loc-checkbox:hover {{ background: rgba(79,195,247,0.15); }}
        .loc-checkbox input {{ margin-right: 8px; accent-color: #4fc3f7; }}
        .loc-checkbox span {{ white-space: nowrap; }}
        .dropdown-actions {{ display: flex; gap: 8px; padding: 8px 12px; border-top: 1px solid rgba(255,255,255,0.1); }}
        .dropdown-actions button {{ flex: 1; padding: 4px 8px; font-size: 0.75em; border-radius: 4px; border: none; cursor: pointer; }}
        .dropdown-actions .select-all {{ background: #4fc3f7; color: #1a1a2e; }}
        .dropdown-actions .clear-all {{ background: rgba(255,255,255,0.1); color: #b0bec5; }}
        .quick-btn {{ background: rgba(255,255,255,0.06); color: #78909c; border: 1px solid rgba(255,255,255,0.1); padding: 4px 10px; font-size: 0.75em; border-radius: 12px; cursor: pointer; transition: all 0.2s ease; }}
        .quick-btn:hover {{ background: rgba(79,195,247,0.15); color: #4fc3f7; }}
        .content {{ padding: 20px 40px; max-width: 1400px; margin: 0 auto; }}
        .summary {{ background: #fff; padding: 20px 24px; border-radius: 12px; margin-bottom: 30px; border-left: 4px solid #4fc3f7; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }}
        .summary p {{ margin: 8px 0; color: #4a5568; }}
        .summary strong {{ color: #2b6cb0; }}
        .plot {{ margin: 20px 0; background: #fff; padding: 20px; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }}
        .plot.hidden {{ display: none; }}
        h2 {{ color: #2b6cb0; margin-top: 40px; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; font-weight: 600; }}
        h2.hidden {{ display: none; }}
        h3 {{ color: #4a5568; margin-top: 30px; font-size: 1.1em; font-weight: 500; }}
        h3.hidden {{ display: none; }}
        .footer {{ margin-top: 50px; color: #a0aec0; font-size: 0.9em; text-align: center; padding-bottom: 20px; }}
    </style>
</head>
<body data-countdown="{countdown_seconds}">
    <div class="sticky-header">
        <div class="header-top">
            <h1>Cement Australia Supply Chain Simulation</h1>
            <div><button class="run-btn" id="runBtn" onclick="runSimulation()">Run Single Simulation</button><span id="status"></span></div>
        </div>
        <div class="filter-bar">
            <span class="filter-label">Categories:</span>
            <button class="quick-btn" onclick="selectAllCategories()">All</button>
            <button class="quick-btn" onclick="clearAllCategories()">None</button>
            <button class="filter-btn active" data-category="inventory" onclick="toggleCategory('inventory')">Inventory</button>
            <button class="filter-btn active" data-category="manufacturing" onclick="toggleCategory('manufacturing')">Manufacturing</button>
            <button class="filter-btn active" data-category="shipping" onclick="toggleCategory('shipping')">Shipping</button>
            <div class="filter-divider"></div>
            <span class="filter-label">Products:</span>
            <button class="quick-btn" onclick="selectAllProducts()">All</button>
            <button class="quick-btn" onclick="clearAllProducts()">None</button>
            {product_buttons}
            <div class="filter-divider"></div>
            <span class="filter-label">Locations:</span>
            <div class="dropdown">
                <button class="dropdown-btn" onclick="toggleDropdown()">Select Locations ▾</button>
                <div class="dropdown-content" id="locationDropdown">
                    <div class="dropdown-actions">
                        <button class="select-all" onclick="selectAllLocations()">All</button>
                        <button class="clear-all" onclick="clearAllLocations()">None</button>
                    </div>
                    {location_checkboxes}
                </div>
            </div>
        </div>
    </div>
    <div class="content">
        <div class="summary"><p><strong>Generated:</strong> <span id="genTime" data-utc="{run_time_utc}"></span> <span style="color:#78909c">(run time {display_runtime}s)</span></p><p><strong>Total Unmet Demand:</strong> <span style="color: {'red' if total_unmet > 0 else 'green'}">{total_unmet:,.0f} tons</span></p><p><strong>Status:</strong> Complete</p></div>
        <script>
            (function() {{
                var el = document.getElementById('genTime');
                var utc = el.dataset.utc;
                var d = new Date(utc);
                var y = d.getFullYear();
                var m = String(d.getMonth() + 1).padStart(2, '0');
                var day = String(d.getDate()).padStart(2, '0');
                var h = String(d.getHours()).padStart(2, '0');
                var min = String(d.getMinutes()).padStart(2, '0');
                el.textContent = y + '-' + m + '-' + day + ' ' + h + ':' + min;
            }})();
        </script>
"""

    # Pre-serialize all figures using orjson (5-8x faster than default)
    import plotly.io as pio
    fig_items = [(i, item) for i, item in enumerate(content) if item["type"] == "fig" and item.get("fig")]
    fig_json_map = {}
    
    for idx, item in fig_items:
        fig_json_map[idx] = pio.to_json(item["fig"], engine="orjson", validate=False)
    
    # Build HTML using list (faster than string concatenation)
    html_parts = [html]
    div_id = 0
    for i, item in enumerate(content):
        category = item.get("category", "")
        product = item.get("product", "")
        location = item.get("location", "")
        if item["type"] == "header":
            html_parts.append(f'<h2 class="section-header" data-category="{category}" data-location="{location}">Location: {location}</h2>')
        elif item["type"] == "fig" and item.get("fig"):
            title = item.get("title", "")
            if title:
                html_parts.append(f'<h3 class="plot-title" data-category="{category}" data-product="{product}" data-location="{location}">{title}</h3>')
            fig_json = fig_json_map.get(i, item["fig"].to_json())
            html_parts.append(f'<div id="plot-{div_id}" class="plot" data-category="{category}" data-product="{product}" data-location="{location}"></div><script>Plotly.newPlot("plot-{div_id}", {fig_json});</script>')
            div_id += 1

    html_parts.append("""</div><div class="footer"><p>Generated by sim_run_report_plot.py</p></div>
    <script>
    const categoryState = { inventory: true, manufacturing: true, shipping: true };
    const productState = {};
    const locationState = {};
    
    document.querySelectorAll('.product-btn').forEach(btn => {
        productState[btn.dataset.product] = true;
        btn.classList.add('active');
    });
    
    document.querySelectorAll('.loc-checkbox input').forEach(cb => {
        locationState[cb.dataset.location] = cb.checked;
    });

    function toggleCategory(cat) {
        categoryState[cat] = !categoryState[cat];
        document.querySelector(`[data-category="${cat}"].filter-btn`).classList.toggle('active', categoryState[cat]);
        applyFilters();
    }
    
    function selectAllCategories() {
        ['inventory', 'manufacturing', 'shipping'].forEach(cat => {
            categoryState[cat] = true;
            document.querySelector(`[data-category="${cat}"].filter-btn`).classList.add('active');
        });
        applyFilters();
    }
    
    function clearAllCategories() {
        ['inventory', 'manufacturing', 'shipping'].forEach(cat => {
            categoryState[cat] = false;
            document.querySelector(`[data-category="${cat}"].filter-btn`).classList.remove('active');
        });
        applyFilters();
    }

    function toggleProduct(prod) {
        productState[prod] = !productState[prod];
        document.querySelector(`[data-product="${prod}"].filter-btn`).classList.toggle('active', productState[prod]);
        applyFilters();
    }
    
    function selectAllProducts() {
        document.querySelectorAll('.product-btn').forEach(btn => {
            productState[btn.dataset.product] = true;
            btn.classList.add('active');
        });
        applyFilters();
    }
    
    function clearAllProducts() {
        document.querySelectorAll('.product-btn').forEach(btn => {
            productState[btn.dataset.product] = false;
            btn.classList.remove('active');
        });
        applyFilters();
    }
    
    function toggleLocation(loc) {
        const cb = document.querySelector(`.loc-checkbox input[data-location="${loc}"]`);
        locationState[loc] = cb.checked;
        applyFilters();
    }
    
    function toggleDropdown() {
        document.getElementById('locationDropdown').classList.toggle('show');
    }
    
    function selectAllLocations() {
        document.querySelectorAll('.loc-checkbox input').forEach(cb => {
            cb.checked = true;
            locationState[cb.dataset.location] = true;
        });
        applyFilters();
    }
    
    function clearAllLocations() {
        document.querySelectorAll('.loc-checkbox input').forEach(cb => {
            cb.checked = false;
            locationState[cb.dataset.location] = false;
        });
        applyFilters();
    }
    
    // Close dropdown when clicking outside
    document.addEventListener('click', function(e) {
        if (!e.target.closest('.dropdown')) {
            document.getElementById('locationDropdown').classList.remove('show');
        }
    });

    function applyFilters() {
        document.querySelectorAll('.plot, .plot-title, .section-header').forEach(el => {
            const cat = el.dataset.category;
            const prod = el.dataset.product || '';
            const loc = el.dataset.location || '';
            let show = true;
            if (cat && !categoryState[cat]) show = false;
            if (prod && !productState[prod]) show = false;
            if (loc && !locationState[loc]) show = false;
            el.classList.toggle('hidden', !show);
        });
    }

    let countdownInterval = null;
    
    async function runSimulation() {
        const btn = document.getElementById('runBtn');
        const status = document.getElementById('status');
        const prevRuntime = parseInt(document.body.dataset.countdown) || 30;
        let countdown = prevRuntime;
        
        btn.disabled = true;
        btn.className = 'run-btn running-badge';
        btn.textContent = countdown + 's';
        status.innerHTML = '<span style="color:#fff;margin-left:8px;">Running...</span>';
        
        countdownInterval = setInterval(() => {
            countdown--;
            if (countdown > 0) {
                btn.textContent = countdown + 's';
            } else {
                btn.textContent = '...';
                btn.className = 'run-btn finishing-badge';
                status.innerHTML = '<span style="color:#fff;margin-left:8px;">Finishing Up...</span>';
            }
        }, 1000);
        
        try {
            const response = await fetch('/run-simulation', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ horizon_days: 365, random_opening: true }) });
            clearInterval(countdownInterval);
            const result = await response.json();
            if (result.success && result.report_ready) {
                btn.textContent = 'Complete';
                btn.className = 'run-btn complete-badge';
                status.innerHTML = '<span style="color:#a5d6a7;margin-left:8px;">Reloading...</span>';
                setTimeout(() => window.location.reload(), 500);
            } else {
                status.textContent = 'Error: ' + (result.output || 'Unknown error');
                btn.disabled = false;
                btn.className = 'run-btn';
                btn.textContent = 'Run Single Simulation';
            }
        } catch (err) {
            clearInterval(countdownInterval);
            status.textContent = 'Error: ' + err.message;
            btn.disabled = false;
            btn.className = 'run-btn';
            btn.textContent = 'Run Single Simulation';
        }
    }
    </script></body></html>""")

    html_path.write_text(''.join(html_parts), encoding="utf-8")
    print(f"Interactive HTML report: {html_path}")