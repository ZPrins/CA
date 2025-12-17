# sim_run_report_plot.py
from typing import List, Tuple
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
from sim_run_config import config


def plot_results(sim, out_dir: Path, routes: list | None = None, makes: list | None = None, graph_sequence: list | None = None):
    if not config.write_plots: return
    
    makes = makes or []
    graph_sequence = graph_sequence or []  # List of (Location, Equipment, Process) tuples

    # 1. Inventory Data (Daily)
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

    # 2. Process Action Log for Flows (Rail In, Ship In, etc.)
    flows = {}  # store_key -> {day -> {rail_in: 0, ship_in: 0...}}

    if sim.action_log:
        df_log = pd.DataFrame(sim.action_log)
        df_log["day"] = (pd.to_numeric(df_log["time_h"], errors="coerce") / 24.0).astype(int) + 1

        # Ensure qty_t
        if "qty_t" not in df_log.columns and "qty" in df_log.columns:
            df_log["qty_t"] = pd.to_numeric(df_log["qty"], errors="coerce").fillna(0)

        # --- Aggregation Helper ---
        def aggregate_flow(mask, equipment_type, direction):
            # direction: 'in' or 'out'
            subset = df_log[mask].copy()
            if subset.empty: return

            # Derive store_key based on flow direction from log schema
            # Our action_log uses 'from_store' and 'to_store' keys
            if direction == 'in':
                subset["store_key"] = subset.get("to_store")
            else:
                subset["store_key"] = subset.get("from_store")

            # Ensure numeric qty column
            if "qty_t" not in subset.columns and "qty" in subset.columns:
                subset["qty_t"] = pd.to_numeric(subset["qty"], errors="coerce").fillna(0)

            # Drop rows where store_key is missing
            subset = subset.dropna(subset=["store_key"]) 

            if subset.empty: return

            # Group by Store and Day
            grouped = subset.groupby(["store_key", "day"])["qty_t"].sum().reset_index()

            for _, row in grouped.iterrows():
                sk = row["store_key"]
                d = int(row["day"])
                q = row["qty_t"]

                if sk not in flows: flows[sk] = {}
                if d not in flows[sk]: flows[sk][d] = {}

                key = f"{equipment_type}_{direction}"  # e.g. "Train_in", "Ship_out"
                flows[sk][d][key] = flows[sk][d].get(key, 0) + q

        # Train In (Unload at Store)
        aggregate_flow((df_log["event"] == "Unload") & (df_log["equipment"] == "Train"), "Train", "in")
        # Ship In (Unload at Store)
        aggregate_flow((df_log["event"].isin(["Unload", "ShipUnload"])) & (df_log["equipment"] == "Ship"), "Ship", "in")

        # Train Out (Load at Store)
        aggregate_flow((df_log["event"] == "Load") & (df_log["equipment"] == "Train"), "Train", "out")
        # Ship Out (Load at Store)
        aggregate_flow((df_log["event"].isin(["Load", "ShipLoad"])) & (df_log["equipment"] == "Ship"), "Ship", "out")

        # Production Output (Make) - material added TO store
        aggregate_flow((df_log["event"] == "Produce"), "Production", "in")
        
        # Production Consumption (Make) - material consumed FROM store for production
        aggregate_flow((df_log["event"] == "Produce"), "Consumption", "out")

    # 2b. Downtime Events by Equipment (unit_key) and Day
    downtime_by_equipment = {}  # unit_key -> {day -> {'Maintenance': hours, 'Breakdown': hours}}
    # Also track which equipment outputs to which stores
    equipment_to_stores = {}  # unit_key -> set of store_keys
    
    # Build equipment -> stores mapping from MAKES CONFIG (not just log events)
    # This ensures we capture the relationship even if equipment was down the whole time
    for make_unit in makes:
        loc = getattr(make_unit, 'location', '')
        equip = getattr(make_unit, 'equipment', '')
        if loc and equip:
            unit_key = f"{loc}|{equip}"
            if unit_key not in equipment_to_stores:
                equipment_to_stores[unit_key] = set()
            # Get output stores from candidates
            candidates = getattr(make_unit, 'candidates', []) or []
            for cand in candidates:
                out_key = getattr(cand, 'out_store_key', None)
                if out_key:
                    equipment_to_stores[unit_key].add(out_key)
                out_keys = getattr(cand, 'out_store_keys', []) or []
                for ok in out_keys:
                    if ok:
                        equipment_to_stores[unit_key].add(ok)
    
    if sim.action_log:
        df_log = pd.DataFrame(sim.action_log)
        df_log["day"] = (pd.to_numeric(df_log["time_h"], errors="coerce") / 24.0).astype(int) + 1
        
        # Also add equipment -> stores from production events (supplements config)
        production_events = df_log[(df_log["process"] == "Make") & (df_log["event"] == "Produce")]
        if not production_events.empty:
            for _, row in production_events.iterrows():
                loc = row.get("location", "")
                equip = row.get("equipment", "")
                to_store = row.get("to_store", "")
                if loc and equip and to_store:
                    unit_key = f"{loc}|{equip}"
                    if unit_key not in equipment_to_stores:
                        equipment_to_stores[unit_key] = set()
                    equipment_to_stores[unit_key].add(to_store)
        
        # Track downtime by equipment
        downtime_events = df_log[df_log["process"] == "Downtime"]
        if not downtime_events.empty:
            for _, row in downtime_events.iterrows():
                loc = row.get("location", "Unknown")
                equip = row.get("equipment", "Unknown")
                d = int(row["day"])
                event_type = row.get("event", "Unknown")
                hours = float(row.get("qty", 0) or 0)
                
                unit_key = f"{loc}|{equip}"
                if unit_key not in downtime_by_equipment:
                    downtime_by_equipment[unit_key] = {}
                if d not in downtime_by_equipment[unit_key]:
                    downtime_by_equipment[unit_key][d] = {"Maintenance": 0, "Breakdown": 0}
                
                if event_type in downtime_by_equipment[unit_key][d]:
                    downtime_by_equipment[unit_key][d][event_type] += hours
    
    # Build reverse mapping: store_key -> set of equipment unit_keys that supply it
    store_to_equipment = {}
    for unit_key, stores in equipment_to_stores.items():
        for store_key in stores:
            if store_key not in store_to_equipment:
                store_to_equipment[store_key] = set()
            store_to_equipment[store_key].add(unit_key)

    # 3. Transport Timelines (separate for Train and Ship)
    train_transport_fig = None
    ship_route_group_figs = []
    vessel_state_fig = None
    fleet_util_fig = None
    manufacturing_figs = {}
    if sim.action_log:
        df_log_all = pd.DataFrame(sim.action_log)
        train_transport_fig = _generate_transport_plot(df_log_all, equipment_type="Train")
        ship_route_group_figs = _generate_ship_timeline_by_route_group(df_log_all)
        vessel_state_fig = _generate_vessel_state_chart(df_log_all)
        fleet_util_fig = _generate_fleet_utilisation_chart(df_log_all)
        manufacturing_figs = _generate_manufacturing_charts(df_log_all)

    # 4. Store Figures
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
            
            # Production Output (bars behind everything else)
            if "Production_in" in data.columns and data["Production_in"].sum() > 0:
                fig.add_trace(go.Bar(x=data["day"], y=data["Production_in"], name="Production (t)",
                                     marker=dict(color="rgba(255, 127, 14, 0.4)"), width=0.8), secondary_y=True)

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
            
            # Level (on top so it's always visible)
            fig.add_trace(
                go.Scatter(x=data["day"], y=data["level"], name="Level (t)", line=dict(color="blue", width=2)),
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
                # Maintenance days - add shaded regions behind all other traces
                maint_days = [d for d in store_downtime if store_downtime[d].get("Maintenance", 0) > 0]
                for maint_day in maint_days:
                    fig.add_vrect(
                        x0=maint_day - 0.5, x1=maint_day + 0.5,
                        fillcolor="rgba(255, 152, 0, 0.15)",
                        layer="below",
                        line_width=0,
                    )
                # Add invisible trace for legend
                if maint_days:
                    fig.add_trace(go.Scatter(
                        x=[None], y=[None],
                        name="Maintenance",
                        mode='markers',
                        marker=dict(symbol='square', size=12, color='rgba(255, 152, 0, 0.4)'),
                        showlegend=True
                    ), secondary_y=False)
                
                # Breakdown days (full day = 24 hours) - add red shaded regions
                breakdown_days = [d for d in store_downtime if store_downtime[d].get("Breakdown", 0) >= 24]
                for bd_day in breakdown_days:
                    fig.add_vrect(
                        x0=bd_day - 0.5, x1=bd_day + 0.5,
                        fillcolor="rgba(244, 67, 54, 0.15)",
                        layer="below",
                        line_width=0,
                    )
                # Add invisible trace for legend
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

    # 5. Collect all products from inventory data for filters
    all_products = set()
    if not df_inv.empty:
        for sk in df_inv["store_key"].unique():
            parts = str(sk).split("|")
            if len(parts) >= 4:
                all_products.add(parts[3])

    # 6. HTML Generation
    content = []

    if train_transport_fig:
        content.append({"type": "fig", "fig": train_transport_fig, "title": "Rail Transport Timeline", "category": "shipping"})
    
    if fleet_util_fig:
        content.append({"type": "fig", "fig": fleet_util_fig, "title": "Ship Fleet Utilization", "category": "shipping"})
    
    if vessel_state_fig:
        content.append({"type": "fig", "fig": vessel_state_fig, "title": "Fleet State Over Time", "category": "shipping"})
    
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

    _generate_html_report(sim, out_dir, content, sorted(all_products), location_order)


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
            # Add shaded regions for maintenance days behind all other traces
            for maint_day in maint_days:
                fig.add_vrect(
                    x0=maint_day - 0.5, x1=maint_day + 0.5,
                    fillcolor="rgba(255, 152, 0, 0.2)",
                    layer="below",
                    line_width=0,
                )
            # Add invisible trace for legend
            if maint_days:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    name="Maintenance",
                    mode='markers',
                    marker=dict(symbol='square', size=12, color='rgba(255, 152, 0, 0.5)'),
                    showlegend=True
                ))
            
            # Breakdown days (full day = 24 hours) - add red shaded regions
            breakdown_days = [d for d in unit_dt if unit_dt[d].get('Breakdown', 0) >= 24]
            for bd_day in breakdown_days:
                fig.add_vrect(
                    x0=bd_day - 0.5, x1=bd_day + 0.5,
                    fillcolor="rgba(244, 67, 54, 0.2)",
                    layer="below",
                    line_width=0,
                )
            # Add invisible trace for legend
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

    vessel_states = {}
    max_day = int(state_changes['time_h'].max() / 24) + 1

    daily_util = []
    for day in range(1, max_day + 1):
        day_start = (day - 1) * 24
        day_end = day * 24

        day_events = state_changes[state_changes['time_h'] < day_end]
        for _, row in day_events.iterrows():
            vid = row.get('vessel_id')
            new_state = row.get('ship_state')
            if vid is not None and new_state is not None:
                vessel_states[vid] = new_state

        total_vessels = len(vessel_states)
        if total_vessels > 0:
            idle_count = sum(1 for v, st in vessel_states.items() if st == 'IDLE')
            utilized_count = total_vessels - idle_count
            util_pct = (utilized_count / total_vessels) * 100
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


def _generate_html_report(sim, out_dir: Path, content: list, products: list = None, locations: list = None):
    html_path = out_dir / "sim_outputs_plots_all.html"
    total_unmet = sum(sim.unmet.values())
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    products = products or []
    locations = locations or []

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
        .run-btn:disabled {{ background: #444; cursor: not-allowed; transform: none; box-shadow: none; }}
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
<body>
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
        <div class="summary"><p><strong>Generated:</strong> {run_time}</p><p><strong>Total Unmet Demand:</strong> <span style="color: {'red' if total_unmet > 0 else 'green'}">{total_unmet:,.0f} tons</span></p><p><strong>Status:</strong> Complete</p></div>
"""

    div_id = 0
    for item in content:
        category = item.get("category", "")
        product = item.get("product", "")
        location = item.get("location", "")
        if item["type"] == "header":
            html += f'<h2 class="section-header" data-category="{category}" data-location="{location}">Location: {location}</h2>'
        elif item["type"] == "fig" and item.get("fig"):
            title = item.get("title", "")
            if title:
                html += f'<h3 class="plot-title" data-category="{category}" data-product="{product}" data-location="{location}">{title}</h3>'
            html += f'<div id="plot-{div_id}" class="plot" data-category="{category}" data-product="{product}" data-location="{location}"></div><script>Plotly.newPlot("plot-{div_id}", {item["fig"].to_json()});</script>'
            div_id += 1

    html += """</div><div class="footer"><p>Generated by sim_run_report_plot.py</p></div>
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

    async function runSimulation() {
        const btn = document.getElementById('runBtn');
        const status = document.getElementById('status');
        btn.disabled = true;
        btn.textContent = 'Running...';
        status.textContent = 'Simulation in progress...';
        try {
            const response = await fetch('/run-simulation', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ horizon_days: 365, random_opening: true }) });
            const result = await response.json();
            if (result.success && result.report_ready) {
                status.textContent = 'Complete! Reloading...';
                setTimeout(() => window.location.reload(), 500);
            } else {
                status.textContent = 'Error: ' + (result.output || 'Unknown error');
                btn.disabled = false;
                btn.textContent = 'Run Single Simulation';
            }
        } catch (err) {
            status.textContent = 'Error: ' + err.message;
            btn.disabled = false;
            btn.textContent = 'Run Single Simulation';
        }
    }
    </script></body></html>"""

    html_path.write_text(html, encoding="utf-8")
    print(f"Interactive HTML report: {html_path}")