# sim_run_report_plot.py
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
from sim_run_config import config


def plot_results(sim, out_dir: Path, routes: list | None = None):
    if not config.write_plots: return

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

    # 3. Transport Timelines (separate for Train and Ship)
    train_transport_fig = None
    ship_transport_fig = None
    if sim.action_log:
        df_log_all = pd.DataFrame(sim.action_log)
        train_transport_fig = _generate_transport_plot(df_log_all, equipment_type="Train")
        ship_transport_fig = _generate_transport_plot(df_log_all, equipment_type="Ship")

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

            # Level
            fig.add_trace(
                go.Scatter(x=data["day"], y=data["level"], name="Level (t)", line=dict(color="blue", width=2)),
                secondary_y=False)

            # Demand
            if data["demand_per_day"].sum() > 0:
                fig.add_trace(go.Scatter(x=data["day"], y=data["demand_per_day"], name="Demand (t/d)",
                                         line=dict(color="green", dash="dot", width=1)), secondary_y=True)

            # --- FLOW TRACES (The missing piece!) ---
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

            # Ship Out
            if "Ship_out" in data.columns and data["Ship_out"].sum() > 0:
                fig.add_trace(go.Scatter(x=data["day"], y=data["Ship_out"], name="Ship Out (t)", mode='markers',
                                         marker=dict(symbol='triangle-up', size=10, color='#17becf')), secondary_y=True)

            # Production Output
            if "Production_in" in data.columns and data["Production_in"].sum() > 0:
                fig.add_trace(go.Scatter(x=data["day"], y=data["Production_in"], name="Production (t)",
                                         line=dict(color="#ff7f0e", dash="dash", width=1)), secondary_y=True)
            
            # Consumption (material consumed from this store for production)
            if "Consumption_out" in data.columns and data["Consumption_out"].sum() > 0:
                fig.add_trace(go.Scatter(x=data["day"], y=data["Consumption_out"], name="Consumption (t)",
                                         line=dict(color="#d62728", dash="dashdot", width=1.5)), secondary_y=True)

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

    # 5. HTML Generation
    content = []
    if train_transport_fig:
        content.append({"type": "fig", "fig": train_transport_fig, "title": "Rail Transport Timeline"})
    if ship_transport_fig:
        content.append({"type": "fig", "fig": ship_transport_fig, "title": "Ship Transport Timeline"})

    if not df_inv.empty:
        df_inv["location_only"] = df_inv["store_key"].apply(
            lambda sk: str(sk).split("|")[1] if "|" in str(sk) else "Other")
        for loc in sorted(df_inv["location_only"].unique()):
            content.append({"type": "header", "location": loc})
            loc_stores = df_inv[df_inv["location_only"] == loc]["store_key"].unique()
            for sk in sorted(loc_stores):
                if sk in store_figs:
                    content.append({"type": "fig", "fig": store_figs[sk]})

    _generate_html_report(sim, out_dir, content)


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


def _generate_html_report(sim, out_dir: Path, content: list):
    html_path = out_dir / "sim_outputs_plots_all.html"
    total_unmet = sum(sim.unmet.values())
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Supply Chain Report</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        body {{ font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; padding-top: 70px; background: #f9f9f9; }}
        .sticky-header {{ position: fixed; top: 0; left: 0; right: 0; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 12px 40px; z-index: 1000; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 2px 10px rgba(0,0,0,0.3); }}
        .sticky-header h1 {{ color: #00d4ff; margin: 0; font-size: 1.4em; }}
        .run-btn {{ background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%); color: #1a1a2e; border: none; padding: 12px 24px; font-size: 1em; font-weight: bold; border-radius: 8px; cursor: pointer; transition: all 0.3s ease; }}
        .run-btn:hover {{ transform: scale(1.05); box-shadow: 0 4px 15px rgba(0,212,255,0.4); }}
        .run-btn:disabled {{ background: #666; cursor: not-allowed; }}
        #status {{ color: #ccc; font-size: 0.9em; margin-left: 15px; }}
        .content {{ padding: 20px 40px; }}
        .summary {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 30px; border-left: 5px solid #00d4ff; }}
        .plot {{ margin: 20px 0; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        h2 {{ color: #2c3e50; margin-top: 40px; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
        h3 {{ color: #34495e; margin-top: 30px; font-size: 1.1em; }}
        .footer {{ margin-top: 50px; color: #7f8c8d; font-size: 0.9em; text-align: center; padding-bottom: 20px; }}
    </style>
</head>
<body>
    <div class="sticky-header">
        <h1>Cement Australia Supply Chain Simulation</h1>
        <div><button class="run-btn" id="runBtn" onclick="runSimulation()">Run New Simulation</button><span id="status"></span></div>
    </div>
    <div class="content">
        <div class="summary"><p><strong>Generated:</strong> {run_time}</p><p><strong>Total Unmet Demand:</strong> <span style="color: {'red' if total_unmet > 0 else 'green'}">{total_unmet:,.0f} tons</span></p><p><strong>Status:</strong> Complete</p></div>
"""

    div_id = 0
    for item in content:
        if item["type"] == "header":
            html += f"<h2>Location: {item['location']}</h2>"
        elif item["type"] == "fig" and item.get("fig"):
            title = item.get("title", "")
            if title: html += f"<h3>{title}</h3>"
            html += f'<div id="plot-{div_id}" class="plot"></div><script>Plotly.newPlot("plot-{div_id}", {item["fig"].to_json()});</script>'
            div_id += 1

    html += """</div><div class=\"footer\"><p>Generated by sim_run_report_plot.py</p></div>
    <script>
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
                btn.textContent = 'Run New Simulation';
            }
        } catch (err) {
            status.textContent = 'Error: ' + err.message;
            btn.disabled = false;
            btn.textContent = 'Run New Simulation';
        }
    }
    </script></body></html>"""

    html_path.write_text(html, encoding="utf-8")
    print(f"Interactive HTML report: {html_path}")