# sim_run_grok_helpers.py - FINAL FIXED - NO DUPLICATES, FULL YEAR, ZIG-ZAG
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
from sim_run_grok_config import config

def write_csv_outputs(sim, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Inventory snapshots with day number
    if sim.inventory_snapshots:
        df_snap = pd.DataFrame(sim.inventory_snapshots)
        df_snap["time_h"] = pd.to_numeric(df_snap["time_h"], errors="coerce")
        df_snap = df_snap.dropna(subset=["time_h"])
        df_snap["day"] = (df_snap["time_h"] / 24).astype(int) + 1
        df_snap = df_snap.sort_values(["time_h", "store_key"])
        df_snap.to_csv(out_dir / "sim_outputs_inventory_daily.csv", index=False)

    # Action log with day
    if sim.action_log:
        df_log = pd.DataFrame(sim.action_log)
        df_log["time_h"] = pd.to_numeric(df_log["time_h"], errors="coerce")
        df_log = df_log.dropna(subset=["time_h"])
        df_log["day"] = (df_log["time_h"] / 24).astype(int) + 1
        df_log["hour"] = (df_log["time_h"] // 1).astype(int) + 1
        cols = ["day", "hour", "time_h", "event"] + [c for c in df_log.columns if c not in ["day", "hour", "time_h", "event"]]
        df_log[cols].to_csv(out_dir / "sim_outputs_sim_log.csv", index=False)

    # Store levels
    ending = []
    for key, cont in sim.stores.items():
        parts = key.split("|")
        pc = parts[0] if len(parts) > 0 else ""
        loc = parts[1] if len(parts) > 1 else ""
        eq = parts[2] if len(parts) > 2 else ""
        inp = parts[3] if len(parts) > 3 else ""
        ending.append({
            "Store": key,
            "Product_Class": pc,
            "Location": loc,
            "Equipment": eq,
            "Input": inp,
            "Level": round(cont.level, 1),
            "Capacity": cont.capacity,
            "Fill_Pct": round(cont.level / cont.capacity, 3) if cont.capacity > 0 else 0
        })
    pd.DataFrame(ending).to_csv(out_dir / "sim_outputs_store_levels.csv", index=False)

    # Unmet demand
    unmet_rows = [{"Key": k, "Unmet": round(v, 1)} for k, v in sim.unmet.items()]
    pd.DataFrame(unmet_rows).to_csv(out_dir / "sim_outputs_unmet_demand.csv", index=False)

    print(f"CSV outputs written to {out_dir}")

def plot_results(sim, out_dir: Path):
    if not config.write_plots or not sim.inventory_snapshots:
        print("Skipping plots (disabled or no data)")
        return

    df = pd.DataFrame(sim.inventory_snapshots)
    df["time_h"] = pd.to_numeric(df["time_h"], errors="coerce")
    df = df.dropna(subset=["time_h"])
    df["day"] = (df["time_h"] / 24).astype(int) + 1  # Day 1 to 365
    
    # Aggregate to one row per day per store (use end-of-day values)
    df = df.sort_values(["store_key", "time_h"])
    df = df.groupby(["store_key", "day"]).last().reset_index()
    
    # Round values to 0 decimal places
    df["level"] = df["level"].round(0).astype(int)
    df["fill_pct"] = df["fill_pct"].round(2)

    # Secondary axis grouping
    capacities = df.groupby("store_key")["capacity"].first()
    large_threshold = capacities.median() * 5
    large_stores = capacities[capacities > large_threshold].index.tolist()
    small_stores = [s for s in df["store_key"].unique() if s not in large_stores]

    figs = []

    # Individual plots
    for store_key in df["store_key"].unique():
        data = df[df["store_key"] == store_key]

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(x=data["day"], y=data["level"], name="Level (tons)", line=dict(color="blue"),
                      hovertemplate="Day %{x}: %{y:,.0f} tons<extra></extra>"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=data["day"], y=(data["fill_pct"] * 100).round(0), name="Fill %", line=dict(color="green", dash="dot"),
                      hovertemplate="Day %{x}: %{y:.0f}%<extra></extra>"),
            secondary_y=True,
        )

        cap = data["capacity"].iloc[0]
        fig.update_layout(
            title=f"Inventory: {store_key}<br><sub>Capacity: {cap:,.0f} tons</sub>",
            xaxis_title="Day",
            yaxis_title="Tons",
            yaxis2_title="Fill %",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
            height=600
        )
        # Drop decimals on axis tick labels
        fig.update_yaxes(tickformat=",.0f", secondary_y=False)
        fig.update_yaxes(tickformat=",.0f", secondary_y=True)
        fig.update_xaxes(tick0=1, dtick=30, range=[1, 366])
        figs.append(fig)

    # Summary plot
    fig_summary = make_subplots(specs=[[{"secondary_y": True}]])

    for store_key in small_stores:
        data = df[df["store_key"] == store_key]
        fig_summary.add_trace(
            go.Scatter(x=data["day"], y=data["level"], name=store_key.split("|")[-1],
                      hovertemplate="%{y:,.0f} tons<extra></extra>"),
            secondary_y=False,
        )
    for store_key in large_stores:
        data = df[df["store_key"] == store_key]
        fig_summary.add_trace(
            go.Scatter(x=data["day"], y=data["level"], name=store_key.split("|")[-1] + " (large)",
                      hovertemplate="%{y:,.0f} tons<extra></extra>"),
            secondary_y=True,
        )

    fig_summary.update_layout(
        title="Summary: All Store Levels",
        xaxis_title="Day",
        yaxis_title="Tons (Small Stores)",
        yaxis2_title="Tons (Large Stores)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=700
    )
    # Drop decimals on axis tick labels for summary as well
    fig_summary.update_yaxes(tickformat=",.0f", secondary_y=False)
    fig_summary.update_yaxes(tickformat=",.0f", secondary_y=True)
    fig_summary.update_xaxes(tick0=1, dtick=30, range=[1, 366])
    figs.insert(0, fig_summary)

    generate_html_report(sim, out_dir, figs)

def generate_html_report(sim, out_dir: Path, figs: list):
    html_path = out_dir / "sim_outputs_plots_all.html"
    total_unmet = sum(sim.unmet.values())
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Cement Australia Simulation Report</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding-top: 70px; background: #f9f9f9; }}
        .sticky-header {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 12px 40px;
            z-index: 1000;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }}
        .sticky-header h1 {{
            color: #00d4ff;
            margin: 0;
            font-size: 1.4em;
        }}
        .run-btn {{
            background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
            color: #1a1a2e;
            border: none;
            padding: 12px 24px;
            font-size: 1em;
            font-weight: bold;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        .run-btn:hover {{
            transform: scale(1.05);
            box-shadow: 0 4px 15px rgba(0,212,255,0.4);
        }}
        .run-btn:disabled {{
            background: #666;
            cursor: not-allowed;
            transform: none;
        }}
        .run-btn.running {{
            animation: pulse 1.5s infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.6; }}
        }}
        .content {{ padding: 20px 40px; }}
        h2 {{ color: #2c3e50; }}
        .summary {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 30px; }}
        .plot {{ margin: 40px 0; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .footer {{ margin-top: 50px; color: #7f8c8d; font-size: 0.9em; text-align: center; }}
        #status {{ color: #ccc; font-size: 0.9em; margin-left: 15px; }}
    </style>
</head>
<body>
    <div class="sticky-header">
        <h1>Cement Australia Supply Chain Simulation</h1>
        <div>
            <button class="run-btn" id="runBtn" onclick="runSimulation()">Run New Simulation</button>
            <span id="status"></span>
        </div>
    </div>
    
    <div class="content">
        <div class="summary">
            <p><strong>Generated:</strong> <span id="genTime">{run_time}</span></p>
            <p><strong>Horizon:</strong> 365 days</p>
            <p><strong>Total Unmet Demand:</strong> <span id="unmetDemand">{total_unmet:,.0f}</span> tons</p>
            <p><strong>Stores Simulated:</strong> {len(sim.stores)}</p>
        </div>

        <h2>Interactive Inventory Charts (Day 1-365)</h2>
        <p>Hover - Zoom - Pan - Toggle legend</p>
"""

    for i, fig in enumerate(figs):
        div_id = f"plot-{i}"
        fig_html = fig.to_html(include_plotlyjs=False, full_html=False, div_id=div_id, config={'responsive': True})
        html += f'<div class="plot">{fig_html}</div>'

    html += """
    </div>
    <div class="footer">
        <p>Generated by sim_run_grok.py</p>
    </div>
    
    <script>
    async function runSimulation() {
        const btn = document.getElementById('runBtn');
        const status = document.getElementById('status');
        
        btn.disabled = true;
        btn.classList.add('running');
        btn.textContent = 'Running...';
        status.textContent = 'Simulation in progress...';
        
        try {
            const response = await fetch('/run-simulation', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ horizon_days: 365, random_opening: true })
            });
            
            const result = await response.json();
            
            if (result.success && result.report_ready) {
                status.textContent = 'Complete! Reloading...';
                setTimeout(() => window.location.reload(), 500);
            } else {
                status.textContent = 'Error: ' + (result.output || 'Unknown error');
                btn.disabled = false;
                btn.classList.remove('running');
                btn.textContent = 'Run New Simulation';
            }
        } catch (err) {
            status.textContent = 'Error: ' + err.message;
            btn.disabled = false;
            btn.classList.remove('running');
            btn.textContent = 'Run New Simulation';
        }
    }
    </script>
</body>
</html>
"""

    html_path.write_text(html, encoding="utf-8")
    print(f"Interactive HTML report (no duplicates, full year): {html_path}")

def generate_standalone(settings, stores, makes, moves, demands, out_dir: Path):
    code_lines = [
        "# sim_outputs_simpy_model.py - Clean standalone simulation model",
        f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "from sim_run_grok_core import SupplyChainSimulation, StoreConfig, ProductionCandidate, MakeUnit, TransportRoute, Demand",
        "",
        f"settings = {repr(settings)}",
        f"stores_cfg = {repr(stores)}",
        f"makes = {repr(makes)}",
        f"moves = {repr(moves)}",
        f"demands = {repr(demands)}",
        "",
        "if __name__ == '__main__':",
        "    sim = SupplyChainSimulation(settings)",
        "    sim.run(stores_cfg, makes, moves, demands)",
    ]
    standalone_path = out_dir / "sim_outputs_simpy_model.py"
    standalone_path.write_text("\n".join(code_lines), encoding="utf-8")
    print(f"Standalone model: {standalone_path}")