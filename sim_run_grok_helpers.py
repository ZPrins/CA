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
        # Round known quantity fields to 2 decimals if present
        for qc in ["qty", "qty_t", "unmet"]:
            if qc in df_log.columns:
                df_log[qc] = pd.to_numeric(df_log[qc], errors="coerce").round(2)
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

    # Unmet demand (rounded to 2 decimals)
    unmet_rows = [{"Key": k, "Unmet": round(float(v), 2)} for k, v in sim.unmet.items()]
    pd.DataFrame(unmet_rows).to_csv(out_dir / "sim_outputs_unmet_demand.csv", index=False)

    print(f"CSV outputs written to {out_dir}")

def plot_results(sim, out_dir: Path, routes: list | None = None):
    if not config.write_plots or not sim.inventory_snapshots:
        print("Skipping plots (disabled or no data)")
        return

    # Build a supplier map: for each destination store_key, list of origin store_keys supplying it
    supplier_map: dict[str, list[str]] = {}
    if routes:
        try:
            # Use sets during accumulation to avoid duplicates, then convert to sorted lists
            tmp: dict[str, set[str]] = {}
            for r in routes:
                # Each route may have multiple origin/dest store keys
                try:
                    origins = getattr(r, 'origin_stores', []) or []
                    dests = getattr(r, 'dest_stores', []) or []
                    for d in dests:
                        s = tmp.setdefault(str(d), set())
                        for o in origins:
                            s.add(str(o))
                except Exception:
                    continue
            # Finalize into sorted lists for stable titles
            for d, s in tmp.items():
                supplier_map[d] = sorted(list(s))
        except Exception:
            supplier_map = {}

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

    # Attach demand per time unit for each store (tons per configured demand step)
    try:
        rate_map = getattr(sim, 'demand_rate_map', {}) or {}
    except Exception:
        rate_map = {}
    # Align demand display resolution to daily buckets used by the graphs
    daily_factor = 24.0  # hours per day
    df["demand_per_day"] = df["store_key"].map(lambda k: float(rate_map.get(str(k), 0.0)) * daily_factor).round(1)

    # Compute production per day by store from action_log (Produced events)
    prod_per_day = None
    rail_in_per_day = None
    ship_in_per_day = None
    rail_out_per_day = None
    ship_out_per_day = None
    try:
        if sim.action_log:
            log_df = pd.DataFrame(sim.action_log)
            if not log_df.empty:
                log_df["time_h"] = pd.to_numeric(log_df["time_h"], errors="coerce")
                log_df = log_df.dropna(subset=["time_h"]) 
                log_df["day"] = (log_df["time_h"] / 24).astype(int) + 1
                # We require to_store_key on Produced events (added in core). Fallback: try infer from text if missing
                mask = (log_df["event"] == "Produced")
                cols = [c for c in ["to_store_key", "qty"] if c in log_df.columns]
                if mask.any() and set(cols) >= {"qty"}:
                    # If to_store_key is missing, skip those rows to avoid misattribution
                    if "to_store_key" in log_df.columns:
                        prod_df = log_df.loc[mask & log_df["to_store_key"].notna(), ["to_store_key", "day", "qty"]].copy()
                        prod_df["qty"] = pd.to_numeric(prod_df["qty"], errors="coerce").fillna(0.0)
                        prod_df["qty"] = prod_df["qty"].round(1)
                        prod_per_day = prod_df.groupby(["to_store_key", "day"])['qty'].sum().reset_index()
                # Compute inbound by mode (TRAIN/SHIP) from Unloaded events
                if "event" in log_df.columns and "to_store" in log_df.columns and "qty" in log_df.columns:
                    unl = log_df[log_df["event"] == "Unloaded"].copy()
                    if not unl.empty:
                        unl["qty"] = pd.to_numeric(unl["qty"], errors="coerce").fillna(0.0).round(1)
                        # Normalize mode column presence
                        mode_col = "mode" if "mode" in unl.columns else None
                        if mode_col:
                            rail = unl[unl[mode_col].str.upper() == "TRAIN"] if unl[mode_col].dtype == object else unl[unl[mode_col] == "TRAIN"]
                            ship = unl[unl[mode_col].str.upper() == "SHIP"] if unl[mode_col].dtype == object else unl[unl[mode_col] == "SHIP"]
                            if not rail.empty:
                                rail_in_per_day = rail.groupby(["to_store", "day"])['qty'].sum().reset_index()
                            if not ship.empty:
                                ship_in_per_day = ship.groupby(["to_store", "day"])['qty'].sum().reset_index()
                # Compute outbound by mode (TRAIN/SHIP) from Loaded events
                if "event" in log_df.columns and "from_store" in log_df.columns and "qty" in log_df.columns:
                    lod = log_df[log_df["event"] == "Loaded"].copy()
                    if not lod.empty:
                        lod["qty"] = pd.to_numeric(lod["qty"], errors="coerce").fillna(0.0).round(1)
                        mode_col2 = "mode" if "mode" in lod.columns else None
                        if mode_col2:
                            rail_o = lod[lod[mode_col2].str.upper() == "TRAIN"] if lod[mode_col2].dtype == object else lod[lod[mode_col2] == "TRAIN"]
                            ship_o = lod[lod[mode_col2].str.upper() == "SHIP"] if lod[mode_col2].dtype == object else lod[lod[mode_col2] == "SHIP"]
                            if not rail_o.empty:
                                rail_out_per_day = rail_o.groupby(["from_store", "day"])['qty'].sum().reset_index()
                            if not ship_o.empty:
                                ship_out_per_day = ship_o.groupby(["from_store", "day"])['qty'].sum().reset_index()
    except Exception:
        prod_per_day = None
        rail_in_per_day = None
        ship_in_per_day = None
        rail_out_per_day = None
        ship_out_per_day = None

    # Secondary axis grouping
    capacities = df.groupby("store_key")["capacity"].first()
    large_threshold = capacities.median() * 5
    large_stores = capacities[capacities > large_threshold].index.tolist()
    small_stores = [s for s in df["store_key"].unique() if s not in large_stores]

    # Build individual figs and index them by store_key
    store_figs: dict[str, go.Figure] = {}

    for store_key in df["store_key"].unique():
        data = df[df["store_key"] == store_key]

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(x=data["day"], y=data["level"], name="Level (tons)", line=dict(color="blue"),
                      hovertemplate="Day %{x}: %{y:,.0f} tons<extra></extra>"),
            secondary_y=False,
        )
        # Secondary axis series are rounded to 1 decimal and ticked to 1 decimal
        fig.add_trace(
            go.Scatter(x=data["day"], y=data.get("demand_per_day", pd.Series([0]*len(data))).round(1), name="Demand per Day (tons)", line=dict(color="green", dash="dot"),
                      hovertemplate="Day %{x}: %{y:,.1f} tons<extra></extra>"),
            secondary_y=True,
        )
        # Add Production per Day (tons) if available for this store
        if prod_per_day is not None:
            try:
                prod_series = prod_per_day[prod_per_day["to_store_key"] == store_key][["day", "qty"]]
                if not prod_series.empty:
                    # Merge to ensure alignment with existing days
                    merged = pd.merge(data[["day"]], prod_series, on="day", how="left")
                    y_prod = merged["qty"].fillna(0.0).round(1)
                    fig.add_trace(
                        go.Scatter(x=data["day"], y=y_prod, name="Production per Day (tons)", line=dict(color="#ff7f0e", dash="dash"),
                                  hovertemplate="Day %{x}: %{y:,.1f} tons<extra></extra>"),
                        secondary_y=True,
                    )
            except Exception:
                pass
        # Add Rail In (tons) if any TRAIN unloads to this store
        if rail_in_per_day is not None:
            try:
                rail_series = rail_in_per_day[rail_in_per_day["to_store"] == store_key][["day", "qty"]]
                if not rail_series.empty:
                    merged = pd.merge(data[["day"]], rail_series, on="day", how="left")
                    y_rail = merged["qty"].fillna(0.0).round(1)
                    fig.add_trace(
                        go.Scatter(x=data["day"], y=y_rail, name="Rail In (tons)", line=dict(color="#6a3d9a", dash="dot"),
                                   hovertemplate="Day %{x}: %{y:,.1f} tons<extra></extra>"),
                        secondary_y=True,
                    )
            except Exception:
                pass
        # Add Ship In (tons) if any SHIP unloads to this store
        if ship_in_per_day is not None:
            try:
                ship_series = ship_in_per_day[ship_in_per_day["to_store"] == store_key][["day", "qty"]]
                if not ship_series.empty:
                    merged = pd.merge(data[["day"]], ship_series, on="day", how="left")
                    y_ship = merged["qty"].fillna(0.0).round(1)
                    fig.add_trace(
                        go.Scatter(x=data["day"], y=y_ship, name="Ship In (tons)", line=dict(color="#1f78b4", dash="dashdot"),
                                   hovertemplate="Day %{x}: %{y:,.1f} tons<extra></extra>"),
                        secondary_y=True,
                    )
            except Exception:
                pass
        # Add Rail Out (tons) if any TRAIN loads from this store
        if rail_out_per_day is not None:
            try:
                rail_o_series = rail_out_per_day[rail_out_per_day["from_store"] == store_key][["day", "qty"]]
                if not rail_o_series.empty:
                    merged = pd.merge(data[["day"]], rail_o_series, on="day", how="left")
                    y_rail_o = merged["qty"].fillna(0.0).round(1)
                    fig.add_trace(
                        go.Scatter(x=data["day"], y=y_rail_o, name="Rail Out (tons)", line=dict(color="#8c564b", dash="solid"),
                                   hovertemplate="Day %{x}: %{y:,.1f} tons<extra></extra>"),
                        secondary_y=True,
                    )
            except Exception:
                pass
        # Add Ship Out (tons) if any SHIP loads from this store
        if ship_out_per_day is not None:
            try:
                ship_o_series = ship_out_per_day[ship_out_per_day["from_store"] == store_key][["day", "qty"]]
                if not ship_o_series.empty:
                    merged = pd.merge(data[["day"]], ship_o_series, on="day", how="left")
                    y_ship_o = merged["qty"].fillna(0.0).round(1)
                    fig.add_trace(
                        go.Scatter(x=data["day"], y=y_ship_o, name="Ship Out (tons)", line=dict(color="#17becf", dash="longdash"),
                                   hovertemplate="Day %{x}: %{y:,.1f} tons<extra></extra>"),
                        secondary_y=True,
                    )
            except Exception:
                pass

        cap = data["capacity"].iloc[0]
        # Compose title with supplier info if available
        suppliers = supplier_map.get(store_key, []) if 'supplier_map' in locals() else []
        if suppliers:
            supplied_from_text = "; ".join(suppliers)
            title_html = f"Inventory: {store_key}<br><sub>Supplied From: {supplied_from_text}</sub><br><sub>Capacity: {cap:,.0f} tons</sub>"
        else:
            title_html = f"Inventory: {store_key}<br><sub>Capacity: {cap:,.0f} tons</sub>"
        fig.update_layout(
            title=title_html,
            xaxis_title="Day",
            yaxis_title="Tons",
            yaxis2_title="Per Day (tons)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
            height=600
        )
        # Tick formats: primary axis integer tons; secondary axis 1 decimal; both zero-based ranges
        fig.update_yaxes(tickformat=",.0f", rangemode="tozero", secondary_y=False)
        fig.update_yaxes(tickformat=",.1f", rangemode="tozero", secondary_y=True)
        # Day axis: keep nice ticks but let it autoscale by default
        fig.update_xaxes(tick0=1, dtick=30, autorange=True)
        store_figs[store_key] = fig

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
    # Drop decimals on axis tick labels for summary as well and force zero-based ranges
    fig_summary.update_yaxes(tickformat=",.0f", rangemode="tozero", secondary_y=False)
    fig_summary.update_yaxes(tickformat=",.0f", rangemode="tozero", secondary_y=True)
    fig_summary.update_xaxes(tick0=1, dtick=30, autorange=True)

    # Build structured content grouped by location
    content = []
    content.append({"type": "summary", "fig": fig_summary})

    # Derive location per store_key
    df["location_only"] = df["store_key"].apply(lambda sk: (str(sk).split("|")[1] if isinstance(sk, str) and "|" in str(sk) else ""))
    for loc in sorted(df["location_only"].unique(), key=lambda x: (x == "", x)):
        loc_stores = [sk for sk in df[df["location_only"] == loc]["store_key"].unique()]
        content.append({"type": "header", "location": loc or "(no location)"})
        for sk in sorted(loc_stores):
            fig = store_figs.get(sk)
            if fig is not None:
                content.append({"type": "fig", "fig": fig, "store_key": sk, "location": loc})

    generate_html_report(sim, out_dir, content)

def generate_html_report(sim, out_dir: Path, content: list):
    html_path = out_dir / "sim_outputs_plots_all.html"
    total_unmet = sum(sim.unmet.values())
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset=\"utf-8\">
    <meta http-equiv=\"Content-Type\" content=\"text/html; charset=utf-8\" />
    <title>Cement Australia Simulation Report</title>
    <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
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
        h3 {{ color: #34495e; margin-top: 30px; }}
        .summary {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 30px; }}
        .plot {{ margin: 20px 0; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .footer {{ margin-top: 50px; color: #7f8c8d; font-size: 0.9em; text-align: center; }}
        #status {{ color: #ccc; font-size: 0.9em; margin-left: 15px; }}
    </style>
</head>
<body>
    <div class=\"sticky-header\">
        <h1>Cement Australia Supply Chain Simulation</h1>
        <div>
            <button class=\"run-btn\" id=\"runBtn\" onclick=\"runSimulation()\">Run New Simulation</button>
            <span id=\"status\"></span>
        </div>
    </div>
    
    <div class=\"content\">
        <div class=\"summary\">\n            <p><strong>Generated:</strong> <span id=\"genTime\">{run_time}</span></p>\n            <p><strong>Horizon:</strong> 365 days</p>\n            <p><strong>Total Unmet Demand:</strong> <span id=\"unmetDemand\">{total_unmet:,.0f}</span> tons</p>\n            <p><strong>Stores Simulated:</strong> {len(sim.stores)}</p>\n        </div>

        <h2>Interactive Inventory Charts (Day 1-365)</h2>
        <p>Hover - Zoom - Pan - Toggle legend</p>
"""

    # Render structured content: summary plot first, then groups by location
    plot_index = 0
    for item in content:
        typ = item.get("type") if isinstance(item, dict) else None
        if typ == "header":
            loc = item.get("location", "")
            html += f"<h3>Location: {loc}</h3>\n"
        elif typ in ("fig", "summary"):
            fig = item.get("fig")
            if fig is None:
                continue
            div_id = f"plot-{plot_index}"
            plot_index += 1
            fig_html = fig.to_html(include_plotlyjs=False, full_html=False, div_id=div_id, config={'responsive': True})
            html += f'<div class="plot">{fig_html}</div>'

    html += """
    </div>
    <div class=\"footer\">\n        <p>Generated by sim_run_grok.py</p>\n    </div>
    
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
    from pprint import pformat
    import math
    ts = datetime.now().strftime('%Y-%m-%d %H:%M')

    def _is_nan(x):
        try:
            return isinstance(x, float) and math.isnan(x)
        except Exception:
            return False

    # Sanitize data to avoid emitting bare `nan` in generated Python
    try:
        # Settings: replace any NaN numeric with 0.0
        if isinstance(settings, dict):
            for k, v in list(settings.items()):
                if _is_nan(v):
                    settings[k] = 0.0
        # Stores
        for s in stores:
            try:
                if _is_nan(getattr(s, 'capacity', None)):
                    s.capacity = 0.0
                if _is_nan(getattr(s, 'opening_low', None)):
                    s.opening_low = 0.0
                if _is_nan(getattr(s, 'opening_high', None)):
                    s.opening_high = 0.0
                # Clamp openings between 0 and capacity
                cap = float(getattr(s, 'capacity', 0.0) or 0.0)
                s.opening_low = max(0.0, min(float(getattr(s, 'opening_low', 0.0) or 0.0), cap))
                s.opening_high = max(0.0, min(float(getattr(s, 'opening_high', 0.0) or 0.0), cap))
            except Exception:
                pass
        # Makes and candidates
        for m in makes:
            try:
                if _is_nan(getattr(m, 'step_hours', None)):
                    m.step_hours = 1.0
                for c in getattr(m, 'candidates', []) or []:
                    try:
                        if _is_nan(getattr(c, 'rate_tph', None)):
                            c.rate_tph = 0.0
                        if _is_nan(getattr(c, 'consumption_pct', None)) or getattr(c, 'consumption_pct', None) is None:
                            c.consumption_pct = 1.0
                    except Exception:
                        pass
            except Exception:
                pass
        # Moves
        for mv in moves:
            try:
                if _is_nan(getattr(mv, 'payload_t', None)):
                    mv.payload_t = 0.0
                if _is_nan(getattr(mv, 'load_rate_tph', None)):
                    mv.load_rate_tph = 500.0
                if _is_nan(getattr(mv, 'unload_rate_tph', None)):
                    mv.unload_rate_tph = 400.0
                if _is_nan(getattr(mv, 'to_min', None)):
                    mv.to_min = 0.0
                if _is_nan(getattr(mv, 'back_min', None)):
                    mv.back_min = 0.0
                # n_units should be int >=1
                try:
                    n = int(getattr(mv, 'n_units', 1) or 1)
                    if n < 1:
                        n = 1
                    mv.n_units = n
                except Exception:
                    mv.n_units = 1
            except Exception:
                pass
        # Demands
        for d in demands:
            try:
                if _is_nan(getattr(d, 'rate_per_hour', None)):
                    d.rate_per_hour = 0.0
            except Exception:
                pass
    except Exception:
        # Best-effort cleanup; continue even if sanitization partially fails
        pass

    def fmt(obj):
        return pformat(obj, width=100, indent=2, sort_dicts=False)

    code_lines = [
        '"""',
        'sim_outputs_simpy_model.py — Standalone Grok simulation model',
        f'Generated on {ts}',
        '',
        'Auto-generated from sim_run_grok.py. Edit your source data/workbook, not this file.',
        'This file is intended to be easy to read: inputs are grouped into clear sections',
        'with Python-literal formatting for booleans (True/False) and None.',
        '"""',
        '',
        '# Runtime dependencies (no business logic here — delegated to core):',
        'from sim_run_grok_core import SupplyChainSimulation, StoreConfig, ProductionCandidate, MakeUnit, TransportRoute, Demand',
        '',
        '# ---------------------------------------------------------------------------',
        '# SETTINGS — global knobs for this run',
        '# ---------------------------------------------------------------------------',
        f'SETTINGS = {fmt(settings)}',
        '',
        '# ---------------------------------------------------------------------------',
        '# STORES — each entry describes a tank/node in the network',
        '# ---------------------------------------------------------------------------',
        f'STORES = {fmt(stores)}',
        '',
        '# ---------------------------------------------------------------------------',
        '# MAKES — production units with their candidate outputs and consumption rules',
        '# ---------------------------------------------------------------------------',
        f'MAKES = {fmt(makes)}',
        '',
        '# ---------------------------------------------------------------------------',
        '# MOVES — transport routes with timing and capacity',
        '# ---------------------------------------------------------------------------',
        f'MOVES = {fmt(moves)}',
        '',
        '# ---------------------------------------------------------------------------',
        '# DEMANDS — consumer sinks pulling from stores each step',
        '# ---------------------------------------------------------------------------',
        f'DEMANDS = {fmt(demands)}',
        '',
        'def main():',
        '    # Build and run the simulation with the inputs above',
        '    sim = SupplyChainSimulation(SETTINGS)',
        '    sim.run(STORES, MAKES, MOVES, DEMANDS)',
        '',
        "if __name__ == '__main__':",
        '    main()',
    ]
    standalone_path = out_dir / 'sim_outputs_simpy_model.py'
    standalone_path.write_text('\n'.join(code_lines), encoding='utf-8')
    print(f'Standalone model: {standalone_path}')