# sim_run_report_variability.py
"""
Variability Distribution Report Generator

Generates a separate HTML page showing actual distribution data from
simulation runs - breakdown events, berth waiting times, and opening stocks.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sim_run_types import StoreConfig, MakeUnit


def collect_variability_data(sim, store_configs: list, makes: list, settings: dict) -> dict:
    """
    Collect actual variability data from the simulation run.
    
    Returns dict with actual simulation data for variability analysis.
    """
    df_log = pd.DataFrame(sim.action_log) if sim.action_log else pd.DataFrame()
    
    variability = {
        'settings': {
            'random_opening': settings.get('random_opening', True),
            'random_seed': settings.get('random_seed'),
            'horizon_days': settings.get('horizon_days', 365)
        },
        'breakdown_events': [],
        'downtime_by_equipment': {},
        'berth_waiting_events': [],
        'opening_stocks': [],
        'configured_downtime': []
    }
    
    # Collect opening stock values (actual values used in simulation)
    for cfg in store_configs:
        actual_opening = getattr(cfg, '_actual_opening', None)
        if actual_opening is None:
            actual_opening = (cfg.opening_low + cfg.opening_high) / 2
        variability['opening_stocks'].append({
            'store_key': cfg.key,
            'opening_low': cfg.opening_low,
            'opening_high': cfg.opening_high,
            'actual_value': actual_opening,
            'capacity': cfg.capacity
        })
    
    # Collect configured downtime percentages
    for make in makes:
        if make.unplanned_downtime_pct and make.unplanned_downtime_pct > 0:
            variability['configured_downtime'].append({
                'equipment': make.equipment,
                'location': make.location,
                'downtime_pct': make.unplanned_downtime_pct * 100
            })
    
    if df_log.empty:
        return variability
    
    # Parse breakdown events - prefer BreakdownStart (has duration in qty) or fall back to grouping
    breakdown_start_rows = df_log[(df_log['process'] == 'Downtime') & (df_log['event'] == 'BreakdownStart')]
    
    if not breakdown_start_rows.empty:
        for _, row in breakdown_start_rows.iterrows():
            equip = row['equipment']
            duration = row['qty'] if pd.notna(row['qty']) and row['qty'] > 0 else 1
            
            variability['breakdown_events'].append({
                'equipment': equip,
                'start_time': row['time_h'],
                'duration_hours': int(duration)
            })
            
            if equip not in variability['downtime_by_equipment']:
                variability['downtime_by_equipment'][equip] = {'count': 0, 'total_hours': 0}
            variability['downtime_by_equipment'][equip]['count'] += 1
            variability['downtime_by_equipment'][equip]['total_hours'] += int(duration)
    else:
        breakdown_rows = df_log[(df_log['process'] == 'Downtime') & (df_log['event'] == 'Breakdown')]
        if not breakdown_rows.empty:
            for equip in breakdown_rows['equipment'].unique():
                equip_rows = breakdown_rows[breakdown_rows['equipment'] == equip].sort_values('time_h')
                times = equip_rows['time_h'].tolist()
                
                if equip not in variability['downtime_by_equipment']:
                    variability['downtime_by_equipment'][equip] = {'count': 0, 'total_hours': 0}
                
                i = 0
                while i < len(times):
                    start = times[i]
                    duration = 1
                    while i + 1 < len(times) and times[i + 1] - times[i] <= 1.5:
                        duration += 1
                        i += 1
                    variability['breakdown_events'].append({
                        'equipment': equip,
                        'start_time': start,
                        'duration_hours': duration
                    })
                    variability['downtime_by_equipment'][equip]['count'] += 1
                    variability['downtime_by_equipment'][equip]['total_hours'] += duration
                    i += 1
    
    return variability


def generate_variability_report(variability: dict, out_dir: Path) -> Path:
    """
    Generate an HTML report with actual distribution data from simulation.
    Uses light theme to match main report style.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    html_parts = []
    
    html_parts.append(f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Variability Analysis</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f8fafc;
            color: #1e293b;
            min-height: 100vh;
            padding: 24px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ 
            text-align: center; 
            margin-bottom: 8px;
            font-size: 28px;
            color: #0f172a;
        }}
        .subtitle {{
            text-align: center;
            color: #64748b;
            margin-bottom: 24px;
            font-size: 14px;
        }}
        .back-link {{
            display: inline-block;
            margin-bottom: 20px;
            color: #2563eb;
            text-decoration: none;
            font-size: 14px;
        }}
        .back-link:hover {{ text-decoration: underline; }}
        .section {{
            background: #ffffff;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .section-title {{
            font-size: 18px;
            font-weight: 600;
            color: #0f172a;
            margin-bottom: 8px;
            padding-bottom: 8px;
            border-bottom: 2px solid #e2e8f0;
        }}
        .section-desc {{
            color: #64748b;
            font-size: 13px;
            margin-bottom: 16px;
            line-height: 1.5;
        }}
        .plot-container {{
            background: #f8fafc;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
            border: 1px solid #e2e8f0;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 12px;
            margin-bottom: 16px;
        }}
        .stat-card {{
            background: #f1f5f9;
            border-radius: 8px;
            padding: 12px;
            text-align: center;
            border: 1px solid #e2e8f0;
        }}
        .stat-value {{ font-size: 24px; font-weight: 600; color: #2563eb; }}
        .stat-label {{ font-size: 11px; color: #64748b; margin-top: 4px; }}
        .no-data {{ 
            color: #94a3b8; 
            font-style: italic; 
            padding: 40px 20px; 
            text-align: center; 
            background: #f8fafc;
            border-radius: 8px;
        }}
        .param-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            margin-bottom: 16px;
        }}
        .param-table th {{
            text-align: left;
            padding: 10px 12px;
            background: #f1f5f9;
            color: #475569;
            border-bottom: 2px solid #e2e8f0;
            font-weight: 600;
        }}
        .param-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid #e2e8f0;
        }}
        .param-table tr:hover td {{ background: #f8fafc; }}
    </style>
</head>
<body>
    <div class="container">
        <a href="/outputs/sim_outputs_plots_all.html" class="back-link">&larr; Back to Interactive Report</a>
        <h1>Variability Analysis</h1>
        <p class="subtitle">Actual simulation data | Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Horizon: {variability['settings']['horizon_days']} days</p>
''')
    
    # Section 1: Breakdown Events by Equipment
    breakdown_events = variability.get('breakdown_events', [])
    downtime_by_equip = variability.get('downtime_by_equipment', {})
    
    if breakdown_events:
        df_events = pd.DataFrame(breakdown_events)
        
        total_events = len(breakdown_events)
        avg_duration = df_events['duration_hours'].mean()
        max_duration = df_events['duration_hours'].max()
        total_downtime = df_events['duration_hours'].sum()
        
        fig_breakdown = go.Figure()
        fig_breakdown.add_trace(go.Histogram(
            x=df_events['duration_hours'],
            nbinsx=20,
            name='Breakdown Durations',
            marker_color='#f97316'
        ))
        
        fig_breakdown.add_vline(x=avg_duration, line_dash="dash", line_color="#2563eb",
                               annotation_text=f"Avg: {avg_duration:.1f}h", annotation_position="top right")
        
        fig_breakdown.update_layout(
            xaxis_title='Duration (hours)',
            yaxis_title='Frequency',
            template='plotly_white',
            height=300,
            margin=dict(l=60, r=40, t=30, b=60),
            font=dict(size=12)
        )
        
        breakdown_html = fig_breakdown.to_html(full_html=False, include_plotlyjs=False)
        
        equip_names = list(downtime_by_equip.keys())
        equip_counts = [downtime_by_equip[e]['count'] for e in equip_names]
        equip_hours = [downtime_by_equip[e]['total_hours'] for e in equip_names]
        
        fig_by_equip = make_subplots(specs=[[{"secondary_y": True}]])
        fig_by_equip.add_trace(
            go.Bar(x=equip_names, y=equip_counts, name='Event Count', marker_color='#2563eb'),
            secondary_y=False
        )
        fig_by_equip.add_trace(
            go.Scatter(x=equip_names, y=equip_hours, mode='lines+markers', name='Total Hours', 
                      line=dict(color='#f97316', width=2), marker=dict(size=8)),
            secondary_y=True
        )
        fig_by_equip.update_layout(
            template='plotly_white',
            height=300,
            margin=dict(l=60, r=60, t=30, b=80),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            font=dict(size=12)
        )
        fig_by_equip.update_xaxes(tickangle=45)
        fig_by_equip.update_yaxes(title_text='Number of Events', secondary_y=False)
        fig_by_equip.update_yaxes(title_text='Total Downtime (hours)', secondary_y=True)
        
        by_equip_html = fig_by_equip.to_html(full_html=False, include_plotlyjs=False)
        
        html_parts.append(f'''
        <div class="section">
            <div class="section-title">1. Breakdown Events (Actual from Simulation)</div>
            <div class="section-desc">
                Distribution of <strong>actual breakdown events</strong> that occurred during the simulation run.
                Each event represents an unplanned equipment stoppage.
            </div>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{total_events}</div>
                    <div class="stat-label">Total Events</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{avg_duration:.1f}h</div>
                    <div class="stat-label">Avg Duration</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{max_duration:.0f}h</div>
                    <div class="stat-label">Max Duration</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{total_downtime:.0f}h</div>
                    <div class="stat-label">Total Downtime</div>
                </div>
            </div>
            <div class="plot-container">
                <strong>Duration Distribution</strong>
                {breakdown_html}
            </div>
            <div class="plot-container">
                <strong>Events by Equipment</strong>
                {by_equip_html}
            </div>
        </div>
''')
    else:
        html_parts.append('''
        <div class="section">
            <div class="section-title">1. Breakdown Events</div>
            <div class="no-data">No breakdown events occurred during this simulation run.</div>
        </div>
''')
    
    # Section 2: Configured Downtime Reference
    configured = variability.get('configured_downtime', [])
    if configured:
        table_rows = ""
        for c in configured:
            table_rows += f"<tr><td>{c['equipment']}</td><td>{c['location']}</td><td>{c['downtime_pct']:.1f}%</td></tr>"
        
        html_parts.append(f'''
        <div class="section">
            <div class="section-title">2. Configured Downtime Targets</div>
            <div class="section-desc">
                Reference table showing the <strong>target unplanned downtime percentage</strong> configured for each equipment.
                The simulation uses these targets to probabilistically trigger breakdowns.
            </div>
            <table class="param-table">
                <tr><th>Equipment</th><th>Location</th><th>Target Downtime %</th></tr>
                {table_rows}
            </table>
        </div>
''')
    
    # Section 3: Opening Stock by Store
    opening_stocks = variability.get('opening_stocks', [])
    random_opening = variability['settings'].get('random_opening', True)
    
    if opening_stocks:
        df_opening = pd.DataFrame(opening_stocks)
        df_opening = df_opening[df_opening['capacity'] > 0].sort_values('store_key')
        
        if not df_opening.empty:
            fig_opening = go.Figure()
            
            fig_opening.add_trace(go.Bar(
                x=df_opening['store_key'],
                y=df_opening['actual_value'],
                name='Actual Opening',
                marker_color='#10b981',
                text=[f"{v:,.0f}" for v in df_opening['actual_value']],
                textposition='outside',
                textfont=dict(size=9)
            ))
            
            fig_opening.add_trace(go.Scatter(
                x=df_opening['store_key'],
                y=df_opening['capacity'],
                mode='markers',
                name='Capacity',
                marker=dict(color='#ef4444', size=8, symbol='line-ew-open', line=dict(width=2))
            ))
            
            fig_opening.update_layout(
                xaxis_title='Store',
                yaxis_title='Tonnes',
                template='plotly_white',
                height=400,
                margin=dict(l=80, r=40, t=30, b=120),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                font=dict(size=11)
            )
            fig_opening.update_xaxes(tickangle=45)
            
            opening_html = fig_opening.to_html(full_html=False, include_plotlyjs=False)
            
            status = "Random opening stock enabled - values sampled from configured ranges." if random_opening else "Random opening disabled - using midpoint of configured ranges."
            
            html_parts.append(f'''
        <div class="section">
            <div class="section-title">3. Opening Stock by Store</div>
            <div class="section-desc">
                {status} The chart shows <strong>actual opening stock values</strong> used in this simulation run
                compared to store capacity.
            </div>
            <div class="plot-container">
                {opening_html}
            </div>
        </div>
''')
        else:
            html_parts.append('''
        <div class="section">
            <div class="section-title">3. Opening Stock by Store</div>
            <div class="no-data">No stores with capacity configured.</div>
        </div>
''')
    
    html_parts.append('''
    </div>
</body>
</html>
''')
    
    output_file = out_dir / 'sim_outputs_variability.html'
    output_file.write_text(''.join(html_parts), encoding='utf-8')
    
    return output_file
