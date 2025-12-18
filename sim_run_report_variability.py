# sim_run_report_variability.py
"""
Variability Distribution Report Generator

Generates a separate HTML page showing probability density graphs for:
- Breakdown duration per equipment
- Opening stock distributions (theoretical uniform PDF)
- Berth waiting times from ship state logs
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
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
        'breakdown_by_equipment': {},
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
    
    # Parse breakdown events - group by equipment for per-equipment PDFs
    breakdown_start_rows = df_log[(df_log['process'] == 'Downtime') & (df_log['event'] == 'BreakdownStart')]
    
    if not breakdown_start_rows.empty:
        for _, row in breakdown_start_rows.iterrows():
            equip = row['equipment']
            duration = row['qty'] if pd.notna(row['qty']) and row['qty'] > 0 else 1
            
            variability['breakdown_events'].append({
                'equipment': equip,
                'start_time': row['time_h'],
                'duration_hours': float(duration)
            })
            
            if equip not in variability['breakdown_by_equipment']:
                variability['breakdown_by_equipment'][equip] = []
            variability['breakdown_by_equipment'][equip].append(float(duration))
    else:
        breakdown_rows = df_log[(df_log['process'] == 'Downtime') & (df_log['event'] == 'Breakdown')]
        if not breakdown_rows.empty:
            for equip in breakdown_rows['equipment'].unique():
                equip_rows = breakdown_rows[breakdown_rows['equipment'] == equip].sort_values('time_h')
                times = equip_rows['time_h'].tolist()
                
                if equip not in variability['breakdown_by_equipment']:
                    variability['breakdown_by_equipment'][equip] = []
                
                i = 0
                while i < len(times):
                    duration = 1
                    while i + 1 < len(times) and times[i + 1] - times[i] <= 1.5:
                        duration += 1
                        i += 1
                    variability['breakdown_events'].append({
                        'equipment': equip,
                        'start_time': times[i - duration + 1] if duration > 1 else times[i],
                        'duration_hours': float(duration)
                    })
                    variability['breakdown_by_equipment'][equip].append(float(duration))
                    i += 1
    
    # Parse ship state changes to extract berth waiting times
    # Track time spent in WAITING_FOR_BERTH state per vessel
    if 'ship_state' in df_log.columns and 'vessel_id' in df_log.columns:
        ship_rows = df_log[df_log['process'] == 'ShipState'].sort_values(['vessel_id', 'time_h'])
        
        if not ship_rows.empty:
            for vessel_id in ship_rows['vessel_id'].dropna().unique():
                vessel_rows = ship_rows[ship_rows['vessel_id'] == vessel_id].sort_values('time_h').reset_index(drop=True)
                rows_list = vessel_rows.to_dict('records')
                
                for i, row in enumerate(rows_list):
                    state = str(row.get('ship_state', '')).upper()
                    
                    if state == 'WAITING_FOR_BERTH':
                        waiting_start = row['time_h']
                        waiting_location = row['location']
                        
                        # Find the next state change (any state that exits WAITING_FOR_BERTH)
                        for j in range(i + 1, len(rows_list)):
                            next_row = rows_list[j]
                            next_state = str(next_row.get('ship_state', '')).upper()
                            
                            if next_state != 'WAITING_FOR_BERTH':
                                wait_time = next_row['time_h'] - waiting_start
                                # Record even if wait_time is 0 or very small (instant berth access)
                                variability['berth_waiting_events'].append({
                                    'vessel_id': vessel_id,
                                    'location': waiting_location,
                                    'wait_hours': max(0.0, wait_time)
                                })
                                break
    
    return variability


def generate_variability_report(variability: dict, out_dir: Path) -> Path:
    """
    Generate an HTML report with probability density graphs for variability analysis.
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
        .plot-title {{
            font-weight: 600;
            font-size: 14px;
            margin-bottom: 8px;
            color: #334155;
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
        .equipment-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 16px;
        }}
        .store-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 16px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <a href="/outputs/sim_outputs_plots_all.html" class="back-link">&larr; Back to Interactive Report</a>
        <h1>Variability Analysis</h1>
        <p class="subtitle">Probability Density Graphs | Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Horizon: {variability['settings']['horizon_days']} days</p>
''')
    
    # Section 1: Breakdown Duration PDFs by Equipment
    breakdown_by_equip = variability.get('breakdown_by_equipment', {})
    
    if breakdown_by_equip:
        total_events = sum(len(v) for v in breakdown_by_equip.values())
        all_durations = [d for durations in breakdown_by_equip.values() for d in durations]
        avg_duration = np.mean(all_durations) if all_durations else 0
        max_duration = max(all_durations) if all_durations else 0
        
        html_parts.append(f'''
        <div class="section">
            <div class="section-title">1. Breakdown Duration Probability Density by Equipment</div>
            <div class="section-desc">
                Each chart shows the <strong>probability density function (PDF)</strong> of breakdown durations 
                for a specific piece of equipment. The shape reveals whether breakdowns tend to be short, long, 
                or widely variable for each unit.
            </div>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{len(breakdown_by_equip)}</div>
                    <div class="stat-label">Equipment with Breakdowns</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{total_events}</div>
                    <div class="stat-label">Total Events</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{avg_duration:.1f}h</div>
                    <div class="stat-label">Overall Avg Duration</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{max_duration:.0f}h</div>
                    <div class="stat-label">Max Duration</div>
                </div>
            </div>
            <div class="equipment-grid">
''')
        
        colors = ['#f97316', '#2563eb', '#10b981', '#8b5cf6', '#ef4444', '#06b6d4', '#eab308', '#ec4899']
        
        for idx, (equip, durations) in enumerate(sorted(breakdown_by_equip.items())):
            color = colors[idx % len(colors)]
            n_events = len(durations)
            eq_avg = np.mean(durations)
            eq_max = max(durations)
            
            fig = go.Figure()
            
            if n_events >= 2:
                # Create histogram with normalization to approximate PDF
                fig.add_trace(go.Histogram(
                    x=durations,
                    histnorm='probability density',
                    name='Density',
                    marker_color=color,
                    opacity=0.85,
                    nbinsx=min(15, max(5, n_events // 3))
                ))
                
                # Add vertical line for mean
                fig.add_vline(x=eq_avg, line_dash="dash", line_color="#1e293b",
                             annotation_text=f"Mean: {eq_avg:.1f}h", annotation_position="top right")
            else:
                # Single event - show as bar
                fig.add_trace(go.Bar(
                    x=[durations[0]],
                    y=[1],
                    marker_color=color,
                    width=0.5,
                    text=[f'{durations[0]:.1f}h'],
                    textposition='outside'
                ))
            
            fig.update_layout(
                xaxis_title='Duration (hours)',
                yaxis_title='Probability Density',
                template='plotly_white',
                height=250,
                margin=dict(l=50, r=30, t=30, b=50),
                font=dict(size=11),
                showlegend=False,
                bargap=0.15
            )
            
            chart_html = fig.to_html(full_html=False, include_plotlyjs=False)
            
            html_parts.append(f'''
                <div class="plot-container">
                    <div class="plot-title">{equip} <span style="color:#64748b;font-weight:normal">({n_events} events, avg {eq_avg:.1f}h, max {eq_max:.0f}h)</span></div>
                    {chart_html}
                </div>
''')
        
        html_parts.append('''
            </div>
        </div>
''')
    else:
        html_parts.append('''
        <div class="section">
            <div class="section-title">1. Breakdown Duration Probability Density by Equipment</div>
            <div class="no-data">No breakdown events occurred during this simulation run.</div>
        </div>
''')
    
    # Section 2: Berth Waiting Times by Berth (same format as breakdown by equipment)
    berth_events = variability.get('berth_waiting_events', [])
    
    if berth_events:
        df_berth = pd.DataFrame(berth_events)
        avg_wait = df_berth['wait_hours'].mean()
        max_wait = df_berth['wait_hours'].max()
        total_wait_events = len(berth_events)
        
        # Group by location (berth)
        berth_by_location = {}
        for evt in berth_events:
            loc = evt['location']
            if loc not in berth_by_location:
                berth_by_location[loc] = []
            berth_by_location[loc].append(evt['wait_hours'])
        
        html_parts.append(f'''
        <div class="section">
            <div class="section-title">2. Berth Waiting Time Probability Density by Berth</div>
            <div class="section-desc">
                Each chart shows the <strong>probability density function (PDF)</strong> of waiting times
                at a specific berth/port. Longer wait times indicate port congestion or capacity constraints.
            </div>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{len(berth_by_location)}</div>
                    <div class="stat-label">Berths with Waits</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{total_wait_events}</div>
                    <div class="stat-label">Total Wait Events</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{avg_wait:.1f}h</div>
                    <div class="stat-label">Overall Avg Wait</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{max_wait:.1f}h</div>
                    <div class="stat-label">Max Wait</div>
                </div>
            </div>
            <div class="equipment-grid">
''')
        
        berth_colors = ['#8b5cf6', '#2563eb', '#10b981', '#f97316', '#ef4444', '#06b6d4', '#eab308', '#ec4899']
        
        for idx, (berth, wait_times) in enumerate(sorted(berth_by_location.items())):
            color = berth_colors[idx % len(berth_colors)]
            n_events = len(wait_times)
            berth_avg = np.mean(wait_times)
            berth_max = max(wait_times)
            
            fig = go.Figure()
            
            if n_events >= 2:
                fig.add_trace(go.Histogram(
                    x=wait_times,
                    histnorm='probability density',
                    name='Density',
                    marker_color=color,
                    opacity=0.85,
                    nbinsx=min(15, max(5, n_events // 3))
                ))
                
                fig.add_vline(x=berth_avg, line_dash="dash", line_color="#1e293b",
                             annotation_text=f"Mean: {berth_avg:.1f}h", annotation_position="top right")
            else:
                fig.add_trace(go.Bar(
                    x=[wait_times[0]],
                    y=[1],
                    marker_color=color,
                    width=0.5,
                    text=[f'{wait_times[0]:.1f}h'],
                    textposition='outside'
                ))
            
            fig.update_layout(
                xaxis_title='Wait Time (hours)',
                yaxis_title='Probability Density',
                template='plotly_white',
                height=250,
                margin=dict(l=50, r=30, t=30, b=50),
                font=dict(size=11),
                showlegend=False,
                bargap=0.15
            )
            
            chart_html = fig.to_html(full_html=False, include_plotlyjs=False)
            
            html_parts.append(f'''
                <div class="plot-container">
                    <div class="plot-title">{berth} <span style="color:#64748b;font-weight:normal">({n_events} events, avg {berth_avg:.1f}h, max {berth_max:.1f}h)</span></div>
                    {chart_html}
                </div>
''')
        
        html_parts.append('''
            </div>
        </div>
''')
    else:
        html_parts.append('''
        <div class="section">
            <div class="section-title">2. Berth Waiting Time Probability Density by Berth</div>
            <div class="no-data">No significant berth waiting events recorded. Ships had immediate berth access, or ship state logging is not yet enabled.</div>
        </div>
''')
    
    # Section 3: Opening Stock Probability Density (Theoretical Uniform PDFs)
    opening_stocks = variability.get('opening_stocks', [])
    random_opening = variability['settings'].get('random_opening', True)
    
    # Filter to stores with meaningful ranges
    stores_with_range = [s for s in opening_stocks if s['opening_high'] > s['opening_low'] and s['capacity'] > 0]
    
    if stores_with_range and random_opening:
        html_parts.append(f'''
        <div class="section">
            <div class="section-title">3. Opening Stock Probability Density by Store</div>
            <div class="section-desc">
                Each chart shows the <strong>theoretical uniform probability density</strong> for opening stock.
                When random opening is enabled, values are sampled uniformly between the configured low and high bounds.
                The red dot shows the <strong>actual value</strong> sampled in this simulation run.
            </div>
            <div class="store-grid">
''')
        
        for store in sorted(stores_with_range, key=lambda x: x['store_key']):
            low = store['opening_low'] / 1000  # Convert to kT
            high = store['opening_high'] / 1000  # Convert to kT
            actual = store['actual_value'] / 1000  # Convert to kT
            capacity = store['capacity'] / 1000  # Convert to kT
            range_width = high - low
            pdf_height = 1.0 / range_width if range_width > 0 else 0
            
            fig = go.Figure()
            
            # Draw uniform PDF as filled area
            x_pdf = [low, low, high, high]
            y_pdf = [0, pdf_height, pdf_height, 0]
            
            fig.add_trace(go.Scatter(
                x=x_pdf,
                y=y_pdf,
                fill='toself',
                fillcolor='rgba(16, 185, 129, 0.3)',
                line=dict(color='#10b981', width=2),
                name='Uniform PDF',
                mode='lines'
            ))
            
            # Mark actual value
            fig.add_trace(go.Scatter(
                x=[actual],
                y=[pdf_height / 2],
                mode='markers',
                marker=dict(color='#ef4444', size=12, symbol='diamond'),
                name=f'Actual: {actual:.1f} kT'
            ))
            
            # Add capacity reference line
            if capacity <= high * 1.5:
                fig.add_vline(x=capacity, line_dash="dot", line_color="#64748b",
                             annotation_text="Capacity", annotation_position="top")
            
            fig.update_layout(
                xaxis_title='Opening Stock (kT)',
                yaxis_title='Probability Density',
                template='plotly_white',
                height=220,
                margin=dict(l=50, r=30, t=30, b=50),
                font=dict(size=10),
                showlegend=False,
                xaxis=dict(range=[max(0, low - range_width * 0.15), high + range_width * 0.15])
            )
            
            chart_html = fig.to_html(full_html=False, include_plotlyjs=False)
            
            html_parts.append(f'''
                <div class="plot-container">
                    <div class="plot-title">{store['store_key']} <span style="color:#64748b;font-weight:normal">(Range: {low:.1f} - {high:.1f} kT, Actual: {actual:.1f} kT)</span></div>
                    {chart_html}
                </div>
''')
        
        html_parts.append('''
            </div>
        </div>
''')
    elif not random_opening:
        html_parts.append('''
        <div class="section">
            <div class="section-title">3. Opening Stock Probability Density by Store</div>
            <div class="no-data">Random opening stock is disabled. All stores start at the midpoint of their configured range (deterministic).</div>
        </div>
''')
    else:
        html_parts.append('''
        <div class="section">
            <div class="section-title">3. Opening Stock Probability Density by Store</div>
            <div class="no-data">No stores with configurable opening stock ranges found.</div>
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
