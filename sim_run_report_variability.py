# sim_run_report_variability.py
"""
Variability Distribution Report Generator

Generates a separate HTML page showing distribution curves for all 
stochastic elements in the simulation model.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import math
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sim_run_types import StoreConfig, MakeUnit


def collect_variability_metadata(store_configs: list, makes: list, settings: dict) -> dict:
    """
    Collect metadata about all variability drivers in the simulation.
    
    Returns dict with structure:
    {
        'opening_stock': [{'store_key': str, 'low': float, 'high': float}, ...],
        'breakdown_duration': {'mean_hours': float, 'sigma': float},
        'breakdown_probability': [{'equipment': str, 'location': str, 'downtime_pct': float}, ...],
    }
    """
    variability = {
        'opening_stock': [],
        'breakdown_duration': {
            'mean_hours': 3.0,
            'sigma': 0.8
        },
        'breakdown_probability': [],
        'settings': {
            'random_opening': settings.get('random_opening', True),
            'random_seed': settings.get('random_seed'),
            'horizon_days': settings.get('horizon_days', 365)
        }
    }
    
    # Collect opening stock distributions (Uniform)
    for cfg in store_configs:
        if cfg.opening_low != cfg.opening_high:
            variability['opening_stock'].append({
                'store_key': cfg.key,
                'low': cfg.opening_low,
                'high': cfg.opening_high,
                'capacity': cfg.capacity
            })
    
    # Collect breakdown probability data (for equipment with unplanned downtime)
    for make in makes:
        if make.unplanned_downtime_pct and make.unplanned_downtime_pct > 0:
            variability['breakdown_probability'].append({
                'equipment': make.equipment,
                'location': make.location,
                'downtime_pct': make.unplanned_downtime_pct,
                'maintenance_days': len(make.maintenance_days) if make.maintenance_days else 0
            })
    
    return variability


def _generate_lognormal_pdf(mean_hours: float, sigma: float, x_range: np.ndarray) -> np.ndarray:
    """Generate lognormal PDF values for the given x range."""
    mu = math.log(mean_hours) - (sigma ** 2) / 2
    pdf = (1 / (x_range * sigma * np.sqrt(2 * np.pi))) * \
          np.exp(-((np.log(x_range) - mu) ** 2) / (2 * sigma ** 2))
    return pdf


def _generate_lognormal_samples(mean_hours: float, sigma: float, n_samples: int = 1000) -> np.ndarray:
    """Generate sample values from lognormal distribution."""
    mu = math.log(mean_hours) - (sigma ** 2) / 2
    samples = np.random.lognormal(mu, sigma, n_samples)
    return np.maximum(1, np.round(samples))


def generate_variability_report(variability: dict, out_dir: Path) -> Path:
    """
    Generate an HTML report with distribution curves for all variability drivers.
    
    Args:
        variability: Dict from collect_variability_metadata()
        out_dir: Output directory for the report
        
    Returns:
        Path to generated HTML file
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    html_parts = []
    
    # HTML header
    html_parts.append(f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Variability Distributions</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 24px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ 
            text-align: center; 
            margin-bottom: 8px;
            font-size: 28px;
            color: #4fc3f7;
        }}
        .subtitle {{
            text-align: center;
            color: #90a4ae;
            margin-bottom: 24px;
            font-size: 14px;
        }}
        .back-link {{
            display: inline-block;
            margin-bottom: 20px;
            color: #4fc3f7;
            text-decoration: none;
            font-size: 14px;
        }}
        .back-link:hover {{ text-decoration: underline; }}
        .section {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .section-title {{
            font-size: 18px;
            font-weight: 600;
            color: #81d4fa;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .section-desc {{
            color: #b0bec5;
            font-size: 13px;
            margin-bottom: 16px;
            line-height: 1.5;
        }}
        .plot-container {{
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
        }}
        .param-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            margin-bottom: 16px;
        }}
        .param-table th {{
            text-align: left;
            padding: 8px 12px;
            background: rgba(79,195,247,0.1);
            color: #4fc3f7;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .param-table td {{
            padding: 8px 12px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}
        .param-table tr:hover td {{ background: rgba(255,255,255,0.02); }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px;
            margin-bottom: 16px;
        }}
        .stat-card {{
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            padding: 12px;
            text-align: center;
        }}
        .stat-value {{ font-size: 24px; font-weight: 600; color: #4fc3f7; }}
        .stat-label {{ font-size: 11px; color: #90a4ae; margin-top: 4px; }}
        .no-data {{ color: #90a4ae; font-style: italic; padding: 20px; text-align: center; }}
    </style>
</head>
<body>
    <div class="container">
        <a href="sim_outputs_plots_all.html" class="back-link">&larr; Back to Interactive Report</a>
        <h1>Variability Distributions</h1>
        <p class="subtitle">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Simulation Horizon: {variability['settings']['horizon_days']} days</p>
''')
    
    # Section 1: Breakdown Duration (Lognormal)
    breakdown_data = variability.get('breakdown_duration', {})
    mean_hours = breakdown_data.get('mean_hours', 3.0)
    sigma = breakdown_data.get('sigma', 0.8)
    mu = math.log(mean_hours) - (sigma ** 2) / 2
    
    # Calculate theoretical stats
    theoretical_mean = np.exp(mu + sigma**2 / 2)
    theoretical_median = np.exp(mu)
    theoretical_mode = np.exp(mu - sigma**2)
    theoretical_std = np.sqrt((np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2))
    
    # Generate PDF curve
    x_vals = np.linspace(0.1, 20, 200)
    y_vals = _generate_lognormal_pdf(mean_hours, sigma, x_vals)
    
    # Generate sample histogram
    samples = _generate_lognormal_samples(mean_hours, sigma, 5000)
    
    fig_breakdown = go.Figure()
    
    # Histogram of samples
    fig_breakdown.add_trace(go.Histogram(
        x=samples,
        nbinsx=30,
        name='Sampled Durations',
        marker_color='rgba(79, 195, 247, 0.6)',
        histnorm='probability density'
    ))
    
    # Theoretical PDF
    fig_breakdown.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines',
        name='Theoretical PDF',
        line=dict(color='#ff7043', width=3)
    ))
    
    # Add vertical lines for mean and median
    fig_breakdown.add_vline(x=theoretical_mean, line_dash="dash", line_color="#66bb6a", 
                           annotation_text=f"Mean: {theoretical_mean:.1f}h")
    fig_breakdown.add_vline(x=theoretical_median, line_dash="dot", line_color="#ffa726",
                           annotation_text=f"Median: {theoretical_median:.1f}h")
    
    fig_breakdown.update_layout(
        title=None,
        xaxis_title='Breakdown Duration (hours)',
        yaxis_title='Probability Density',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.2)',
        height=350,
        margin=dict(l=60, r=40, t=20, b=60),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        font=dict(size=12)
    )
    
    breakdown_html = fig_breakdown.to_html(full_html=False, include_plotlyjs=False)
    
    html_parts.append(f'''
        <div class="section">
            <div class="section-title">1. Breakdown Duration Distribution (Lognormal)</div>
            <div class="section-desc">
                When equipment experiences an unplanned breakdown, the duration is sampled from a 
                <strong>lognormal distribution</strong>. This produces realistic right-skewed durations 
                where most breakdowns are short but occasional long outages occur.
            </div>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{theoretical_mean:.1f}h</div>
                    <div class="stat-label">Mean Duration</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{theoretical_median:.1f}h</div>
                    <div class="stat-label">Median Duration</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{theoretical_mode:.1f}h</div>
                    <div class="stat-label">Mode (Most Likely)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{theoretical_std:.1f}h</div>
                    <div class="stat-label">Std Deviation</div>
                </div>
            </div>
            <div class="plot-container">
                {breakdown_html}
            </div>
            <table class="param-table">
                <tr><th>Parameter</th><th>Value</th><th>Description</th></tr>
                <tr><td>Target Mean</td><td>{mean_hours} hours</td><td>Configured mean breakdown duration</td></tr>
                <tr><td>Sigma (σ)</td><td>{sigma}</td><td>Shape parameter controlling spread</td></tr>
                <tr><td>Mu (μ)</td><td>{mu:.4f}</td><td>Calculated location parameter: ln(mean) - σ²/2</td></tr>
                <tr><td>Minimum</td><td>1 hour</td><td>Floor applied after sampling</td></tr>
            </table>
        </div>
''')
    
    # Section 2: Breakdown Probability by Equipment
    breakdown_probs = variability.get('breakdown_probability', [])
    
    if breakdown_probs:
        # Create bar chart of downtime percentages
        equipment_labels = [f"{bp['equipment']}<br>({bp['location']})" for bp in breakdown_probs]
        downtime_pcts = [bp['downtime_pct'] * 100 for bp in breakdown_probs]
        maintenance_days = [bp['maintenance_days'] for bp in breakdown_probs]
        
        fig_probs = go.Figure()
        fig_probs.add_trace(go.Bar(
            x=equipment_labels,
            y=downtime_pcts,
            name='Unplanned Downtime %',
            marker_color='rgba(255, 112, 67, 0.8)',
            text=[f'{p:.1f}%' for p in downtime_pcts],
            textposition='outside'
        ))
        
        fig_probs.update_layout(
            title=None,
            xaxis_title='Equipment',
            yaxis_title='Downtime Percentage (%)',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.2)',
            height=300,
            margin=dict(l=60, r=40, t=20, b=80),
            font=dict(size=12)
        )
        
        probs_html = fig_probs.to_html(full_html=False, include_plotlyjs=False)
        
        # Build table rows
        table_rows = ""
        for bp in breakdown_probs:
            table_rows += f'''<tr>
                <td>{bp['equipment']}</td>
                <td>{bp['location']}</td>
                <td>{bp['downtime_pct']*100:.1f}%</td>
                <td>{bp['maintenance_days']} days/year</td>
            </tr>'''
        
        html_parts.append(f'''
        <div class="section">
            <div class="section-title">2. Unplanned Downtime by Equipment</div>
            <div class="section-desc">
                Each production unit has a configured <strong>unplanned downtime percentage</strong>. 
                The simulation dynamically triggers breakdowns to achieve this target over time.
                When available hours accumulate, breakdowns are triggered probabilistically.
            </div>
            <div class="plot-container">
                {probs_html}
            </div>
            <table class="param-table">
                <tr><th>Equipment</th><th>Location</th><th>Unplanned Downtime</th><th>Planned Maintenance</th></tr>
                {table_rows}
            </table>
        </div>
''')
    else:
        html_parts.append('''
        <div class="section">
            <div class="section-title">2. Unplanned Downtime by Equipment</div>
            <div class="no-data">No equipment has unplanned downtime configured.</div>
        </div>
''')
    
    # Section 3: Opening Stock Distribution (Uniform)
    opening_stocks = variability.get('opening_stock', [])
    random_opening = variability['settings'].get('random_opening', True)
    
    if opening_stocks and random_opening:
        # Create multiple uniform distribution visualizations
        n_stores = len(opening_stocks)
        
        # Sort by range size for better visualization
        opening_stocks_sorted = sorted(opening_stocks, key=lambda x: x['high'] - x['low'], reverse=True)
        
        # Create a range chart
        fig_opening = go.Figure()
        
        store_names = [s['store_key'] for s in opening_stocks_sorted[:20]]  # Top 20
        lows = [s['low'] for s in opening_stocks_sorted[:20]]
        highs = [s['high'] for s in opening_stocks_sorted[:20]]
        midpoints = [(s['low'] + s['high']) / 2 for s in opening_stocks_sorted[:20]]
        ranges = [s['high'] - s['low'] for s in opening_stocks_sorted[:20]]
        
        # Error bars showing range
        fig_opening.add_trace(go.Scatter(
            x=midpoints,
            y=store_names,
            mode='markers',
            marker=dict(size=12, color='#4fc3f7'),
            error_x=dict(
                type='data',
                symmetric=False,
                array=[h - m for h, m in zip(highs, midpoints)],
                arrayminus=[m - l for m, l in zip(midpoints, lows)],
                color='rgba(79,195,247,0.6)',
                thickness=8
            ),
            name='Opening Stock Range',
            hovertemplate='%{y}<br>Range: %{customdata[0]:,.0f} - %{customdata[1]:,.0f} t<extra></extra>',
            customdata=list(zip(lows, highs))
        ))
        
        fig_opening.update_layout(
            title=None,
            xaxis_title='Opening Stock (tonnes)',
            yaxis_title='Store',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.2)',
            height=max(300, len(store_names) * 25),
            margin=dict(l=200, r=40, t=20, b=60),
            font=dict(size=11)
        )
        
        opening_html = fig_opening.to_html(full_html=False, include_plotlyjs=False)
        
        # Sample uniform distribution for illustration
        fig_uniform = go.Figure()
        x_uniform = np.linspace(0, 100, 100)
        y_uniform = np.ones_like(x_uniform) / 100  # Uniform PDF
        
        fig_uniform.add_trace(go.Scatter(
            x=x_uniform,
            y=y_uniform,
            fill='tozeroy',
            mode='lines',
            line=dict(color='#4fc3f7', width=2),
            fillcolor='rgba(79,195,247,0.3)',
            name='Uniform Distribution'
        ))
        
        fig_uniform.update_layout(
            title='Uniform Distribution Shape',
            xaxis_title='Value (as % of range)',
            yaxis_title='Probability Density',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.2)',
            height=200,
            margin=dict(l=60, r=40, t=40, b=60),
            font=dict(size=11)
        )
        
        uniform_html = fig_uniform.to_html(full_html=False, include_plotlyjs=False)
        
        # Build table rows (first 15)
        table_rows = ""
        for s in opening_stocks_sorted[:15]:
            range_val = s['high'] - s['low']
            pct_capacity = (s['high'] / s['capacity'] * 100) if s['capacity'] > 0 else 0
            table_rows += f'''<tr>
                <td>{s['store_key']}</td>
                <td>{s['low']:,.0f}</td>
                <td>{s['high']:,.0f}</td>
                <td>{range_val:,.0f}</td>
                <td>{s['capacity']:,.0f}</td>
            </tr>'''
        
        if len(opening_stocks_sorted) > 15:
            table_rows += f'<tr><td colspan="5" style="text-align:center; color:#90a4ae;">... and {len(opening_stocks_sorted) - 15} more stores</td></tr>'
        
        html_parts.append(f'''
        <div class="section">
            <div class="section-title">3. Opening Stock Distribution (Uniform)</div>
            <div class="section-desc">
                When <strong>Random Opening Stock</strong> is enabled, each store's initial inventory 
                is sampled from a <strong>uniform distribution</strong> between its configured Low and High values.
                Every value in the range is equally likely.
            </div>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{len(opening_stocks)}</div>
                    <div class="stat-label">Stores with Variable Opening</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">Uniform</div>
                    <div class="stat-label">Distribution Type</div>
                </div>
            </div>
            <div class="plot-container">
                {opening_html}
            </div>
            <div class="plot-container">
                {uniform_html}
            </div>
            <table class="param-table">
                <tr><th>Store</th><th>Low (t)</th><th>High (t)</th><th>Range (t)</th><th>Capacity (t)</th></tr>
                {table_rows}
            </table>
        </div>
''')
    else:
        status = "Random Opening Stock is disabled." if not random_opening else "All stores have fixed opening stock (Low = High)."
        html_parts.append(f'''
        <div class="section">
            <div class="section-title">3. Opening Stock Distribution (Uniform)</div>
            <div class="no-data">{status}</div>
        </div>
''')
    
    # Close HTML
    html_parts.append('''
    </div>
</body>
</html>
''')
    
    # Write file
    output_file = out_dir / 'sim_outputs_variability.html'
    output_file.write_text(''.join(html_parts), encoding='utf-8')
    
    return output_file
