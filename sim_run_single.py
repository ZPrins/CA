"""
Single Simulation Runner Module

Handles running a single simulation with full artifact generation.
Used by the Flask app for streaming single simulation runs.
"""

import os
import subprocess
from pathlib import Path

from sim_run_config import config

OUT_DIR = Path(config.out_dir)


def run_single_simulation_subprocess(horizon_days: int, random_opening: bool, random_seed: str = '') -> subprocess.Popen:
    """
    Start a single simulation as a subprocess with streaming output.
    
    Args:
        horizon_days: Number of days to simulate
        random_opening: Whether to randomize opening stock levels
        random_seed: Random seed string (empty for random)
    
    Returns:
        subprocess.Popen object for streaming output
    """
    env = os.environ.copy()
    env['SIM_HORIZON_DAYS'] = str(horizon_days)
    env['SIM_RANDOM_OPENING'] = 'true' if random_opening else 'false'
    env['SIM_RANDOM_SEED'] = random_seed
    env['PYTHONUNBUFFERED'] = '1'
    
    proc = subprocess.Popen(
        ['python', '-u', 'sim_run.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        encoding='utf-8',
        errors='replace'
    )
    
    return proc


def run_single_simulation_blocking(horizon_days: int, random_opening: bool, random_seed: str = '', timeout: int = 300) -> dict:
    """
    Run a single simulation and wait for completion.
    
    Args:
        horizon_days: Number of days to simulate
        random_opening: Whether to randomize opening stock levels
        random_seed: Random seed string (empty for random)
        timeout: Maximum time in seconds to wait
    
    Returns:
        dict with 'success', 'output', and 'report_ready' keys
    """
    env = os.environ.copy()
    env['SIM_HORIZON_DAYS'] = str(horizon_days)
    env['SIM_RANDOM_OPENING'] = str(random_opening)
    env['SIM_RANDOM_SEED'] = random_seed
    
    try:
        result = subprocess.run(
            ['python', 'sim_run.py'],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env
        )
        
        success = result.returncode == 0
        output = result.stdout + result.stderr
        report_html = OUT_DIR / 'sim_outputs_plots_all.html'
        
        return {
            'success': success,
            'output': output,
            'report_ready': report_html.exists()
        }
    except subprocess.TimeoutExpired:
        return {'success': False, 'output': 'Simulation timed out', 'report_ready': False}
    except Exception as e:
        return {'success': False, 'output': str(e), 'report_ready': False}


def check_report_exists() -> bool:
    """Check if the simulation report HTML file exists."""
    return (OUT_DIR / 'sim_outputs_plots_all.html').exists()


def get_csv_files() -> list:
    """Get list of CSV output files."""
    if OUT_DIR.exists():
        return [f.name for f in OUT_DIR.glob('*.csv')]
    return []
