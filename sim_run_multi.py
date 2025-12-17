"""
Multi-Run Simulation Runner Module

Handles running multiple simulations in parallel with KPI extraction.
Used by the Flask app for batch simulation runs.
"""

import os
import subprocess
import json
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Generator


def run_single_sim_for_kpi(args: Tuple[int, int, bool, int]) -> Dict[str, Any]:
    """
    Run a single simulation and extract KPIs. Used for multiprocessing.
    
    Args:
        args: Tuple of (run_idx, horizon_days, random_opening, base_seed)
    
    Returns:
        dict with KPIs or error information
    """
    run_idx, horizon_days, random_opening, base_seed = args
    
    try:
        if base_seed is not None:
            seed = base_seed + run_idx
        else:
            seed = None
        
        env = os.environ.copy()
        env['SIM_HORIZON_DAYS'] = str(horizon_days)
        env['SIM_RANDOM_OPENING'] = 'true' if random_opening else 'false'
        env['SIM_RANDOM_SEED'] = str(seed) if seed is not None else ''
        env['SIM_QUIET_MODE'] = 'true'
        env['SIM_KPI_ONLY'] = 'true'
        env['PYTHONUNBUFFERED'] = '1'
        
        result = subprocess.run(
            ['python', '-c', '''
import os
import json
os.environ["SIM_QUIET_MODE"] = "true"
os.environ["SIM_KPI_ONLY"] = "true"
from sim_run import run_simulation
result = run_simulation(artifacts="kpi_only")
print("KPI_JSON:" + json.dumps(result.get("kpis", {})))
'''],
            capture_output=True,
            text=True,
            timeout=600,
            env=env
        )
        
        if result.returncode != 0:
            return {'run_idx': run_idx, 'error': result.stderr}
        
        kpis = {'run_idx': run_idx}
        for line in result.stdout.split('\n'):
            if line.startswith('KPI_JSON:'):
                try:
                    kpis.update(json.loads(line[9:]))
                except:
                    pass
        
        return kpis
        
    except Exception as e:
        return {'run_idx': run_idx, 'error': str(e)}


def run_multi_simulation(horizon_days: int, random_opening: bool, num_runs: int) -> Dict[str, Any]:
    """
    Run multiple simulations in parallel and collect KPIs.
    
    Args:
        horizon_days: Number of days to simulate
        random_opening: Whether to randomize opening stock levels
        num_runs: Number of simulation runs (max 100)
    
    Returns:
        dict with 'success', 'kpis', 'errors', 'total_runs', 'successful_runs'
    """
    num_runs = min(num_runs, 100)
    base_seed = random.randint(1, 100000)
    
    run_args = [
        (i, horizon_days, random_opening, base_seed)
        for i in range(num_runs)
    ]
    
    max_workers = min(multiprocessing.cpu_count(), 4, num_runs)
    
    kpis = []
    errors = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_sim_for_kpi, args): args[0] for args in run_args}
        
        for future in as_completed(futures):
            result = future.result()
            if 'error' in result:
                errors.append(result)
            else:
                kpis.append(result)
    
    kpis.sort(key=lambda x: x.get('run_idx', 0))
    
    return {
        'success': len(kpis) > 0,
        'kpis': kpis,
        'errors': errors,
        'total_runs': num_runs,
        'successful_runs': len(kpis)
    }


def stream_multi_simulation(horizon_days: int, random_opening: bool, num_runs: int, 
                            stop_check=None) -> Generator[str, None, None]:
    """
    Run multiple simulations and yield SSE progress messages.
    
    Args:
        horizon_days: Number of days to simulate
        random_opening: Whether to randomize opening stock levels
        num_runs: Number of simulation runs (max 100)
        stop_check: Callable that returns True if stop was requested
    
    Yields:
        SSE-formatted strings for progress updates
    """
    num_runs = min(num_runs, 100)
    base_seed = random.randint(1, 100000)
    
    run_args = [
        (i, horizon_days, random_opening, base_seed)
        for i in range(num_runs)
    ]
    
    max_workers = min(multiprocessing.cpu_count(), 4, num_runs)
    
    kpis = []
    errors = []
    completed = 0
    
    yield f'event: start\ndata: {{"total": {num_runs}, "workers": {max_workers}}}\n\n'
    yield f'data: Starting {num_runs} simulation(s) with {max_workers} parallel workers...\n\n'
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_sim_for_kpi, args): args[0] for args in run_args}
        
        for future in as_completed(futures):
            if stop_check and stop_check():
                executor.shutdown(wait=False, cancel_futures=True)
                yield 'data: Simulation stopped by user\n\n'
                yield f'event: done\ndata: {{"stopped": true}}\n\n'
                return
            
            result = future.result()
            completed += 1
            run_idx = result.get('run_idx', '?')
            
            if 'error' in result:
                errors.append(result)
                yield f'data: Run {run_idx + 1}/{num_runs}: ERROR - {result["error"][:100]}\n\n'
            else:
                kpis.append(result)
                unmet = result.get('total_unmet_demand', 0)
                prod = result.get('total_production', 0)
                ships = result.get('ship_trips', 0)
                trains = result.get('train_trips', 0)
                yield f'data: Run {run_idx + 1}/{num_runs}: Unmet={unmet:,.0f}t, Prod={prod:,.0f}t, Ships={ships}, Trains={trains}\n\n'
            
            pct = int((completed / num_runs) * 100)
            yield f'event: progress\ndata: {{"completed": {completed}, "total": {num_runs}, "pct": {pct}}}\n\n'
    
    kpis.sort(key=lambda x: x.get('run_idx', 0))
    
    payload = {
        'success': len(kpis) > 0,
        'kpis': kpis,
        'errors': errors,
        'total_runs': num_runs,
        'successful_runs': len(kpis)
    }
    yield f'event: done\ndata: {json.dumps(payload)}\n\n'


def calculate_kpi_averages(kpis: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate average KPIs from a list of simulation results.
    
    Args:
        kpis: List of KPI dictionaries from simulation runs
    
    Returns:
        dict with average values for each KPI
    """
    if not kpis:
        return {
            'avg_unmet_demand': 0,
            'avg_production': 0,
            'avg_inventory_pct': 0,
            'avg_ship_trips': 0,
            'avg_train_trips': 0
        }
    
    n = len(kpis)
    return {
        'avg_unmet_demand': sum(k.get('total_unmet_demand', 0) for k in kpis) / n,
        'avg_production': sum(k.get('total_production', 0) for k in kpis) / n,
        'avg_inventory_pct': sum(k.get('avg_inventory_pct', 0) for k in kpis) / n,
        'avg_ship_trips': sum(k.get('ship_trips', 0) for k in kpis) / n,
        'avg_train_trips': sum(k.get('train_trips', 0) for k in kpis) / n
    }
