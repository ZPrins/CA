from flask import Flask, render_template, request, jsonify, send_from_directory, Response, stream_with_context
import time
import subprocess
import os
import json
import tempfile
from pathlib import Path
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Move config import to top level (assumes config doesn't import app)
from sim_run_config import config

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Ensure output directory exists reference
OUT_DIR = Path(config.out_dir)
INPUT_FILE = 'generated_model_inputs.xlsx'


def load_model_params():
    """Load Store and Move_Ship parameters from Excel for UI editing."""
    store_data = []
    move_ship_data = []
    
    if os.path.exists(INPUT_FILE):
        try:
            xls = pd.read_excel(INPUT_FILE, sheet_name=None)
            
            # Load Store sheet
            for sheet_name in xls.keys():
                if sheet_name.lower() == 'store':
                    df = xls[sheet_name].dropna(how='all')
                    df.columns = [str(c).strip() for c in df.columns]
                    # Filter out unnamed columns and convert NaN to empty string
                    df = df[[c for c in df.columns if not c.startswith('Unnamed')]]
                    store_data = df.fillna('').to_dict('records')
                    break
            
            # Load Move_Ship sheet
            for sheet_name in xls.keys():
                if sheet_name.lower() == 'move_ship':
                    df = xls[sheet_name].dropna(how='all')
                    df.columns = [str(c).strip() for c in df.columns]
                    # Filter out unnamed columns and convert NaN to empty string
                    df = df[[c for c in df.columns if not c.startswith('Unnamed')]]
                    move_ship_data = df.fillna('').to_dict('records')
                    break
                    
        except Exception as e:
            print(f"Error loading model params: {e}")
    
    return {'store': store_data, 'move_ship': move_ship_data}


@app.route('/')
def index():
    outputs_exist = (OUT_DIR / 'sim_outputs_plots_all.html').exists()
    csv_files = list(OUT_DIR.glob('*.csv')) if OUT_DIR.exists() else []
    model_params = load_model_params()

    return render_template('index.html',
                           config=config,
                           outputs_exist=outputs_exist,
                           csv_files=[f.name for f in csv_files],
                           model_params=model_params)


@app.route('/api/model-params')
def get_model_params():
    """API endpoint to get model parameters."""
    return jsonify(load_model_params())


TEMP_OVERRIDES_FILE = 'temp_model_overrides.json'


@app.route('/api/save-params', methods=['POST'])
def save_params():
    """Save user-modified parameters to a temp file for simulation to use."""
    try:
        data = request.get_json() or {}
        with open(TEMP_OVERRIDES_FILE, 'w') as f:
            json.dump(data, f)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/run-simulation', methods=['POST'])
def run_simulation():
    try:
        data = request.get_json() or {}

        # Parse inputs
        horizon = int(data.get('horizon_days', 365))
        random_opening = data.get('random_opening', True)
        random_seed = data.get('random_seed', None)

        # Format seed for env var
        if random_seed == '' or random_seed is None:
            seed_str = ''
        else:
            seed_str = str(int(random_seed))

        # Prepare environment variables for the subprocess
        env = os.environ.copy()
        env['SIM_HORIZON_DAYS'] = str(horizon)
        env['SIM_RANDOM_OPENING'] = str(random_opening)
        env['SIM_RANDOM_SEED'] = seed_str

        # Run the modular simulation script
        result = subprocess.run(
            ['python', 'sim_run.py'],
            capture_output=True,
            text=True,
            timeout=300,
            env=env
        )

        success = result.returncode == 0
        output = result.stdout + result.stderr

        report_html = OUT_DIR / 'sim_outputs_plots_all.html'

        return jsonify({
            'success': success,
            'output': output,
            'report_ready': report_html.exists()
        })
    except subprocess.TimeoutExpired:
        return jsonify({'success': False, 'output': 'Simulation timed out (5 min limit)'})
    except Exception as e:
        return jsonify({'success': False, 'output': str(e)})


@app.route('/stream-simulation')
def stream_simulation():
    # SSE endpoint that streams live logs from the sim process
    try:
        # Query params
        horizon = int(request.args.get('horizon_days', 365))
        random_opening = request.args.get('random_opening', 'true').lower() == 'true'
        rs = request.args.get('random_seed', '')

        seed_str = str(int(rs)) if rs and rs.strip() else ''

        env = os.environ.copy()
        env['SIM_HORIZON_DAYS'] = str(horizon)
        env['SIM_RANDOM_OPENING'] = 'true' if random_opening else 'false'
        env['SIM_RANDOM_SEED'] = seed_str
        env['PYTHONUNBUFFERED'] = '1'

        # Start child process in unbuffered mode
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

        def sse_lines():
            try:
                yield 'event: start\ndata: Simulation started\n\n'
                last_heartbeat = time.time()

                if proc.stdout is not None:
                    while True:
                        line = proc.stdout.readline()
                        if line:
                            msg = line.rstrip('\r\n')
                            if msg:
                                if msg.startswith('Progress:'):
                                    yield f'event: progress\ndata: {msg}\n\n'
                                else:
                                    yield f'data: {msg}\n\n'
                            last_heartbeat = time.time()
                        else:
                            # No line available; check if process ended
                            if proc.poll() is not None:
                                break

                            # Heartbeat
                            now = time.time()
                            if now - last_heartbeat >= 1.5:
                                yield ': keep-alive\n\n'
                                last_heartbeat = now
                            time.sleep(0.2)

                proc.wait()

                report_html = OUT_DIR / 'sim_outputs_plots_all.html'
                payload = {
                    'success': proc.returncode == 0,
                    'report_ready': report_html.exists(),
                    'exit_code': proc.returncode,
                }
                import json as _json
                yield 'event: done\ndata: ' + _json.dumps(payload) + '\n\n'
            finally:
                try:
                    if proc.poll() is None:
                        proc.terminate()
                except Exception:
                    pass

        headers = {
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        }
        return Response(stream_with_context(sse_lines()), mimetype='text/event-stream', headers=headers)
    except Exception as e:
        return jsonify({'success': False, 'output': str(e)})


@app.route('/report')
def report():
    report_path = OUT_DIR / 'sim_outputs_plots_all.html'
    if report_path.exists():
        return send_from_directory(str(OUT_DIR), 'sim_outputs_plots_all.html')
    return "Report not found. Run a simulation first.", 404


@app.route('/outputs/<filename>')
def get_output_file(filename):
    return send_from_directory(str(OUT_DIR), filename)


@app.route('/api/status')
def status():
    return jsonify({
        'report_exists': (OUT_DIR / 'sim_outputs_plots_all.html').exists(),
        'csv_files': [f.name for f in OUT_DIR.glob('*.csv')] if OUT_DIR.exists() else []
    })


def run_single_sim_for_kpi(args):
    """Run a single simulation and extract KPIs. Used for multiprocessing."""
    run_idx, horizon_days, random_opening, base_seed = args
    
    try:
        # Each run gets a unique seed based on the base seed and run index
        if base_seed is not None:
            seed = base_seed + run_idx
        else:
            seed = None
        
        env = os.environ.copy()
        env['SIM_HORIZON_DAYS'] = str(horizon_days)
        env['SIM_RANDOM_OPENING'] = 'true' if random_opening else 'false'
        env['SIM_RANDOM_SEED'] = str(seed) if seed is not None else ''
        env['SIM_QUIET_MODE'] = 'true'  # Disable detailed logging
        env['PYTHONUNBUFFERED'] = '1'
        
        # Run simulation
        result = subprocess.run(
            ['python', 'sim_run.py'],
            capture_output=True,
            text=True,
            timeout=600,
            env=env
        )
        
        if result.returncode != 0:
            return {'run_idx': run_idx, 'error': result.stderr}
        
        # Extract KPIs from output files
        kpis = extract_kpis_from_outputs()
        kpis['run_idx'] = run_idx
        return kpis
        
    except Exception as e:
        return {'run_idx': run_idx, 'error': str(e)}


def extract_kpis_from_outputs():
    """Extract key KPIs from simulation output files."""
    kpis = {
        'total_unmet_demand': 0,
        'total_production': 0,
        'avg_inventory_pct': 0,
        'ship_trips': 0,
        'train_trips': 0
    }
    
    try:
        # Read simulation log for production and transport stats
        log_file = OUT_DIR / 'sim_outputs_sim_log.csv'
        if log_file.exists():
            df = pd.read_csv(log_file)
            
            # Total production
            prod = df[(df['process'] == 'Make') & (df['event'] == 'Produce')]
            if not prod.empty and 'qty' in prod.columns:
                kpis['total_production'] = prod['qty'].astype(float).sum()
            
            # Ship trips
            ship_moves = df[(df['process'] == 'Move') & (df['equipment'] == 'Ship') & (df['event'] == 'ShipUnload')]
            kpis['ship_trips'] = len(ship_moves)
            
            # Train trips
            train_moves = df[(df['process'] == 'Move') & (df['equipment'] == 'Train') & (df['event'] == 'Unload')]
            kpis['train_trips'] = len(train_moves)
        
        # Read unmet demand
        unmet_file = OUT_DIR / 'sim_outputs_unmet_demand.csv'
        if unmet_file.exists():
            df_unmet = pd.read_csv(unmet_file)
            if 'unmet_qty' in df_unmet.columns:
                kpis['total_unmet_demand'] = df_unmet['unmet_qty'].astype(float).sum()
        
        # Read inventory for average utilization
        inv_file = OUT_DIR / 'sim_outputs_inventory_daily.csv'
        if inv_file.exists():
            df_inv = pd.read_csv(inv_file)
            if 'level' in df_inv.columns and 'capacity' in df_inv.columns:
                df_inv['pct'] = (df_inv['level'] / df_inv['capacity']) * 100
                kpis['avg_inventory_pct'] = df_inv['pct'].mean()
                
    except Exception as e:
        print(f"Error extracting KPIs: {e}")
    
    return kpis


@app.route('/run-multi-simulation', methods=['POST'])
def run_multi_simulation():
    """Run multiple simulations in parallel using multiprocessing."""
    try:
        data = request.get_json() or {}
        
        horizon_days = int(data.get('horizon_days', 365))
        random_opening = data.get('random_opening', True)
        num_runs = int(data.get('num_runs', 1))
        
        # Cap at 100 runs
        num_runs = min(num_runs, 100)
        
        # Use a random base seed for all runs
        import random
        base_seed = random.randint(1, 100000)
        
        # Prepare arguments for each run
        run_args = [
            (i, horizon_days, random_opening, base_seed)
            for i in range(num_runs)
        ]
        
        # Use number of CPU cores, capped at 4 for stability
        max_workers = min(multiprocessing.cpu_count(), 4, num_runs)
        
        kpis = []
        errors = []
        
        # Run simulations in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_single_sim_for_kpi, args): args[0] for args in run_args}
            
            for future in as_completed(futures):
                result = future.result()
                if 'error' in result:
                    errors.append(result)
                else:
                    kpis.append(result)
        
        # Sort KPIs by run index
        kpis.sort(key=lambda x: x.get('run_idx', 0))
        
        return jsonify({
            'success': len(kpis) > 0,
            'kpis': kpis,
            'errors': errors,
            'total_runs': num_runs,
            'successful_runs': len(kpis)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)