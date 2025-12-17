from flask import Flask, render_template, request, jsonify, send_from_directory, Response, stream_with_context
import time
import os
import json
import subprocess
import tempfile
from pathlib import Path
import pandas as pd

from sim_run_config import config
from sim_run_single import (
    run_single_simulation_subprocess,
    run_single_simulation_blocking,
    check_report_exists,
    get_csv_files
)
from sim_run_multi import (
    run_multi_simulation,
    stream_multi_simulation
)

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Track running processes for stop functionality
running_processes = []
stop_requested = False

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
def run_simulation_route():
    """Run a single simulation (blocking mode)."""
    try:
        data = request.get_json() or {}
        
        horizon = int(data.get('horizon_days', 365))
        random_opening = data.get('random_opening', True)
        random_seed = data.get('random_seed', None)
        
        seed_str = '' if random_seed in ('', None) else str(int(random_seed))
        
        result = run_single_simulation_blocking(horizon, random_opening, seed_str)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'output': str(e)})


@app.route('/stop-simulation', methods=['POST'])
def stop_simulation():
    """Stop all running simulation processes."""
    global running_processes, stop_requested
    stop_requested = True
    terminated = 0
    
    for proc in running_processes[:]:
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                terminated += 1
        except Exception:
            pass
        running_processes.remove(proc)
    
    return jsonify({'success': True, 'terminated': terminated})


@app.route('/stream-simulation')
def stream_simulation():
    """SSE endpoint that streams live logs from a single simulation."""
    global running_processes, stop_requested
    stop_requested = False
    
    try:
        horizon = int(request.args.get('horizon_days', 365))
        random_opening = request.args.get('random_opening', 'true').lower() == 'true'
        rs = request.args.get('random_seed', '')
        seed_str = str(int(rs)) if rs and rs.strip() else ''

        proc = run_single_simulation_subprocess(horizon, random_opening, seed_str)
        running_processes.append(proc)

        def sse_lines():
            try:
                yield 'event: start\ndata: Simulation started\n\n'
                last_heartbeat = time.time()
                report_notified = False

                if proc.stdout is not None:
                    while True:
                        if stop_requested:
                            yield 'data: Simulation stopped by user\n\n'
                            break
                            
                        line = proc.stdout.readline()
                        if line:
                            msg = line.rstrip('\r\n')
                            if msg:
                                if msg.startswith('Progress:'):
                                    yield f'event: progress\ndata: {msg}\n\n'
                                else:
                                    yield f'data: {msg}\n\n'
                                # Check if report is ready after HTML generation message
                                if not report_notified and check_report_exists():
                                    report_notified = True
                                    yield 'event: report_ready\ndata: true\n\n'
                            last_heartbeat = time.time()
                        else:
                            if proc.poll() is not None:
                                break
                            now = time.time()
                            if now - last_heartbeat >= 1.5:
                                yield ': keep-alive\n\n'
                                last_heartbeat = now
                            time.sleep(0.2)

                proc.wait()

                was_stopped = stop_requested
                payload = {
                    'success': proc.returncode == 0 and not was_stopped,
                    'report_ready': check_report_exists(),
                    'exit_code': proc.returncode,
                    'stopped': was_stopped
                }
                yield 'event: done\ndata: ' + json.dumps(payload) + '\n\n'
            finally:
                try:
                    if proc.poll() is None:
                        proc.terminate()
                    if proc in running_processes:
                        running_processes.remove(proc)
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
    """Get current simulation status."""
    return jsonify({
        'report_exists': check_report_exists(),
        'csv_files': get_csv_files()
    })


@app.route('/run-multi-simulation', methods=['POST'])
def run_multi_simulation_route():
    """Run multiple simulations in parallel (blocking mode)."""
    try:
        data = request.get_json() or {}
        
        horizon_days = int(data.get('horizon_days', 365))
        random_opening = data.get('random_opening', True)
        num_runs = int(data.get('num_runs', 1))
        
        result = run_multi_simulation(horizon_days, random_opening, num_runs)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/stream-multi-simulation')
def stream_multi_simulation_route():
    """SSE endpoint that streams progress for multi-run simulations."""
    global stop_requested
    stop_requested = False
    
    try:
        horizon_days = int(request.args.get('horizon_days', 365))
        random_opening = request.args.get('random_opening', 'true').lower() == 'true'
        num_runs = int(request.args.get('num_runs', 1))
        
        def stop_check():
            return stop_requested
        
        def sse_wrapper():
            for msg in stream_multi_simulation(horizon_days, random_opening, num_runs, stop_check):
                yield msg
        
        headers = {
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        }
        return Response(stream_with_context(sse_wrapper()), mimetype='text/event-stream', headers=headers)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)