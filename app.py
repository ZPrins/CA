from flask import Flask, render_template, request, jsonify, send_from_directory, Response, stream_with_context
import time
import os
import json
import subprocess
import tempfile
import re
import threading
import queue
from pathlib import Path
import pandas as pd

import sim_run
from sim_run_data_ingest import load_data_frames
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
import supply_chain_viz_config

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Track running processes for stop functionality
running_processes = []
stop_requested = False

OUT_DIR = Path(config.out_dir)
INPUT_FILE = 'generated_model_inputs.xlsx'
MODEL_INPUTS_FILE = 'Model Inputs.xlsx'
GENERATED_INPUTS_FILE = 'generated_model_inputs.xlsx'


# Global cache for Excel data to speed up simulation startup
excel_cache = {
    'data': None,
    'last_mtime': 0
}

def get_cached_raw_data():
    """Get raw data from cache or load it if file changed."""
    global excel_cache
    if not os.path.exists(INPUT_FILE):
        return None
    
    mtime = os.path.getmtime(INPUT_FILE)
    if excel_cache['data'] is None or mtime > excel_cache['last_mtime']:
        print(f"  [INFO] Cache miss or file updated. Loading '{INPUT_FILE}'...")
        excel_cache['data'] = load_data_frames(INPUT_FILE)
        excel_cache['last_mtime'] = mtime
    return excel_cache['data']

def load_model_params():
    """Load Store, Move_Ship, and SHIP_BERTHS parameters from Excel for UI editing."""
    store_data = []
    move_ship_data = []
    ship_berths_data = []
    
    raw_data = get_cached_raw_data()
    if raw_data:
        try:
            # Load Store sheet
            for sheet_name, df in raw_data.items():
                if sheet_name.lower() == 'store':
                    df = df.dropna(how='all')
                    df.columns = [str(c).strip() for c in df.columns]
                    df = df[[c for c in df.columns if not c.startswith('Unnamed')]]
                    store_data = df.fillna('').to_dict('records')
                
                elif sheet_name.lower() == 'move_ship':
                    df = df.dropna(how='all')
                    df.columns = [str(c).strip() for c in df.columns]
                    df = df[[c for c in df.columns if not c.startswith('Unnamed')]]
                    move_ship_data = df.fillna('').to_dict('records')
                
                elif sheet_name.lower() == 'ship_berths':
                    df = df.dropna(how='all')
                    df.columns = [str(c).strip() for c in df.columns]
                    df = df[[c for c in df.columns if not c.startswith('Unnamed')]]
                    ship_berths_data = df.fillna('').to_dict('records')
                    
        except Exception as e:
            print(f"Error extracting model params from cache: {e}")
    
    return {'store': store_data, 'move_ship': move_ship_data, 'ship_berths': ship_berths_data}


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


try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False

def json_dumps(data):
    if HAS_ORJSON:
        return orjson.dumps(data).decode('utf-8')
    return json.dumps(data)

def json_loads(data):
    if HAS_ORJSON:
        return orjson.loads(data)
    return json.loads(data)

@app.route('/api/save-params', methods=['POST'])
def save_params():
    """Save user-modified parameters to a temp file for simulation to use."""
    try:
        data = request.get_json() or {}
        with open(TEMP_OVERRIDES_FILE, 'wb') as f:
            if HAS_ORJSON:
                f.write(orjson.dumps(data))
            else:
                f.write(json.dumps(data).encode('utf-8'))
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
        
        seed_val = None
        if random_seed not in ('', None):
            try:
                seed_val = int(random_seed)
            except:
                pass
        
        settings_override = {
            'horizon_days': horizon,
            'random_opening': random_opening,
        }
        if seed_val is not None:
            settings_override['random_seed'] = seed_val
            
        raw_data = get_cached_raw_data()
        raw_data_copy = {k: df.copy() for k, df in raw_data.items()} if raw_data else None
        
        result = sim_run.run_simulation(
            input_file=INPUT_FILE,
            artifacts='full',
            settings_override=settings_override,
            raw_data=raw_data_copy
        )
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
        
        # In-process simulation execution for 10x faster startup
        log_queue = queue.Queue()
        
        def run_in_thread():
            try:
                raw_data = get_cached_raw_data()
                # Deep copy raw_data to avoid modifying the cache if cleaning/overrides do that
                raw_data_copy = {k: df.copy() for k, df in raw_data.items()} if raw_data else None
                
                settings_override = {
                    'horizon_days': horizon,
                    'random_opening': random_opening,
                }
                if seed_str:
                    settings_override['random_seed'] = int(seed_str)
                
                sim_run.run_simulation(
                    input_file=INPUT_FILE,
                    artifacts='full',
                    settings_override=settings_override,
                    log_callback=lambda msg: log_queue.put(msg),
                    raw_data=raw_data_copy
                )
            except Exception as e:
                log_queue.put(f"[ERROR] In-process simulation failed: {e}")
            finally:
                log_queue.put(None) # Sentinel for end of stream

        thread = threading.Thread(target=run_in_thread)
        thread.daemon = True
        thread.start()

        def sse_lines():
            try:
                yield 'event: start\ndata: Simulation started\n\n'
                report_notified = False

                while True:
                    if stop_requested:
                        yield 'data: Simulation stopped by user\n\n'
                        break
                        
                    try:
                        # Wait for a log message with a timeout to check stop_requested
                        msg = log_queue.get(timeout=1.0)
                        if msg is None: # Sentinel
                            break
                            
                        if msg.startswith('Progress:'):
                            yield f'event: progress\ndata: {msg}\n\n'
                        else:
                            yield f'data: {msg}\n\n'
                        
                        # Only notify report ready when we see the HTML generation message
                        if not report_notified and 'Interactive HTML report generated' in msg:
                            report_notified = True
                            yield 'event: report_ready\ndata: true\n\n'
                            
                    except queue.Empty:
                        yield ': keep-alive\n\n'
                        continue

                was_stopped = stop_requested
                payload = {
                    'success': not was_stopped,
                    'report_ready': check_report_exists(),
                    'stopped': was_stopped
                }
                yield 'event: done\ndata: ' + json.dumps(payload) + '\n\n'
            except Exception as e:
                yield f'data: SSE Error: {str(e)}\n\n'

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


VIZ_CONFIG_FILE = 'supply_chain_viz_config.py'
MODEL_INPUTS_FILE = 'Model Inputs.xlsx'
GENERATED_INPUTS_FILE = 'generated_model_inputs.xlsx'


def get_prepare_inputs_value():
    """Read prepare_inputs value from config file."""
    try:
        with open(VIZ_CONFIG_FILE, 'r') as f:
            content = f.read()
        match = re.search(r'prepare_inputs:\s*bool\s*=\s*(True|False)', content)
        if match:
            return match.group(1) == 'True'
    except Exception:
        pass
    return True


def set_prepare_inputs_value(value: bool):
    """Update prepare_inputs value in config file."""
    try:
        with open(VIZ_CONFIG_FILE, 'r') as f:
            content = f.read()
        new_value = 'True' if value else 'False'
        new_content = re.sub(
            r'(prepare_inputs:\s*bool\s*=\s*)(True|False)',
            f'\\g<1>{new_value}',
            content
        )
        with open(VIZ_CONFIG_FILE, 'w') as f:
            f.write(new_content)
        return True
    except Exception as e:
        print(f"Error updating prepare_inputs: {e}")
        return False


@app.route('/api/viz-config')
def get_viz_config():
    """Get visualization config settings."""
    return jsonify({
        'prepare_inputs': get_prepare_inputs_value()
    })


@app.route('/api/viz-config', methods=['POST'])
def update_viz_config():
    """Update visualization config settings."""
    try:
        data = request.get_json() or {}
        if 'prepare_inputs' in data:
            success = set_prepare_inputs_value(data['prepare_inputs'])
            return jsonify({'success': success})
        return jsonify({'success': False, 'error': 'No valid setting provided'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/network-map')
def network_map():
    """Serve the network map HTML file."""
    if os.path.exists('Network_Map.html'):
        return send_from_directory('.', 'Network_Map.html')
    return "Network map not found. Run Generate Outputs first.", 404


@app.route('/download/model-inputs')
def download_model_inputs():
    """Download Model Inputs.xlsx file."""
    if os.path.exists(MODEL_INPUTS_FILE):
        return send_from_directory('.', MODEL_INPUTS_FILE, as_attachment=True)
    return "File not found", 404


@app.route('/download/generated-inputs')
def download_generated_inputs():
    """Download generated_model_inputs.xlsx file."""
    if os.path.exists(GENERATED_INPUTS_FILE):
        return send_from_directory('.', GENERATED_INPUTS_FILE, as_attachment=True)
    return "File not found", 404


@app.route('/download/network-map')
def download_network_map():
    """Download Network_Map.html network map file."""
    network_map_file = 'Network_Map.html'
    if os.path.exists(network_map_file):
        return send_from_directory('.', network_map_file, as_attachment=True)
    return "File not found", 404


@app.route('/upload/model-inputs', methods=['POST'])
def upload_model_inputs():
    """Upload and replace Model Inputs.xlsx file."""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        file.save(MODEL_INPUTS_FILE)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/file-info/<filename>')
def get_file_info(filename):
    """Get file modification time."""
    file_map = {
        'model-inputs': MODEL_INPUTS_FILE,
        'generated-inputs': GENERATED_INPUTS_FILE,
        'network-map': 'Network_Map.html'
    }
    
    filepath = file_map.get(filename)
    if not filepath or not os.path.exists(filepath):
        return jsonify({'exists': False})
    
    mtime = os.path.getmtime(filepath)
    return jsonify({
        'exists': True,
        'mtime': mtime,
        'mtime_iso': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
    })


@app.route('/stream-data-prep')
def stream_data_prep():
    """SSE endpoint that runs supply_chain_viz.py and streams output."""
    try:
        proc = subprocess.Popen(
            ['python', 'supply_chain_viz.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        running_processes.append(proc)

        def sse_lines():
            try:
                yield 'event: start\ndata: Data prep started\n\n'
                
                if proc.stdout is not None:
                    for line in proc.stdout:
                        msg = line.rstrip('\r\n')
                        if msg:
                            yield f'data: {msg}\n\n'
                
                proc.wait()
                
                payload = {
                    'success': proc.returncode == 0,
                    'exit_code': proc.returncode
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
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)