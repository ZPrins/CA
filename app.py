from flask import Flask, render_template, request, jsonify, send_from_directory, Response, stream_with_context
import time
import subprocess
import os
import json
import tempfile
from pathlib import Path
import pandas as pd

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
                    store_data = df.to_dict('records')
                    break
            
            # Load Move_Ship sheet
            for sheet_name in xls.keys():
                if sheet_name.lower() == 'move_ship':
                    df = xls[sheet_name].dropna(how='all')
                    df.columns = [str(c).strip() for c in df.columns]
                    move_ship_data = df.to_dict('records')
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


if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)