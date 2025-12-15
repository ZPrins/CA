from flask import Flask, render_template, request, jsonify, send_from_directory, Response, stream_with_context
import time
import subprocess
import os
from pathlib import Path

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route('/')
def index():
    from sim_run_grok_config import config
    out_dir = Path(config.out_dir)

    outputs_exist = (out_dir / 'sim_outputs_plots_all.html').exists()
    csv_files = list(out_dir.glob('*.csv')) if out_dir.exists() else []
    
    return render_template('index.html', 
                         config=config,
                         outputs_exist=outputs_exist,
                         csv_files=[f.name for f in csv_files])

@app.route('/run-simulation', methods=['POST'])
def run_simulation():
    try:
        data = request.get_json() or {}
        
        horizon = int(data.get('horizon_days', 365))
        random_opening = data.get('random_opening', True)
        random_seed = data.get('random_seed', None)
        if random_seed == '':
            random_seed = ''
        elif random_seed is not None:
            random_seed = str(int(random_seed))
        else:
            random_seed = ''
        
        env = os.environ.copy()
        env['SIM_HORIZON_DAYS'] = str(horizon)
        env['SIM_RANDOM_OPENING'] = str(random_opening)
        env['SIM_RANDOM_SEED'] = random_seed
        
        result = subprocess.run(
            ['python', 'sim_run_grok.py'],
            capture_output=True,
            text=True,
            timeout=300,
            env=env
        )
        
        success = result.returncode == 0
        output = result.stdout + result.stderr

        # Determine report path based on Grok config
        from sim_run_grok_config import config as grok_config
        out_dir = Path(grok_config.out_dir)
        report_html = out_dir / 'sim_outputs_plots_all.html'
        
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
        try:
            random_seed = str(int(rs)) if rs != '' else ''
        except Exception:
            random_seed = ''

        env = os.environ.copy()
        env['SIM_HORIZON_DAYS'] = str(horizon)
        env['SIM_RANDOM_OPENING'] = 'true' if random_opening else 'false'
        env['SIM_RANDOM_SEED'] = random_seed
        env['PYTHONUNBUFFERED'] = '1'

        # Start child process in unbuffered mode
        proc = subprocess.Popen(
            ['python', '-u', 'sim_run_grok.py'],
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
                # Inform client that stream started
                yield 'event: start\ndata: Simulation started\n\n'
                last_heartbeat = time.time()
                if proc.stdout is not None:
                    # Use readline loop to avoid buffering issues on Windows
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
                            # Send heartbeat every 1.5s to keep connection alive
                            now = time.time()
                            if now - last_heartbeat >= 1.5:
                                yield ': keep-alive\n\n'
                                last_heartbeat = now
                            time.sleep(0.2)
                proc.wait()
                # Check report existence
                from sim_run_grok_config import config as grok_config
                out_dir = Path(grok_config.out_dir)
                report_html = out_dir / 'sim_outputs_plots_all.html'
                payload = {
                    'success': proc.returncode == 0,
                    'report_ready': report_html.exists(),
                    'exit_code': proc.returncode,
                }
                import json as _json
                yield 'event: done\ndata: ' + _json.dumps(payload) + '\n\n'
            finally:
                # Ensure process is terminated if still running
                try:
                    if proc.poll() is None:
                        proc.terminate()
                except Exception:
                    pass
        headers = {
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',  # for Nginx: disable buffering
        }
        return Response(stream_with_context(sse_lines()), mimetype='text/event-stream', headers=headers)
    except Exception as e:
        # If we fail before starting stream, return JSON error
        return jsonify({'success': False, 'output': str(e)})

@app.route('/report')
def report():
    from sim_run_grok_config import config
    out_dir = Path(config.out_dir)
    report_path = out_dir / 'sim_outputs_plots_all.html'
    if report_path.exists():
        return send_from_directory(str(out_dir), 'sim_outputs_plots_all.html')
    return "Report not found. Run a simulation first.", 404

@app.route('/outputs/<filename>')
def get_output_file(filename):
    from sim_run_grok_config import config
    out_dir = Path(config.out_dir)
    return send_from_directory(str(out_dir), filename)

@app.route('/api/status')
def status():
    from sim_run_grok_config import config
    out_dir = Path(config.out_dir)
    return jsonify({
        'report_exists': (out_dir / 'sim_outputs_plots_all.html').exists(),
        'csv_files': [f.name for f in out_dir.glob('*.csv')] if out_dir.exists() else []
    })

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    # Enable threaded server to support streaming SSE without blocking
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
