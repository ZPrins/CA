# Cement Australia – Supply Chain Simulation (SimPy)

Config-driven, modular simulation of production, storage, rail moves, and shipping. The model reads a normalized Excel workbook (or CSVs), builds a SimPy model, and writes HTML/CSV reports. An optional Flask web UI lets you run the sim from your browser.

---

## Quick start

### Prerequisites
- Python 3.10+
- `pip install -r requirements.txt`

Windows (PowerShell):
```powershell
cd "C:\Users\Zephien\oneforecast.com.au\OneForecast - Documents\Projects\Cement Australia\13 Simulation\06 Python"
python -m venv .venv
. .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux (bash):
```bash
cd "~/oneforecast.com.au/OneForecast - Documents/Projects/Cement Australia/13 Simulation/06 Python"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Run the simulation (CLI)
- Default input is `generated_model_inputs.xlsx` (in the project root).
```bash
python sim_run.py
# or explicitly
python sim_run.py generated_model_inputs.xlsx
```
Environment overrides (optional):
```powershell
$env:SIM_HORIZON_DAYS=90
$env:SIM_RANDOM_OPENING=true
$env:SIM_RANDOM_SEED=42
python sim_run.py
```
Outputs are written to `sim_outputs/`:
- `sim_outputs_plots_all.html` (Plotly HTML)
- `sim_outputs_sim_log.csv` (action log)
- `sim_outputs_inventory_daily.csv`, `sim_outputs_store_levels.csv`, `sim_outputs_unmet_demand.csv`
- `sim_outputs_simpy_model.py` (standalone model snapshot)

### Run the web UI (optional)
A lightweight Flask app to kick off runs and download outputs.
```bash
python app.py
# Open http://localhost:5000
```
- Streams progress live.
- Shows a link to the generated HTML report and CSVs in `sim_outputs/`.

---

## Project structure (key files)
- `sim_run.py` — Orchestrates a run end-to-end (ingest → clean → factory → core → reports).
- `sim_run_config.py` — Default runtime settings (output folder, plotting toggles, etc.).
- Layer 1: Ingestion/cleaning
  - `sim_run_data_ingest.py` — Reads a single Excel or split CSVs into DataFrames (case-insensitive sheet names).
  - `sim_run_data_clean.py` — Normalizes columns and derives keys/payloads/times; resolves Store/Move/Demand linkages.
- Layer 2: Object factory
  - `sim_run_data_factory.py` — Builds `StoreConfig`, `MakeUnit`, `TransportRoute`, `Demand` objects.
- Layer 3: Core simulation
  - `sim_run_core.py` — Simulation container: builds stores, starts make/move/deliver processes, logging and snapshots.
  - `sim_run_core_store.py` — Store building helpers.
  - `sim_run_core_make.py` — Production logic with selectable output choice rule.
  - `sim_run_core_move_train.py` — Rail transport logic.
  - `sim_run_core_move_ship.py` — Ship transport logic and berth handling.
  - `sim_run_core_deliver.py` — Continuous demand draw from destination stores.
  - `sim_run_types.py` — Dataclasses used across layers.
- Reporting
  - `sim_run_report_csv.py` — Writes CSV outputs.
  - `sim_run_report_plot.py` — Generates Plotly HTML with inventory and flow markers.
  - `sim_run_report_codegen.py` — Emits a standalone snapshot of the built model.
- Web UI
  - `app.py` — Simple Flask app that runs `sim_run.py` and serves the report/CSVs.
  - `templates/index.html` — Minimal UI template.
- Utilities / extras
  - `supply_chain_viz.py` — Graph + input-prep utilities (uses NetworkX/PyVis) to help build the workbook.
  - `lib/*` — Static JS/CSS assets used by the report or tools.

---

## Inputs and data rules
- Supported input: a single Excel workbook (recommended) or separate CSV files per sheet.
- Recognized sheet names (case-insensitive): `Settings`, `Store`, `Make`, `Deliver`, `Move_TRAIN`, `Move_SHIP`, `SHIP_ROUTES`, `SHIP_BERTHS`, `SHIP_DISTANCES`, `Network`.

### Store keys
Each store is identified by `"{Product_Class}|{Location}|{Equipment}|{Product_Class}"` and is built from `Store` rows.

### Train origin/destination selection
- Move_TRAIN may specify `Origin Store` and `Destination Store`.
  - When present, the cleaner resolves these to exact store keys and uses them as the sole candidates for that route.
  - When absent or unresolvable, it falls back to all stores for `(Location, Product_Class)` and preserves the row order from the `Store` sheet.
- Practical example: Railton → Devonport with `Origin Store = GP_STORE_ST` will source strictly from `GP_STORE_ST`.

### Rail full-load rule
Train moves require full payloads by default (`require_full_payload=True`). A train will only start loading if both are true:
- Origin store has at least one full payload available; and
- Destination store has at least one full payload worth of free capacity.
If either condition is not met, the process waits and retries. This avoids blocked unloads.

### Timing and payloads
- If `Distance` is provided (km) and `To_Min/Back_Min` are empty, speeds fill the travel times.
- Payload is derived from `# Carraiges` × `# Carraige Capacity (ton)` (supports common column variants).

### Demand
- `Deliver` sheet supports either `Annual Demand (Tons)` (converted to hourly) or `Demand per Location/Hour`.

---

## Configuration tips
- See `sim_run_config.py` for defaults. Common keys:
  - `horizon_days` (int)
  - `require_full_payload` (bool, default True for both Train/Ship in this build)
  - `debug_full_payload` (bool)
  - `demand_truck_load_tons`, `demand_step_hours`
  - `write_plots`, `autoscale_default`, `out_dir`
- Environment variables used by `sim_run.py` (take precedence if set):
  - `SIM_HORIZON_DAYS`, `SIM_RANDOM_OPENING`, `SIM_RANDOM_SEED`

---

## Troubleshooting
- `FileNotFoundError: generated_model_inputs.xlsx`
  - Ensure the file exists in the project root or pass a path to `sim_run.py`.
- `ImportError` for Excel engine when reading `.xlsx`
  - `pip install openpyxl` (included via `requirements.txt`).
- No trains appear to move
  - Check that payload and load/unload rates are positive in `Move_TRAIN`.
  - Ensure origin/destination store keys resolve (see selection rules above).
- Report HTML is empty
  - Ensure `write_plots` is enabled in `sim_run_config.py`.

---

## Docker
This project is packaged for Docker. You can build and run the entire application (including the web UI) in a container.

### 1. Build the Docker Image
You can use the provided scripts to (re)build the Docker image after making changes to the project.

**Windows (PowerShell):**
```powershell
.\build_docker.ps1
```

**macOS/Linux/Git Bash:**
```bash
chmod +x build_docker.sh
./build_docker.sh
```

### 2. Run the Docker Container
Once built, run the container:
```bash
docker run -p 5000:5000 simulation-app
```
The application will be available at `http://localhost:5000`.

---

## License / attribution
Internal model for Cement Australia logistics. Built with SimPy and Plotly. See `requirements.txt` for third‑party packages.
