# Cement Australia – Rail/Ship Logistics Simulation (SimPy)

This project models clinker production, milling, rail transport, port operations, and coastal shipping using SimPy. It includes:

- A headless analytic simulation that prints KPIs and optionally plots inventory profiles (`Main SimPy V2.py`).
- A Tkinter-based visualizer that animates trains and ships over a simple network map (`Main Visualizer.py`).

The code is intentionally compact so planners and engineers can tweak parameters in one place (`Config` classes inside each script).

---

## 1) Quick Start

### Prerequisites
- Python 3.10 or newer (3.11/3.12 also fine).
- Packages listed in `requirements.txt`.
- Tkinter for the visualizer:
  - Windows/macOS: usually bundled with the official Python installer.
  - Ubuntu/Debian: `sudo apt-get install python3-tk`

### Setup (recommended: virtual environment)

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

### Run
Analytic simulation (prints stats; can plot):
```bash
python "Main SimPy V2.py"
```

Visualizer (GUI animation):
```bash
python "Main Visualizer.py"
```

If you see `No module named 'simpy'`, ensure you activated the virtual environment and installed requirements.

---

## 2) Project Layout
- `Main SimPy V2.py` — core year-long simulation with:
  - `Config`: rates, capacities, downtimes, routes, and demands
  - `DowntimeScheduler`: creates planned and unplanned downtime events
  - Process functions: kiln, mill, train, ship, and destination demand
  - `DataCollector`: time-series capture of inventory levels
  - `run_simulation()`: entry point (duration, optional plotting)
  - `print_statistics()` and `plot_results()`

- `Main Visualizer.py` — lightweight animation with:
  - `Config`: same physical parameters (no downtime logic here)
  - `Visualizer`: simple map, tanks, and moving vehicle dots
  - `VehicleTracker`: animates trains/ships between nodes
  - SimPy processes: kiln, mill, train, ship, and demand
  - `run_visual_simulation()`: entry point that opens a Tkinter window

- `requirements.txt` — Python dependencies (SimPy, NumPy, Matplotlib). Tkinter is stdlib.

---

## 3) Configuration Guide
Tuning is centralized in the `Config` class at the top of each file. Key sections:

- Production
  - `KILN_MEAN_RATE`, `KILN_STD_DEV`
  - `MILL_MEAN_RATE`, `MILL_STD_DEV`, and (V2) `MILL_CONVERSION_RATE`
  - Downtime percents and derived annual hours (V2 only)

- Storage
  - `CL_CAPACITY`, `CL_INITIAL` (Railton clinker)
  - `GP_R_CAPACITY`, `GP_R_INITIAL` (Railton GP)
  - `GP_D_CAPACITY`, `GP_D_INITIAL` (Davenport GP)

- Rail
  - `TRAIN_CAPACITY`, `TRAINS_PER_DAY`, `TRAIN_TRAVEL_TIME`, `TRAIN_LOAD_TIME`, `TRAIN_UNLOAD_TIME`

- Shipping
  - `SHIPS`: max capacity, loading/unloading rates, travel time, frequency

- Demand and destination stores
  - `DESTINATIONS`/`DESTINATION` maps include capacity, initial stock, and `daily_demand`

Tips:
- Change one parameter at a time; rerun to see impacts on inventories and service levels.
- Use the headless sim (V2) to assess KPIs; use the visualizer to communicate flows and constraints.

---

## 4) Running the Analytic Simulation (V2)
- Open `Main SimPy V2.py` and review `run_simulation()` defaults:
  - `duration=8760` (hours, 1 year)
  - `show_plot=True` to render Matplotlib charts for inventories
- Start the sim:
```bash
python "Main SimPy V2.py"
```
- Outputs:
  - Console statistics (production totals, rail shipments, stockouts)
  - Optional plots: time-series of `CL`, `GP_Railton`, `GP_Davenport`, and destination stores

---

## 5) Running the Visualizer
- The visualizer simplifies some dynamics and focuses on animation:
```bash
python "Main Visualizer.py"
```
- Window shows:
  - Nodes: Railton, Davenport, Osborne, Newcastle, MCF
  - Lines: rail (solid), sea (dashed)
  - Tanks: inventory bars that change color (low = red, normal = orange, high = green)
  - Vehicles: red dot (train), blue dots (ships)

If the window freezes, ensure you are not running it inside a non-GUI terminal environment. On Windows, prefer running from a standard terminal.

---

## 6) Interpreting Results
- Watch for persistent low inventory (red) at destinations → increase ship frequency, capacity, or port stock; or boost rail throughput.
- High, persistent levels near capacity indicate potential overproduction or insufficient transport.
- Use downtime parameters (V2) to examine robustness to outages.

---

## 7) Troubleshooting
- `ModuleNotFoundError: No module named 'simpy'`
  - Activate venv and run `pip install -r requirements.txt`.
- `ImportError: No module named '_tkinter'` (Linux)
  - `sudo apt-get install python3-tk` and reinstall Python if needed.
- Matplotlib backend issues (servers/WSL)
  - Use `matplotlib.use('Agg')` for headless plots or run on a GUI-capable machine.
- Unicode/Path issues on Windows
  - Keep the project path as provided or wrap it in quotes when running from the terminal.

---

## 8) Reproducibility & Randomness
Both scripts use Python's `random` (and NumPy in V2 for plotting). For reproducible runs, seed the RNG at the top of the script before creating the environment, e.g.:
```python
import random
random.seed(42)
```

---

## 9) Extending the Model (Ideas)
- Add multiple berths at Davenport with a `simpy.Resource` capacity > 1.
- Introduce weather delays or loading constraints by season.
- Calibrate demand time-series rather than constant daily demand.
- Model multiple trains or stochastic train departures.
- Add cost/CO2 KPIs, service level penalties, and optimization loops.

---

## 10) License and Attribution
- Add a license file if the model will be shared externally (e.g., MIT/Apache-2.0).
- Attribution: internal model for Cement Australia logistics; built with SimPy.


---

## 11) New: Config‑driven Excel‑based simulation (no CLI needed)

This repository now includes a SimPy model that builds itself from a normalized Excel workbook and a double‑clickable runner. Use this when you want to simulate directly from your `Model Inputs.xlsx` (via a generated workbook) without passing command‑line flags.

### What’s included
- `sim_from_generated.py` — builds a SimPy model from a generated workbook (sheets: `Network`, `Settings`, `Make`, `Store`, `Move`, `Deliver`/`Delivery`).
- `sim_config.py` — a simple `Config` dataclass where you set your input file and options.
- `run_sim.py` — a runner that reads `sim_config.Config` and executes the simulation; friendly to double‑click on Windows.
- `supply_chain_viz.py` — utilities to normalize/prepare inputs and (optionally) generate `generated_model_inputs.xlsx` from `Model Inputs.xlsx`.

### Typical workflow (no command line)
1) Open `sim_config.py` and adjust settings:
   - `in_xlsx`: path to your generated workbook (default: `generated_model_inputs.xlsx`).
   - `product_class`: e.g., `"GP"` or `None` for all.
   - `override_days`: force horizon if you don’t want to use the `Settings` sheet value.
   - `write_csvs`, `open_folder_after`, `pause_on_finish`.
   - `auto_generate=True`: if `in_xlsx` is missing, `run_sim.py` will generate it from `Model Inputs.xlsx`.
2) Double‑click `run_sim.py` in Explorer.
   - The runner loads the config, optionally generates `generated_model_inputs.xlsx`, builds the model, runs it, and prints a summary.
   - If configured, it writes CSVs to `sim_outputs/` and opens that folder.

### What the simulation reads
- `Network` — topology of `Equipment@Location` nodes and process transitions (`Process`, `Input`, `Output`, `Next Location`, `Next Equipment`).
- `Settings` — `Number of Simulation Runs`, `Modeling Horizon (#Days)`, `Time Buckets (Days, Half Days, Hours)`.
- `Make` — per `(Location, Equipment Name, Input)` optional parameters such as `Mean Production Rate (Tons/hr)` and `Consumption %`.
- `Store` — capacities and opening stocks (high/low averaged) per `(Location, Equipment Name, Input)`.
- `Move` — fleet sizing and timing: `#Equipment (99-unlimited)`, `#Parcels`, `Capacity Per Parcel`, `Load Rate (Ton/hr)`, `Travel to Time (Min)`, `Unload Rate (Ton/Hr)`, `Travel back Time (Min)` for each `(Product Class, Location, Equipment Name, Next Location)`.
- `Deliver` (or `Delivery`) — demand per location, e.g., `Demand per Location`.

If some sheets/columns are missing, sensible defaults are applied (e.g., store capacity very large, opening stock 0, make rate 0, consumption 100%, zero move rates/times, demand 0).

### Outputs you’ll see
- Console report when the run finishes:
  - Ending stock levels for each store (`PC | Location | Equipment | Input`).
  - Unmet demand totals per `(PC, Location, Input)` if any.
  - Route stats: trips completed and tons moved for each move lane.
- Optional CSVs (if `write_csvs=True`) saved under `sim_outputs/`.

### Advanced: run the builder directly (optional CLI)
You can still run the builder head‑on if you prefer a terminal:
```bash
python sim_from_generated.py --xlsx "generated_model_inputs.xlsx" --product-class GP --days 60
```
If `--days` isn’t provided, the horizon comes from the `Settings` sheet. If `--product-class` is omitted, all product classes are included.

### Generating the workbook
If you only have `Model Inputs.xlsx`, you can create the normalized workbook using the runner with `auto_generate=True` (default) or manually via `supply_chain_viz`:
```python
from pathlib import Path
from supply_chain_viz import prepare_inputs_generate
prepare_inputs_generate(Path("Model Inputs.xlsx"), Path("generated_model_inputs.xlsx"))
```
This copies relevant sheets, unifies `Deliver`/`Delivery`, and ensures required rows/columns exist.

### Requirements
- See `requirements.txt` (typically includes `simpy`, `pandas`, `openpyxl`, and visualization extras for `supply_chain_viz.py`).
- On Windows, double‑clicking `run_sim.py` works best when `.py` files are associated with your Python interpreter.

### Troubleshooting (this flow)
- `RuntimeError: simpy isn't installed` — activate your venv and `pip install -r requirements.txt`.
- `FileNotFoundError: generated_model_inputs.xlsx` — leave `auto_generate=True` or run the generation step above.
- Demand doesn’t draw — ensure `Deliver` has a `Demand per Location` column and the locations exist as `Store` nodes for the same product class/input.
- Nothing moves — check `Move` sheet parameters (payload, load/unload rates, travel times). Zero rates/times result in no effective movement.

### Roadmap
Future enhancements may include maintenance calendars for `Make`, batching/queueing at loaders/unloaders, multi‑material transport, stochastic variability, and CSV metrics logging per time bucket.
