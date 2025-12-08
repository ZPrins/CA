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
