# Cement Australia Supply Chain Simulation

## Overview

This is a config-driven, modular discrete-event simulation of a cement supply chain using SimPy. The system models production, storage, rail transport, shipping, and customer demand fulfillment. It reads configuration from a normalized Excel workbook (or CSVs), runs a SimPy simulation, and generates HTML/CSV reports with Plotly visualizations.

The simulation covers:
- **Production (Make)**: Cement mills and kilns producing various product classes
- **Storage (Store)**: Silos and warehouses with capacity constraints
- **Transport (Move)**: Rail and ship movements between locations
- **Demand (Deliver)**: Customer deliveries via truck

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Core Simulation Engine
- **Framework**: SimPy 4.0+ for discrete-event simulation
- **Time Unit**: Hours (configurable via `step_hours`)
- **Default Horizon**: 365 days

### Module Structure

The codebase follows a layered architecture:

1. **Configuration Layer** (`sim_run_config.py`)
   - Dataclass-based settings with sensible defaults
   - Environment variable overrides supported (SIM_HORIZON_DAYS, SIM_RANDOM_OPENING, SIM_RANDOM_SEED)

2. **Data Ingestion Layer** (`sim_run_data_ingest.py`, `sim_run_data_clean.py`, `sim_run_data_factory.py`)
   - Reads Excel workbook with sheets: Settings, Store, Make, Deliver, Move_TRAIN, Move_SHIP, etc.
   - Cleans and normalizes data
   - Builds typed dataclass objects (StoreConfig, MakeUnit, TransportRoute, Demand)

3. **Simulation Core** (`sim_run_core.py`)
   - `SupplyChainSimulation` class orchestrates the model
   - Delegates to specialized process modules:
     - `sim_run_core_store.py`: Container initialization
     - `sim_run_core_make.py`: Production processes
     - `sim_run_core_move_train.py`: Rail transport
     - `sim_run_core_move_ship.py`: Maritime transport with berth management
     - `sim_run_core_deliver.py`: Demand consumption

4. **Reporting Layer** (`sim_run_report_*.py`)
   - CSV outputs: inventory snapshots, action logs, final levels, unmet demand
   - Plotly HTML reports with interactive charts
   - Code generation: standalone model snapshot

### Type System (`sim_run_types.py`)
- `StoreConfig`: Storage location with capacity and initial levels
- `ProductionCandidate`: Output option for a production unit
- `MakeUnit`: Production equipment with multiple output candidates
- `TransportRoute`: Rail or ship route with payload, rates, and timing
- `Demand`: Customer demand specification

### Web Interface
- Flask app (`app.py`) provides a browser-based UI
- Template in `templates/index.html`
- Modular simulation runners:
  - `sim_run_single.py`: Single simulation with streaming output and full artifacts
  - `sim_run_multi.py`: Multi-run simulation with parallel execution and KPI extraction
- Features:
  - Single simulation with live console output and full HTML/CSV reports
  - Multi-run simulation (default 50 runs) with parallel processing (up to 4 workers)
  - Stop button to halt running simulations
  - Editable model parameters with change highlighting

### Visualization Tools
- `supply_chain_viz.py`: NetworkX/PyVis visualization of supply chain topology
- Configurable via `supply_chain_viz_config.py`

### HTML Report Features
The generated HTML report (`sim_outputs_plots_all.html`) includes:
- **Run Single Simulation button** with countdown timer based on previous run time
- **Styled progress badges**: Orange (running), purple (finishing), green (complete)
- **Previous inventory comparison**: Grey dotted "Prev Level" line on inventory charts
- **Filtering**: By product and location
- **Interactive charts**:
  - Inventory levels with capacity, demand, and flow data
  - Production output with downtime markers
  - Rail transport timeline
  - Ship route timelines by route group
  - Fleet utilization and state over time
  - **Route Summary**: Stacked bar chart showing average time by status (Loading, In Transit, Waiting for Berth, Unloading) with trip counts on right axis

### Variability Distributions Report
A separate HTML report (`sim_outputs_variability.html`) showing distribution curves for all stochastic elements:
- **Breakdown Duration**: Lognormal distribution (mean ~3h, σ=0.8) with theoretical PDF, sample histogram, and key statistics
- **Unplanned Downtime by Equipment**: Bar chart showing configured downtime percentages per equipment
- **Opening Stock Distribution**: Uniform distribution ranges for each store (when random opening is enabled)
- Link from main report summary for easy navigation

## External Dependencies

### Python Packages
- **pandas** (>=2.1): Data manipulation and Excel/CSV I/O
- **simpy** (>=4.0): Discrete-event simulation engine
- **openpyxl** (>=3.1): Excel file reading
- **plotly** (>=5.20): Interactive HTML charts
- **networkx** (>=3.1): Graph data structures
- **pyvis** (>=0.3.2): Network visualization
- **jinja2** (>=3.1): Template rendering
- **Flask** (>=3.0): Web UI
- **orjson**: Fast JSON serialization

### Data Inputs
- Primary: `generated_model_inputs.xlsx` (Excel workbook in project root)
- Alternative: Individual CSV files matching sheet names

### Output Directory
- All outputs written to `sim_outputs/`
- Key files: `sim_outputs_plots_all.html`, `sim_outputs_sim_log.csv`, `sim_outputs_inventory_daily.csv`

## Recent Changes

- Added Route Summary chart with stacked bars showing average time per status and trip counts
- Implemented countdown timer on Run Simulation button based on previous run time
- Added styled progress badges (orange/purple/green) for simulation status
- Added previous inventory comparison feature (grey dotted line)
- Optimized fleet utilization chart generation (reduced from 6s to <1s)
