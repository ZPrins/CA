# replit.md

## Overview

This project is a discrete-event simulation system for Cement Australia's supply chain logistics, modeling clinker production, milling, rail transport, port operations, and coastal shipping. It uses SimPy for simulation and provides both headless analytics with KPI reporting and a Tkinter-based visualization for animating transport movements. The system reads configuration from Excel workbooks and generates interactive HTML network visualizations using vis-network.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Core Simulation Engine
- **SimPy-based discrete-event simulation** for modeling supply chain flows
- Time-bucketed progression (Hours/Half Days/Days) for production and consumption
- Configurable horizon (default 365 days) with hourly step resolution

### Data Model Components
- **Stores**: Inventory buffers with capacity limits and opening stock ranges
- **Make Units**: Production equipment that consumes inputs and produces outputs at configured rates
- **Move Routes**: Transport routes with load/unload rates, travel times, and payload capacities
- **Demands**: Consumption patterns drawing from local stores

### Configuration System
- Primary config via `sim_config.py` dataclass with simulation parameters
- Visualization config via `supply_chain_viz_config.py` for network layout
- Excel workbook inputs with sheets: Network, Settings, Make, Store, Move, Deliver
- Auto-generation of normalized workbooks from source `Model Inputs.xlsx`

### Output Generation
- CSV exports for inventory snapshots, action logs, and store levels
- Interactive Plotly HTML charts for inventory visualization
- Standalone frozen Python model generation for reproducibility
- Text-based simulation logs

### Web Interface (app.py)
- **Flask-based dashboard** at port 5000 for interactive simulation control
- Configure simulation parameters: horizon days, random seed, random opening stock
- Run simulations directly from browser with real-time output logging
- View interactive HTML reports and download CSV exports
- Settings passed via environment variables (SIM_HORIZON_DAYS, SIM_RANDOM_OPENING, SIM_RANDOM_SEED)

### Visualization Layer
- **PyVis/vis-network** for static network graph HTML export
- Swimlane layout with fixed grid positioning (rows by Location, columns by Product Class)
- Pitchfork annotations for simplified Move edge rendering
- Live animation capability reading from simulation CSV logs

### Key Design Decisions
1. **Dataclass-based configuration**: Chosen over CLI-heavy approach for easier modification by planners/engineers
2. **Composite keys for stores**: Format `{product}|{location}|{equipment}|{input}` enables unique identification
3. **Min-fill-percent production rule**: Make units produce to the store with lowest percentage full
4. **Random opening stock**: Optional uniform sampling between Low/High bounds for stochastic runs

## External Dependencies

### Python Packages (requirements.txt)
- **simpy>=4.0**: Discrete-event simulation engine
- **pandas>=2.2.3**: Data manipulation and Excel/CSV I/O
- **networkx>=3.1**: Graph data structure for supply chain network
- **pyvis>=0.3.2**: Interactive network visualization generation
- **plotly**: Interactive charting for inventory profiles
- **openpyxl>=3.1**: Excel file reading/writing
- **jinja2>=3.1**: HTML template rendering
- **flask**: Web framework for interactive dashboard

### Frontend Libraries (bundled in lib/)
- **vis-network 9.1.2**: Browser-based network visualization
- **tom-select**: Dropdown/select UI component
- **Bootstrap 5.0**: CSS framework for HTML styling (CDN)

### Data Sources
- `Model Inputs.xlsx`: Source workbook with supply chain configuration
- `generated_model_inputs.xlsx`: Normalized workbook for simulation input