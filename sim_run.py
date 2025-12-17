import sys
import os
import json
import time
import pandas as pd

# Check if quiet mode is enabled (for multi-run simulations)
QUIET_MODE = os.environ.get('SIM_QUIET_MODE', 'false').lower() == 'true'

def log(msg):
    """Print message only if not in quiet mode."""
    if not QUIET_MODE:
        print(msg)

# Data Loading Modules
from sim_run_data_ingest import load_data_frames, get_config_map
from sim_run_data_clean import clean_all_data
from sim_run_data_factory import build_store_configs, build_make_units, build_transport_routes, build_demands

# Core Logic
from sim_run_config import config, run_settings
from sim_run_core import SupplyChainSimulation

# Reporting Modules
from sim_run_report_csv import write_csv_outputs
from sim_run_report_plot import plot_results
from sim_run_report_codegen import generate_standalone

TEMP_OVERRIDES_FILE = 'temp_model_overrides.json'


def apply_ui_overrides(raw_data: dict) -> dict:
    """Apply user overrides from the UI to the raw data before cleaning."""
    if not os.path.exists(TEMP_OVERRIDES_FILE):
        return raw_data
    
    try:
        with open(TEMP_OVERRIDES_FILE, 'r') as f:
            overrides = json.load(f)
        
        # Apply Store overrides
        if 'store' in overrides and 'Store' in raw_data and not raw_data['Store'].empty:
            df = raw_data['Store'].copy()
            for i, override in enumerate(overrides['store']):
                if i < len(df):
                    # Map UI field names to DataFrame column names
                    if 'Silo Max Capacity' in override:
                        df.loc[df.index[i], 'Silo Max Capacity'] = override['Silo Max Capacity']
                    if 'Silo Opening Stock (High)' in override:
                        df.loc[df.index[i], 'Silo Opening Stock (High)'] = override['Silo Opening Stock (High)']
                    if 'Silo Opening Stock (Low)' in override:
                        df.loc[df.index[i], 'Silo Opening Stock (Low)'] = override['Silo Opening Stock (Low)']
                    if 'Load Rate (ton/hr)' in override:
                        df.loc[df.index[i], 'Load Rate (ton/hr)'] = override['Load Rate (ton/hr)']
                    if 'Unload Rate (ton/hr)' in override:
                        df.loc[df.index[i], 'Unload Rate (ton/hr)'] = override['Unload Rate (ton/hr)']
            raw_data['Store'] = df
            log(f"  [INFO] Applied UI overrides to Store data ({len(overrides['store'])} rows)")
        
        # Apply Move_SHIP overrides
        if 'move_ship' in overrides and 'Move_SHIP' in raw_data and not raw_data['Move_SHIP'].empty:
            df = raw_data['Move_SHIP'].copy()
            for i, override in enumerate(overrides['move_ship']):
                if i < len(df):
                    if '# Vessels' in override:
                        df.loc[df.index[i], '# Vessels'] = override['# Vessels']
                    if 'Route avg Speed (knots)' in override:
                        df.loc[df.index[i], 'Route avg Speed (knots)'] = override['Route avg Speed (knots)']
                    if '#Holds' in override:
                        df.loc[df.index[i], '#Holds'] = override['#Holds']
                    if 'Payload per Hold' in override:
                        df.loc[df.index[i], 'Payload per Hold'] = override['Payload per Hold']
            raw_data['Move_SHIP'] = df
            log(f"  [INFO] Applied UI overrides to Move_SHIP data ({len(overrides['move_ship'])} rows)")
        
    except Exception as e:
        log(f"  [WARNING] Failed to apply UI overrides: {e}")
    
    return raw_data


def _check_supply_sources(stores_cfg, makes, moves, demands):
    """Check for stores with demand but no supply sources (production, rail, or ship)."""
    
    # Build set of stores that receive supply
    supplied_stores = set()
    
    # 1. Production outputs
    for make in makes:
        for candidate in make.candidates:
            if candidate.out_store_key:
                supplied_stores.add(candidate.out_store_key.upper())
            if candidate.out_store_keys:
                for key in candidate.out_store_keys:
                    if key:
                        supplied_stores.add(key.upper())
    
    # 2. Transport destinations (rail and ship)
    for move in moves:
        if move.dest_stores:
            for store_key in move.dest_stores:
                if store_key:
                    supplied_stores.add(store_key.upper())
        # Also check ship itineraries for unload destinations
        if move.itineraries:
            for it in move.itineraries:
                for step in it:
                    if step.get('kind') == 'unload':
                        store_key = step.get('store_key')
                        if store_key:
                            supplied_stores.add(store_key.upper())
    
    # 3. Get stores with demand
    demand_stores = set()
    for demand in demands:
        if demand.store_key:
            demand_stores.add(demand.store_key.upper())
    
    # 4. Find unsupplied stores (have demand but no supply)
    unsupplied = []
    for store_key_upper in demand_stores:
        if store_key_upper not in supplied_stores:
            # Get store config for initial level info
            store_cfg = next((s for s in stores_cfg if s.key.upper() == store_key_upper), None)
            initial = (store_cfg.opening_low + store_cfg.opening_high) / 2 if store_cfg else 0
            unsupplied.append((store_key_upper, initial))
    
    if unsupplied:
        log(f"\n[WARNING] Stores with demand but NO supply sources:")
        for store_key, initial in sorted(unsupplied):
            init_note = f" (initial: {initial:,.0f}t)" if initial > 0 else ""
            log(f"  - {store_key}{init_note}")
        log("  These stores will deplete and cause unmet demand.\n")


def extract_kpis_from_sim(sim):
    """Extract KPIs directly from simulation object (no file I/O)."""
    kpis = {
        'total_unmet_demand': 0,
        'total_production': 0,
        'avg_inventory_pct': 0,
        'ship_trips': 0,
        'train_trips': 0,
        'unmet_by_key': {}
    }
    
    try:
        # Total unmet demand and breakdown by key
        kpis['total_unmet_demand'] = sum(sim.unmet.values())
        kpis['unmet_by_key'] = dict(sim.unmet)
        
        # Process action log
        if sim.action_log:
            df_log = pd.DataFrame(sim.action_log)
            
            # Total production
            prod = df_log[(df_log['process'] == 'Make') & (df_log['event'] == 'Produce')]
            if not prod.empty and 'qty' in prod.columns:
                kpis['total_production'] = prod['qty'].astype(float).sum()
            
            # Ship trips (unloads = completed trips)
            ship_moves = df_log[(df_log['process'] == 'Move') & (df_log['equipment'] == 'Ship') & (df_log['event'] == 'Unload')]
            kpis['ship_trips'] = len(ship_moves)
            
            # Train trips (unloads = completed trips)
            train_moves = df_log[(df_log['process'] == 'Move') & (df_log['equipment'] == 'Train') & (df_log['event'] == 'Unload')]
            kpis['train_trips'] = len(train_moves)
        
        # Average inventory utilization from snapshots
        if sim.inventory_snapshots:
            df_inv = pd.DataFrame(sim.inventory_snapshots)
            if 'level' in df_inv.columns and 'capacity' in df_inv.columns:
                df_inv['pct'] = (df_inv['level'] / df_inv['capacity']) * 100
                kpis['avg_inventory_pct'] = df_inv['pct'].mean()
                
    except Exception as e:
        log(f"[WARNING] Error extracting KPIs: {e}")
    
    return kpis


def run_simulation(input_file="generated_model_inputs.xlsx", artifacts='full', settings_override=None):
    """
    Run simulation and return results.
    
    Args:
        input_file: Path to Excel input file
        artifacts: 'full' to generate all outputs, 'kpi_only' to skip file generation
        settings_override: Optional dict to override settings
    
    Returns:
        dict with 'success', 'kpis', and optionally 'sim' object
    """
    total_start = time.time()
    step_start = time.time()
    
    log(f"Starting simulation with input file: {input_file}")

    # 1. Load & Clean
    raw_data = load_data_frames(input_file)
    
    # Apply any UI overrides before cleaning
    raw_data = apply_ui_overrides(raw_data)
    
    settings = get_config_map(raw_data.get('Settings', pd.DataFrame()))
    clean_data = clean_all_data(raw_data)

    # 2. Build Objects
    stores_cfg = build_store_configs(clean_data.get('Store', pd.DataFrame()))
    makes = build_make_units(clean_data.get('Make', pd.DataFrame()))
    moves = build_transport_routes(clean_data)
    demands = build_demands(clean_data.get('Deliver', pd.DataFrame()))

    load_elapsed = int(time.time() - step_start)
    step_start = time.time()
    
    # --- SAFETY CHECK ---
    log(f"\nModel Summary: (loaded in {load_elapsed}s)")
    log(f"  Stores:  {len(stores_cfg)}")
    log(f"  Makes:   {len(makes)}")
    log(f"  Moves:   {len(moves)}")
    log(f"  Demands: {len(demands)}")

    # --- CHECK FOR STORES WITH NO SUPPLY SOURCES ---
    _check_supply_sources(stores_cfg, makes, moves, demands)

    if len(stores_cfg) == 0:
        log("\n[ERROR] No stores were loaded! Please check inputs.")
        return {'success': False, 'error': 'No stores loaded', 'kpis': {}}

    # 3. Configure
    settings.update(run_settings)

    # Apply environment variable overrides
    if 'SIM_HORIZON_DAYS' in os.environ:
        try:
            settings['horizon_days'] = int(os.environ['SIM_HORIZON_DAYS'])
        except:
            pass
    if 'SIM_RANDOM_OPENING' in os.environ:
        val = os.environ['SIM_RANDOM_OPENING'].lower()
        settings['random_opening'] = (val == 'true')
    if 'SIM_RANDOM_SEED' in os.environ:
        try:
            settings['random_seed'] = int(os.environ['SIM_RANDOM_SEED'].strip())
        except:
            pass
    
    # Apply explicit settings override
    if settings_override:
        settings.update(settings_override)

    if 'out_dir' not in settings:
        settings['out_dir'] = config.out_dir

    sim = SupplyChainSimulation(settings)
    sim.run(stores_cfg, makes, moves, demands)
    
    sim_elapsed = int(time.time() - step_start)
    step_start = time.time()
    log(f"\nSimulation completed in {sim_elapsed}s")
    
    # Extract KPIs from simulation object (always)
    kpis = extract_kpis_from_sim(sim)

    # 4. Generate reports only if artifacts='full'
    if artifacts == 'full':
        out_dir = settings.get('out_dir', config.out_dir)
        write_csv_outputs(sim, out_dir)
        
        # Get graph sequence from Network sheet for proper ordering
        # Extract (Location, Equipment Name, Process) tuples in order of appearance
        network_df = raw_data.get('Network', pd.DataFrame())
        graph_sequence = []
        if not network_df.empty:
            seen = set()
            for _, row in network_df.iterrows():
                loc = row.get('Location')
                equip = row.get('Equipment Name')
                proc = row.get('Process')
                if pd.notna(loc) and pd.notna(equip) and proc in ['Make', 'Store']:
                    key = (str(loc), str(equip), str(proc))
                    if key not in seen:
                        seen.add(key)
                        graph_sequence.append(key)
        
        plot_results(sim, out_dir, moves, makes, graph_sequence)
        generate_standalone(settings, stores_cfg, makes, moves, demands, out_dir)

    # --- MOVEMENT SUMMARY ---
    df_log = pd.DataFrame(sim.action_log) if sim.action_log else pd.DataFrame()
    if not df_log.empty:
        n_ship_loads = len(df_log[(df_log['event'] == 'Load') & (df_log['equipment'] == 'Ship')])
        n_train_loads = len(df_log[(df_log['event'] == 'Load') & (df_log['equipment'] == 'Train')])
        ship_tons = df_log[(df_log['event'] == 'Load') & (df_log['equipment'] == 'Ship')]['qty'].sum()
        train_tons = df_log[(df_log['event'] == 'Load') & (df_log['equipment'] == 'Train')]['qty'].sum()
    else:
        n_ship_loads = n_train_loads = 0
        ship_tons = train_tons = 0

    log(f"\n=== Transport Summary ===")
    log(f"  Ship Loads:  {n_ship_loads} ({ship_tons:,.0f} tons)")
    log(f"  Train Loads: {n_train_loads} ({train_tons:,.0f} tons)")
    log(f"  Total Unmet: {sum(sim.unmet.values()):,.0f} tons")
    
    # Top 5 Lost Demand by product/location
    if sim.unmet:
        sorted_unmet = sorted(sim.unmet.items(), key=lambda x: x[1], reverse=True)[:5]
        log(f"\n=== Top 5 Lost Demand ===")
        for i, (key, val) in enumerate(sorted_unmet, 1):
            log(f"  {i}. {key}: {val:,.0f} t")
    
    total_elapsed = int(time.time() - total_start)
    
    if artifacts == 'full':
        report_elapsed = int(time.time() - step_start)
        out_dir = settings.get('out_dir', config.out_dir)
        log(f"\nReports generated in {report_elapsed}s")
        log(f"Total runtime: {total_elapsed}s")
        log(f"All complete! Check '{out_dir}' for results.")
    
    return {'success': True, 'kpis': kpis}


def main():
    if len(sys.argv) > 1:
        INPUT_FILE = sys.argv[1]
    else:
        INPUT_FILE = "generated_model_inputs.xlsx"
    
    # Check if running in KPI-only mode
    kpi_only = os.environ.get('SIM_KPI_ONLY', 'false').lower() == 'true'
    artifacts = 'kpi_only' if kpi_only else 'full'
    
    result = run_simulation(INPUT_FILE, artifacts=artifacts)
    
    if not result['success']:
        log(f"\n[ERROR] Simulation failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()