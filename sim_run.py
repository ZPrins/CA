import sys
import os
import pandas as pd

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


def main():
    if len(sys.argv) > 1:
        INPUT_FILE = sys.argv[1]
    else:
        INPUT_FILE = "generated_model_inputs.xlsx"

    print(f"Starting simulation with input file: {INPUT_FILE}")

    # 1. Load & Clean
    raw_data = load_data_frames(INPUT_FILE)
    settings = get_config_map(raw_data.get('Settings', pd.DataFrame()))
    clean_data = clean_all_data(raw_data)

    # 2. Build Objects
    stores_cfg = build_store_configs(clean_data.get('Store', pd.DataFrame()))
    makes = build_make_units(clean_data.get('Make', pd.DataFrame()))
    moves = build_transport_routes(clean_data)
    demands = build_demands(clean_data.get('Deliver', pd.DataFrame()))

    # --- SAFETY CHECK ---
    print(f"\nModel Summary:")
    print(f"  Stores:  {len(stores_cfg)}")
    print(f"  Makes:   {len(makes)}")
    print(f"  Moves:   {len(moves)}")
    print(f"  Demands: {len(demands)}")

    if len(stores_cfg) == 0:
        print("\n[ERROR] No stores were loaded! Please check inputs.")
        return

        # 3. Configure
    settings.update(run_settings)

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

    if 'out_dir' not in settings:
        settings['out_dir'] = config.out_dir

    sim = SupplyChainSimulation(settings)
    sim.run(stores_cfg, makes, moves, demands)

    # 4. Report
    out_dir = settings.get('out_dir', config.out_dir)

    write_csv_outputs(sim, out_dir)
    plot_results(sim, out_dir, moves)
    generate_standalone(settings, stores_cfg, makes, moves, demands, out_dir)

    # --- MOVEMENT SUMMARY ---
    df_log = pd.DataFrame(sim.action_log)
    n_ship_loads = len(df_log[df_log['event'] == 'ShipLoad']) if not df_log.empty else 0
    n_train_loads = len(df_log[df_log['event'] == 'Loaded']) if not df_log.empty else 0

    print(f"\n=== Transport Summary ===")
    print(f"  Ship Loads:  {n_ship_loads}")
    print(f"  Train Loads: {n_train_loads}")
    print(f"  Total Unmet: {sum(sim.unmet.values()):,.0f} tons")
    print(f"\nAll complete! Check '{out_dir}' for results.")


if __name__ == "__main__":
    main()