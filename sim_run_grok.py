import sys
import pandas as pd

# Data Loading Modules
from sim_run_grok_data_ingest import load_data_frames, get_config_map
from sim_run_grok_data_clean import clean_all_data
from sim_run_grok_data_factory import build_store_configs, build_make_units, build_transport_routes, build_demands

# Core Logic
from sim_run_grok_config import config, run_settings
from sim_run_grok_core import SupplyChainSimulation

# NEW: Reporting Modules (Replacing helpers)
from sim_run_grok_report_csv import write_csv_outputs
from sim_run_grok_report_plot import plot_results
from sim_run_grok_report_codegen import generate_standalone


def main():
    if len(sys.argv) > 1:
        INPUT_FILE = sys.argv[1]
    else:
        INPUT_FILE = "generated_model_inputs.xlsx"

    print(f"Starting simulation with input file prefix: {INPUT_FILE}")

    # 1. Load & Clean
    raw_data = load_data_frames(INPUT_FILE)
    settings = get_config_map(raw_data.get('Settings', pd.DataFrame()))
    clean_data = clean_all_data(raw_data)

    # 2. Build Objects
    stores_cfg = build_store_configs(clean_data.get('Store', pd.DataFrame()))
    makes = build_make_units(clean_data.get('Make', pd.DataFrame()))
    moves = build_transport_routes(clean_data)
    demands = build_demands(clean_data.get('Deliver', pd.DataFrame()))

    # 3. Configure & Run
    settings.update(run_settings)

    # Optional: Override settings from config object if not in Excel
    if 'out_dir' not in settings:
        settings['out_dir'] = config.out_dir

    sim = SupplyChainSimulation(settings)
    sim.run(stores_cfg, makes, moves, demands)

    # 4. Report
    out_dir = settings.get('out_dir', config.out_dir)

    write_csv_outputs(sim, out_dir)
    plot_results(sim, out_dir, moves)
    generate_standalone(settings, stores_cfg, makes, moves, demands, out_dir)

    print(f"\nAll complete! Check '{out_dir}' for results.")


if __name__ == "__main__":
    main()