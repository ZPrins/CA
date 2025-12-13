# sim_run_grok.py - Main runner
import os
from sim_run_grok_config import config
from sim_run_grok_helpers import write_csv_outputs, plot_results, generate_standalone
from sim_run_grok_core import SupplyChainSimulation
from sim_run_grok_data_loader import load_data

if __name__ == "__main__":
    print("sim_run_grok.py: Starting simulation...")

    horizon_days = int(os.environ.get('SIM_HORIZON_DAYS', config.horizon_days))
    random_opening = os.environ.get('SIM_RANDOM_OPENING', str(config.random_opening)).lower() == 'true'
    random_seed_env = os.environ.get('SIM_RANDOM_SEED', '')
    random_seed = int(random_seed_env) if random_seed_env else config.random_seed

    settings_override = {
        "horizon_days": horizon_days,
        "random_opening": random_opening,
        "random_seed": random_seed,
    }

    settings, stores, makes, moves, demands = load_data()
    settings.update(settings_override)

    print(f"Loaded: {len(stores)} stores, {len(makes)} make units, {len(moves)} routes, {len(demands)} demands")

    sim = SupplyChainSimulation(settings)
    sim.run(stores, makes, moves, demands)

    out_dir = config.out_dir
    write_csv_outputs(sim, out_dir)
    plot_results(sim, out_dir)
    generate_standalone(settings, stores, makes, moves, demands, out_dir)

    print(f"\nAll complete! Check '{out_dir}' for results, plots, and standalone model.")