# sim_run_report_codegen.py
from pathlib import Path
from datetime import datetime
from pprint import pformat
import math


def generate_standalone(settings, stores, makes, moves, demands, out_dir: Path):
    """
    Generates a standalone Python script ('sim_outputs_simpy_model.py').
    """
    ts = datetime.now().strftime('%Y-%m-%d %H:%M')

    def fmt(obj):
        return pformat(obj, width=100, indent=2, sort_dicts=False)

    code_lines = [
        '"""',
        'sim_outputs_simpy_model.py — Standalone simulation model',
        f'Generated on {ts}',
        '"""',
        '',
        'from sim_run_core import SupplyChainSimulation',
        # NEW: Import types from the types file in the generated code
        'from sim_run_types import StoreConfig, ProductionCandidate, MakeUnit, TransportRoute, Demand',
        '',
        f'SETTINGS = {fmt(settings)}',
        '',
        f'STORES = {fmt(stores)}',
        '',
        f'MAKES = {fmt(makes)}',
        '',
        f'MOVES = {fmt(moves)}',
        '',
        f'DEMANDS = {fmt(demands)}',
        '',
        'def main():',
        '    sim = SupplyChainSimulation(SETTINGS)',
        '    sim.run(STORES, MAKES, MOVES, DEMANDS)',
        '',
        "if __name__ == '__main__':",
        '    main()',
    ]

    standalone_path = out_dir / 'sim_outputs_simpy_model.py'
    standalone_path.write_text('\n'.join(code_lines), encoding='utf-8')
    print(f'Standalone model: {standalone_path}')