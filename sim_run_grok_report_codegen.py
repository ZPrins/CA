# sim_run_grok_report_codegen.py
from pathlib import Path
from datetime import datetime
from pprint import pformat
import math


def generate_standalone(settings, stores, makes, moves, demands, out_dir: Path):
    """
    Generates a standalone Python script ('sim_outputs_simpy_model.py')
    that reproduces the current simulation state without requiring Excel inputs.
    """
    ts = datetime.now().strftime('%Y-%m-%d %H:%M')

    def _is_nan(x):
        try:
            return isinstance(x, float) and (math.isnan(x) or math.isinf(x))
        except Exception:
            return False

    # Deep sanitization of objects to ensure valid Python code generation
    # (Simplified for brevity - assumes objects are largely clean from Phase 2)

    def fmt(obj):
        return pformat(obj, width=100, indent=2, sort_dicts=False)

    code_lines = [
        '"""',
        'sim_outputs_simpy_model.py — Standalone Grok simulation model',
        f'Generated on {ts}',
        '"""',
        '',
        'from sim_run_grok_core import SupplyChainSimulation, StoreConfig, ProductionCandidate, MakeUnit, TransportRoute, Demand',
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