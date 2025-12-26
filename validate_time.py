import pandas as pd
import numpy as np

df = pd.read_csv('sim_outputs/sim_outputs_sim_log.csv')
vessels = df[df['vessel_id'].notna()]['vessel_id'].unique()
horizon = 365 * 24.0

print(f"{'Vessel ID':<10} | {'Total Time':<12} | {'Diff':<10}")
print("-" * 40)

for vid in sorted(vessels):
    total_time = df[df['vessel_id'] == vid]['time'].sum()
    diff = total_time - horizon
    print(f"{vid:<10} | {total_time:<12.6f} | {diff:<10.6e}")
