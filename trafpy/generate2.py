import trafpy.generator as tpg
from trafpy.generator.src.dists import val_dists
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

# --- CONFIGURATION ---
START_DATE = '2025-01-01 00:00'
END_DATE = '2025-01-08 00:00'
FREQUENCY = '10min' 
OUTPUT_FILE = 'trafpy_master_univariate_data.csv'
NUM_FLOWS = 5

def generate_hard_master_dataset():
    time_index = pd.date_range(START_DATE, END_DATE, freq=FREQUENCY, inclusive='left')
    all_flows = []
    freq_seconds = 600 

    print(f"Generating {NUM_FLOWS} flows with overlapping distributions...")

    for f_idx in range(NUM_FLOWS):
        flow_id = f"Flow_{f_idx}"
        volumes = np.zeros(len(time_index))
        is_anomaly = np.zeros(len(time_index), dtype=bool)
        actual_starts = [None] * len(time_index)
        
        # 1. Baseline: mu=14.0, sigma=3.0 (Higher variance makes tails fatter)
        for i in range(len(time_index)):
            flow_sizes = val_dists.gen_lognormal_dist(14.0, 3.0, 1, 1e7, 50)
            volumes[i] = sum(flow_sizes) / 1e12

        # 2. Schedule 10 Stealthy Anomalies per flow
        for _ in range(10):
            start_idx = random.randint(50, len(time_index) - 20)
            duration = random.randint(3, 8)
            
            # Sub-interval Jitter for Sampling Lag
            jitter_seconds = random.randint(1, freq_seconds - 1)
            true_start = time_index[start_idx] - pd.Timedelta(seconds=jitter_seconds)
            
            indices = np.arange(start_idx, start_idx + duration)
            is_anomaly[indices] = True
            
            for idx in indices:
                actual_starts[idx] = true_start
                # HARD MODE: Anomaly mu (14.7) is very close to Baseline mu (14.0)
                # No '+ 0.0006' forced shift here!
                mu = 14.7 if random.random() > 0.5 else 14.3
                flow_sizes = val_dists.gen_lognormal_dist(mu, 3.2, 1, 1e8, 50)
                volumes[idx] = sum(flow_sizes) / 1e12

        df = pd.DataFrame({
            'timestamp': time_index,
            'traffic_volume_Tbits': volumes,
            'is_anomaly': is_anomaly,
            'actual_start': actual_starts,
            'flow_key_id': flow_id
        })
        all_flows.append(df)

    pd.concat(all_flows, ignore_index=True).to_csv(OUTPUT_FILE, index=False)
    print(f"Hard-mode dataset generated: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_hard_master_dataset()
