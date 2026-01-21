import trafpy.generator as tpg
from trafpy.generator.src.dists import val_dists
import pandas as pd
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
START_DATE = '2025-01-01 00:00'
END_DATE = '2025-01-08 00:00'
FREQUENCY = '10min' 
OUTPUT_FILE = 'trafpy_master_univariate_data.csv'
NUM_FLOWS_PER_GROUP = 2 

def generate_stochastic_volume(time_index, is_anomaly_array):
    volumes = []
    events_per_interval = 50
    
    for i in range(len(time_index)):
        if not is_anomaly_array[i]:
            # Normal: mu=18. Scaled by 1e9 to stay in Gbit range.
            flow_sizes = val_dists.gen_lognormal_dist(_mu=18, _sigma=2, min_val=1, max_val=1e9, size=events_per_interval)
            val = sum(flow_sizes) / 1e9 
        else:
            # Anomaly: mu=24 + a forced baseline surge of 5000 Gbits
            flow_sizes = val_dists.gen_lognormal_dist(_mu=24, _sigma=2, min_val=1, max_val=1e12, size=events_per_interval)
            val = (sum(flow_sizes) / 1e9) + 5000 
            
        volumes.append(val)
    return np.array(volumes)

def generate_full_master_dataset():
    time_index = pd.date_range(start=START_DATE, end=END_DATE, freq=FREQUENCY, inclusive='left')
    all_flows = []
    
    master_templates = [
        {'group': 'Backbone_Surge', 'asn': {'src': 1001, 'dst': 4001}, 'anomaly_window': ('2025-01-07 04:00', '2025-01-07 10:00')},
        {'group': 'AS_Shift', 'asn': {'src': 5001, 'dst': 5002}, 'anomaly_window': ('2025-01-07 12:00', '2025-01-07 18:00')}
    ]

    print("Generating High-Contrast Gbit Dataset...")
    for template in master_templates:
        for i in range(NUM_FLOWS_PER_GROUP):
            is_anomaly = (time_index >= template['anomaly_window'][0]) & (time_index <= template['anomaly_window'][1])
            traffic_volume = generate_stochastic_volume(time_index, is_anomaly)
            
            df = pd.DataFrame({
                'timestamp': time_index,
                'traffic_volume_Gbits': traffic_volume, 
                'is_anomaly': is_anomaly,
                'flow_key_id': f"{template['asn']['src'] + i}-{template['asn']['dst'] + i}_{template['group']}"
            })
            all_flows.append(df)

    pd.concat(all_flows).to_csv(OUTPUT_FILE, index=False)
    print("Done. Verification: Anomaly points generated above 5000 Gbits.")

if __name__ == "__main__":
    generate_full_master_dataset()
