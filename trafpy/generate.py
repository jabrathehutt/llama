import trafpy.generator as tpg
from trafpy.generator.src.dists import val_dists
import pandas as pd
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION (Aligned with your previous code) ---
START_DATE = '2025-01-01 00:00'
END_DATE = '2025-01-08 00:00'
FREQUENCY = '10min'  # Aligned with CESNET/TrafPy standard
OUTPUT_FILE = 'trafpy_master_univariate_data.csv'
NUM_FLOWS_PER_GROUP = 2 # Set to 334 for your full dataset

# AS Constants
AS_IDS = {'AS100': 100, 'AS200': 200, 'AS300': 300, 'AS400': 400}

# --- TRAFPY VOLUME ENGINE ---

def generate_stochastic_volume(time_index, is_anomaly_array):
    """Generates bytes using TrafPy distributions based on anomaly state."""
    volumes = []
    events_per_interval = 50
    
    for i in range(len(time_index)):
        if not is_anomaly_array[i]:
            # Normal: Lognormal distribution
            flow_sizes = val_dists.gen_lognormal_dist(14, 2, 1, 1e7, events_per_interval)
            multiplier = 1.0
        else:
            # Anomaly: Exponential heavy-tail
            flow_sizes = val_dists.gen_exponential_dist(0.00001, 1, 1e7, events_per_interval)
            multiplier = 2.5 # Volume shift magnitude
            
        # Convert sum of bytes to Tbits for your specific column format
        volumes.append((sum(flow_sizes) * multiplier) / 1e12)
        
    return np.array(volumes)

# --- MASTER TEMPLATES ---

master_templates = [
    {
        'group': 'TrafPy_Backbone_Linear',
        'asn': {'src': 1001, 'dst': 4001, 'hnd': AS_IDS['AS200'], 'nex': AS_IDS['AS300']},
        'anomaly_window': ('2025-01-07 04:00', '2025-01-07 10:00')
    },
    {
        'group': 'TrafPy_AS_Shift',
        'asn': {'src': 5001, 'dst': 5002, 'hnd': AS_IDS['AS300'], 'nex': AS_IDS['AS100']},
        'anomaly_window': ('2025-01-07 12:00', '2025-01-07 18:00')
    }
]

# --- EXECUTION ---

def generate_full_master_dataset():
    time_index = pd.date_range(start=START_DATE, end=END_DATE, freq=FREQUENCY, inclusive='left')
    all_flows = []

    print(f"Generating Master Dataset: {len(time_index)} timestamps per flow.")

    for template in master_templates:
        print(f"Processing Group: {template['group']}")
        for i in tqdm(range(NUM_FLOWS_PER_GROUP)):
            
            # Metadata
            src_asn = template['asn']['src'] + i
            dst_asn = template['asn']['dst'] + i
            flow_id = f"{src_asn}-{dst_asn}_{template['group']}"
            
            # Ground Truth
            is_anomaly = (time_index >= template['anomaly_window'][0]) & \
                         (time_index <= template['anomaly_window'][1])
            
            # Generate Stochastic Volume
            traffic_volume = generate_stochastic_volume(time_index, is_anomaly)
            
            # Construct DataFrame
            df = pd.DataFrame({
                'timestamp': time_index,
                'traffic_volume_Tbits': traffic_volume,
                'is_anomaly': is_anomaly,
                'flow_key_id': flow_id,
                'sourceAS': src_asn,
                'destinationAS': dst_asn,
                'handoverAS': template['asn']['hnd'],
                'nexthopAS': template['asn']['nex']
            })
            all_flows.append(df)

    final_df = pd.concat(all_flows, ignore_index=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(final_df)} rows to {OUTPUT_FILE}")
    return final_df

if __name__ == "__main__":
    generate_full_master_dataset()
