import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib

# --- CONFIGURATION (UPDATED FOR CLEAN DATA) ---
START_DATE = '2025-01-01 00:00'
END_DATE = '2025-01-08 00:00'
FREQUENCY = '5min'
OUTPUT_FILE = 'clean_training_traffic_data.csv'  # Reflects clean data
FLOW_GROUP_FILE = 'clean_flow_group_mapping.csv'
NUM_FLOWS_PER_GROUP = 334

# AS-Konstanten
AS_IDS = {'AS100': 100, 'AS200': 200, 'AS300': 300, 'AS400': 400}

# --- 1. PATTERN GENERATORS (UNCHANGED) ---

def generate_linear(time_index, params):
    total_minutes = (time_index - time_index[0]).total_seconds() / 60
    traffic = params['traffic_start'] + total_minutes * params.get('traffic_slope', 0.0)
    return np.maximum(0, traffic)

def generate_sinus(time_index, params):
    total_minutes = (time_index - time_index[0]).total_seconds() / 60
    period_daily = 24 * 60 / params.get('traffic_daily_freq', 1)
    shift_daily = params.get('traffic_phase', 0) * 60
    traffic = params['traffic_offset'] + params['traffic_amplitude'] * np.sin(
        2 * np.pi * (total_minutes - shift_daily) / period_daily
    )
    return np.maximum(0, traffic)

def generate_jumps(time_index, params):
    """Note: These are periodic cycles, NOT anomalies."""
    total_minutes = (time_index - time_index[0]).total_seconds() / 60
    length = len(time_index)
    traffic = params['traffic_start'] + total_minutes * params.get('traffic_slope', 0.0)
    jump_interval_min = params['jump_interval_min']
    jump_amplitude = params['jump_amplitude']
    interval_steps = int(jump_interval_min / 5)
    jumps = np.zeros(length)
    for i in range(0, length, interval_steps * 2):
        if i < length: jumps[i] += jump_amplitude
        if i + interval_steps < length: jumps[i + interval_steps] -= jump_amplitude
    return np.maximum(0, traffic + jumps)

PATTERN_MAPPINGS = {'linear': generate_linear, 'sinus': generate_sinus, 'jumps': generate_jumps}

# --- 2. DATA GENERATOR (TRAFFIC ONLY) ---

def generate_data(time_index, config):
    traffic_volume = PATTERN_MAPPINGS[config['pattern']](time_index, config['params'])
    return pd.DataFrame({'traffic_volume_Tbits': traffic_volume}, index=time_index)

# --- 3. MASTER TEMPLATES (ANOMALIES REMOVED) ---

master_templates = [
    # GRUPPE 1: PURE LINEAR FLOWS (Training Baseline)
    {
        'group': 'Pure_Linear', 'pattern': 'linear',
        'template_asn': {'source': 1001, 'dest': 4001, 'handover': AS_IDS['AS200'], 'nexthop': AS_IDS['AS300']},
        'params': {'traffic_start': 40.0, 'traffic_slope': 0.005},
        'anomalies': [], # REMOVED FOR CLEAN DATA
        'count': NUM_FLOWS_PER_GROUP
    },
    # GRUPPE 2: SINUS FLOWS (Training Baseline)
    {
        'group': 'Pure_Sinus', 'pattern': 'sinus',
        'template_asn': {'source': 5001, 'dest': 5002, 'handover': AS_IDS['AS300'], 'nexthop': AS_IDS['AS100']},
        'params': {'traffic_offset': 20.0, 'traffic_amplitude': 10.0, 'traffic_daily_freq': 1, 'traffic_phase': 0},
        'anomalies': [], # REMOVED FOR CLEAN DATA
        'count': NUM_FLOWS_PER_GROUP
    },
    # GRUPPE 3: JUMP PATTERNS (Periodic Base - NO Spike Anomalies)
    {
        'group': 'Jump_Patterns', 'pattern': 'jumps',
        'template_asn': {'source': 6001, 'dest': 6002, 'handover': AS_IDS['AS100'], 'nexthop': AS_IDS['AS400']},
        'params': {'traffic_start': 10.0, 'traffic_slope': 0.0001, 'jump_interval_min': 240, 'jump_amplitude': 5.0},
        'anomalies': [], # REMOVED FOR CLEAN DATA
        'count': NUM_FLOWS_PER_GROUP
    },
]

# --- 4. MAIN EXECUTION LOOP ---

def generate_full_master_dataset():
    time_index = pd.date_range(start=START_DATE, end=END_DATE, freq=FREQUENCY, inclusive='left')
    all_flows_data = []
    group_mappings = []

    print(f"Generating CLEAN dataset: {sum(t['count'] for t in master_templates)} flows.")

    for template in master_templates:
        for i in range(template['count']):
            source_asn = template['template_asn']['source'] + i
            dest_asn = template['template_asn']['dest'] + i
            flow_id = f"{source_asn}-{dest_asn}_{template['pattern']}"

            params_copy = template['params'].copy()
            params_copy['traffic_start'] = params_copy.get('traffic_start', 0.0) + (i * 0.01)

            flow_df = generate_data(time_index, {'pattern': template['pattern'], 'params': params_copy})
            flow_df['flow_key_id'] = flow_id
            flow_df['is_anomaly'] = False
            flow_df['timestamp'] = flow_df.index

            # Metadata
            flow_df['sourceAS'] = source_asn
            flow_df['destinationAS'] = dest_asn
            flow_df['handoverAS'] = template['template_asn']['handover']
            flow_df['nexthopAS'] = template['template_asn']['nexthop']

            group_mappings.append({'flow_key_id': flow_id, 'flow_group': template['group']})
            all_flows_data.append(flow_df.reset_index(drop=True))

    final_df = pd.concat(all_flows_data, ignore_index=True)
    final_df['timestamp'] = pd.to_datetime(final_df['timestamp'])

    final_df.to_csv(OUTPUT_FILE, index=False)
    pd.DataFrame(group_mappings).to_csv(FLOW_GROUP_FILE, index=False)

    print(f"\nCLEAN MASTER DATASET GENERATED: {len(final_df)} rows.")
    print(f"Total Ground Truth Anomalies: {final_df['is_anomaly'].sum()} (Should be 0)")
    return final_df

if __name__ == "__main__":
    final_df = generate_full_master_dataset()

