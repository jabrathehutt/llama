import pandas as pd
import numpy as np
import random
import os

# --- CONFIGURATION ---
START_DATE = '2025-01-01 00:00'
END_DATE = '2025-01-08 00:00'
FREQUENCY = '5min'
OUTPUT_FILE = 'realistic_traffic_data.csv'
FLOW_GROUP_FILE = 'realistic_flow_group_mapping.csv'
NUM_FLOWS_PER_GROUP = 334

AS_IDS = {'AS100': 100, 'AS200': 200, 'AS300': 300, 'AS400': 400}

# --- 1. REALISTIC PATTERN GENERATORS ---

def generate_heavy_tailed_on_off(time_index, params):
    """
    Implements the ON/OFF source model using Pareto distributions[cite: 45, 70].
    This creates realistic burstiness and self-similarity.
    """
    length = len(time_index)
    traffic = np.zeros(length)
    
    # Pareto shape parameters (alpha). alpha < 2 leads to infinite variance 
    alpha_on = params.get('alpha_on', 1.5)
    alpha_off = params.get('alpha_off', 1.2)
    mean_traffic = params.get('mean_traffic', 20.0)
    
    # We simulate multiple internal 'sub-flows' to aggregate 
    num_subflows = 15 
    
    for _ in range(num_subflows):
        cursor = 0
        while cursor < length:
            # ON period duration (Pareto)
            on_dur = int((np.random.pareto(alpha_on) + 1) * 2) 
            # OFF period duration (Pareto)
            off_dur = int((np.random.pareto(alpha_off) + 1) * 5)
            
            start_on = cursor
            end_on = min(cursor + on_dur, length)
            
            # During ON period, source emits traffic [cite: 51]
            traffic[start_on:end_on] += np.random.normal(mean_traffic/num_subflows, 2)
            
            cursor += on_dur + off_dur
            
    return np.maximum(0, traffic)

def generate_daily_bursty(time_index, params):
    """Combines a daily cycle with heavy-tailed bursts for realism."""
    base_traffic = 10 + 5 * np.sin(2 * np.pi * time_index.hour / 24)
    bursts = generate_heavy_tailed_on_off(time_index, params)
    return base_traffic + bursts

# --- 2. DATA GENERATOR WRAPPER ---

def generate_data(time_index, config):
    if config['pattern'] == 'realistic_burst':
        traffic = generate_heavy_tailed_on_off(time_index, config['params'])
    else:
        traffic = generate_daily_bursty(time_index, config['params'])
    return pd.DataFrame({'traffic_volume_Tbits': traffic}, index=time_index)

# --- 3. ANOMALY GENERATOR ---

def add_anomaly(df, anomaly_type, start_time, duration_min, **magnitudes):
    start_ts = pd.to_datetime(start_time)
    end_ts = start_ts + pd.Timedelta(minutes=duration_min)
    anomaly_indices = df[(df.index >= start_ts) & (df.index < end_ts)].index

    if not anomaly_indices.empty:
        mag_traffic = magnitudes.get('magnitude_traffic', 0.0)
        if anomaly_type == 'spike':
            df.loc[anomaly_indices, 'traffic_volume_Tbits'] += mag_traffic
        elif anomaly_type == 'drift':
            num_steps = len(anomaly_indices)
            df.loc[anomaly_indices, 'traffic_volume_Tbits'] += np.linspace(0, mag_traffic, num_steps)
        df.loc[anomaly_indices, 'is_anomaly'] = True
    return df

# --- 4. MASTER TEMPLATES ---

master_templates = [
    {
        'group': 'Realistic_Core', 'pattern': 'realistic_burst',
        'template_asn': {'source': 1001, 'dest': 4001, 'handover': 200, 'nexthop': 300},
        'params': {'alpha_on': 1.7, 'alpha_off': 1.3, 'mean_traffic': 45.0},
        'anomalies': [{'type': 'spike', 'start_time': '2025-01-05 10:00', 'duration_min': 20, 'magnitude_traffic': 50.0}],
        'count': NUM_FLOWS_PER_GROUP
    },
    {
        'group': 'Bursty_Daily', 'pattern': 'daily_burst',
        'template_asn': {'source': 5001, 'dest': 5002, 'handover': 300, 'nexthop': 100},
        'params': {'alpha_on': 1.4, 'alpha_off': 1.1, 'mean_traffic': 25.0},
        'anomalies': [{'type': 'drift', 'start_time': '2025-01-06 08:00', 'duration_min': 300, 'magnitude_traffic': 20.0}],
        'count': NUM_FLOWS_PER_GROUP
    },
    {
        'group': 'High_Variance', 'pattern': 'realistic_burst',
        'template_asn': {'source': 6001, 'dest': 6002, 'handover': 100, 'nexthop': 400},
        'params': {'alpha_on': 1.2, 'alpha_off': 1.05, 'mean_traffic': 15.0},
        'anomalies': [{'type': 'spike', 'start_time': '2025-01-07 18:00', 'duration_min': 15, 'magnitude_traffic': 30.0}],
        'count': NUM_FLOWS_PER_GROUP
    },
]

# --- 5. EXECUTION ---

def generate_full_realistic_dataset():
    time_index = pd.date_range(start=START_DATE, end=END_DATE, freq=FREQUENCY, inclusive='left')
    all_flows_data = []
    group_mappings = []

    for template in master_templates:
        for i in range(template['count']):
            source_asn = template['template_asn']['source'] + i
            dest_asn = template['template_asn']['dest'] + i
            flow_id = f"{source_asn}-{dest_asn}_{template['group']}"

            flow_df = generate_data(time_index, {'pattern': template['pattern'], 'params': template['params']})
            flow_df['flow_key_id'] = flow_id
            flow_df['is_anomaly'] = False
            flow_df['timestamp'] = flow_df.index
            
            for anomaly in template['anomalies']:
                flow_df = add_anomaly(flow_df, **anomaly)

            flow_df['sourceAS'] = source_asn
            flow_df['destinationAS'] = dest_asn
            flow_df['handoverAS'] = template['template_asn']['handover']
            flow_df['nexthopAS'] = template['template_asn']['nexthop']

            group_mappings.append({'flow_key_id': flow_id, 'flow_group': template['group']})
            all_flows_data.append(flow_df.reset_index(drop=True))

    final_df = pd.concat(all_flows_data, ignore_index=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    pd.DataFrame(group_mappings).to_csv(FLOW_GROUP_FILE, index=False)
    print(f"Realistic dataset generated: {len(final_df)} rows.")

if __name__ == "__main__":
    generate_full_realistic_dataset()
