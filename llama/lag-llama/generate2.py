import matplotlib
matplotlib.use('Agg') # For headless server environments

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
START_DATE = '2025-01-01 00:00'
END_DATE = '2025-01-08 00:00'
FREQUENCY = '5min'
OUTPUT_FILE = 'realistic_traffic_data.csv'
FLOW_GROUP_FILE = 'realistic_flow_group_mapping.csv'
PLOT_OUTPUT = 'traffic_analysis_v2.png'
NUM_FLOWS_PER_GROUP = 334

# --- 1. REALISTIC PATTERN GENERATOR ---

def generate_diurnal_on_off(time_index, params):
    """
    Wraps heavy-tailed ON/OFF sources inside a diurnal (daily) cycle.
    This satisfies the research requirement for 'Internet-like' profiles 
    that show both high variability and peak-hour trends[cite: 18, 24].
    """
    length = len(time_index)
    
    # A. Create the Diurnal Base (Sinusoidal Peak Hours)
    # Traffic peaks at 14:00 (2 PM) and hits a trough at 02:00 (2 AM)
    daily_cycle = params.get('base_level', 10) + params.get('amplitude', 8) * np.sin(
        2 * np.pi * (time_index.hour + time_index.minute/60 - 8) / 24
    )
    base_traffic = np.maximum(1, daily_cycle)
    
    # B. Generate the Noah/Joseph Effect Bursts (Pareto ON/OFF) [cite: 40, 68]
    bursts = np.zeros(length)
    alpha_on = params.get('alpha_on', 1.5)
    alpha_off = params.get('alpha_off', 1.2)
    mean_burst_mag = params.get('mean_traffic', 20.0)
    
    num_subflows = 15 # Aggregation creates self-similarity [cite: 68]
    
    for _ in range(num_subflows):
        cursor = 0
        while cursor < length:
            on_dur = int((np.random.pareto(alpha_on) + 1) * 2) 
            off_dur = int((np.random.pareto(alpha_off) + 1) * 5)
            
            start_on = cursor
            end_on = min(cursor + on_dur, length)
            
            # Source emits data trains during the ON period [cite: 71, 81]
            bursts[start_on:end_on] += np.random.normal(mean_burst_mag/num_subflows, 2)
            cursor += on_dur + off_dur
            
    # Combine the base cycle with the stochastic bursts
    return np.maximum(0, base_traffic + bursts)

# --- 2. DATA GENERATOR WRAPPER ---

def generate_data(time_index, config):
    return pd.DataFrame({
        'traffic_volume_Tbits': generate_diurnal_on_off(time_index, config['params'])
    }, index=time_index)

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

# --- 4. MASTER TEMPLATES (All now follow diurnal cycles) ---

master_templates = [
    {
        'group': 'Realistic_Core', 
        'params': {
            'alpha_on': 1.7, 'alpha_off': 1.3, 'mean_traffic': 15.0,
            'base_level': 20, 'amplitude': 10  # Core backbone often has higher baseline
        },
        'anomalies': [{'type': 'spike', 'start_time': '2025-01-05 14:00', 'duration_min': 20, 'magnitude_traffic': 40.0}],
        'count': NUM_FLOWS_PER_GROUP
    },
    {
        'group': 'Bursty_Daily', 
        'params': {
            'alpha_on': 1.4, 'alpha_off': 1.1, 'mean_traffic': 10.0,
            'base_level': 10, 'amplitude': 8   # standard user profile [cite: 23]
        },
        'anomalies': [{'type': 'drift', 'start_time': '2025-01-06 09:00', 'duration_min': 300, 'magnitude_traffic': 15.0}],
        'count': NUM_FLOWS_PER_GROUP
    },
    {
        'group': 'High_Variance', 
        'params': {
            'alpha_on': 1.2, 'alpha_off': 1.05, 'mean_traffic': 8.0,
            'base_level': 5, 'amplitude': 4    # High volatility, low baseline
        },
        'anomalies': [{'type': 'spike', 'start_time': '2025-01-07 19:00', 'duration_min': 15, 'magnitude_traffic': 25.0}],
        'count': NUM_FLOWS_PER_GROUP
    },
]

# --- 5. EXECUTION ---

def generate_full_realistic_dataset():
    time_index = pd.date_range(start=START_DATE, end=END_DATE, freq=FREQUENCY, inclusive='left')
    all_flows_data = []
    group_mappings = []

    print(f"Generating {NUM_FLOWS_PER_GROUP * 3} flows with Diurnal ON/OFF patterns...")

    for template in master_templates:
        for i in range(template['count']):
            source_asn = template['template_asn']['source'] + i if 'template_asn' in template else 1000 + i
            dest_asn = template['template_asn']['dest'] + i if 'template_asn' in template else 4000 + i
            flow_id = f"{source_asn}-{dest_asn}_{template['group']}"

            flow_df = generate_data(time_index, {'params': template['params']})
            flow_df['flow_key_id'] = flow_id
            flow_df['is_anomaly'] = False
            flow_df['timestamp'] = flow_df.index
            flow_df['flow_group'] = template['group']
            
            for anomaly_dict in template['anomalies']:
                a_copy = anomaly_dict.copy()
                a_type = a_copy.pop('type')
                flow_df = add_anomaly(flow_df, anomaly_type=a_type, **a_copy)

            group_mappings.append({'flow_key_id': flow_id, 'flow_group': template['group']})
            all_flows_data.append(flow_df.reset_index(drop=True))

    final_df = pd.concat(all_flows_data, ignore_index=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    pd.DataFrame(group_mappings).to_csv(FLOW_GROUP_FILE, index=False)
    return final_df

def visualize_and_save(final_df):
    print(f"Saving visualization to {PLOT_OUTPUT}...")
    groups = final_df['flow_group'].unique()
    fig, axes = plt.subplots(len(groups), 1, figsize=(15, 12), sharex=True)
    
    for i, group in enumerate(groups):
        flow_id = final_df[final_df['flow_group'] == group]['flow_key_id'].iloc[0]
        flow_data = final_df[final_df['flow_key_id'] == flow_id]
        axes[i].plot(flow_data['timestamp'], flow_data['traffic_volume_Tbits'], 
                     label=f'Traffic ({group})', color='tab:blue', alpha=0.7, linewidth=0.8)
        
        anomalies = flow_data[flow_data['is_anomaly'] == True]
        if not anomalies.empty:
            axes[i].scatter(anomalies['timestamp'], anomalies['traffic_volume_Tbits'], 
                            color='red', s=20, label='Anomaly')
        axes[i].set_title(f"Diurnal Realistic Flow: {group}")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT)
    print("Done.")

if __name__ == "__main__":
    df = generate_full_realistic_dataset()
    visualize_and_save(df)
