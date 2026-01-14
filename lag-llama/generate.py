import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib

# --- CONFIGURATION ---
START_DATE = '2025-01-01 00:00'
END_DATE = '2025-01-08 00:00'  # One Week
FREQUENCY = '5min'
OUTPUT_FILE = 'network_metrics_data.csv'  # FINAL FILENAME
FLOW_GROUP_FILE = 'simplified_mixed_flow_mapping.csv'
NUM_FLOWS_PER_GROUP = 334

# Fixed AS Constants
AS_IDS = {
    'AS100': 100, 'AS200': 200, 'AS300': 300, 'AS400': 400
}


# --- 1. PATTERN GENERATORS (Deterministic, No Noise) ---
# (These remain the same as the previous deterministic, univariate version)

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
    total_minutes = (time_index - time_index[0]).total_seconds() / 60
    length = len(time_index)
    traffic = params['traffic_start'] + total_minutes * params.get('traffic_slope', 0.0)
    jump_interval_min = params['jump_interval_min']
    jump_amplitude = params['jump_amplitude']
    interval_steps = int(jump_interval_min / 5)
    jumps = np.zeros(length)
    for i in range(0, length, interval_steps * 2):
        if i < length:
            jumps[i] += jump_amplitude
        if i + interval_steps < length:
            jumps[i + interval_steps] -= jump_amplitude
    return np.maximum(0, traffic + jumps)


PATTERN_MAPPINGS = {
    'linear': generate_linear,
    'sinus': generate_sinus,
    'jumps': generate_jumps
}


# --- 2. UNIVARIATE DATA GENERATOR (Traffic Only) ---

def generate_univariate_data(time_index, config):
    """Generates ONLY Traffic Volume (Univariate Output)."""

    traffic_volume = PATTERN_MAPPINGS[config['pattern']](time_index, config['params'])

    return pd.DataFrame({
        'traffic_volume_Tbits': traffic_volume,
        # Flow Count and Error Rate columns are removed as per instruction
    }, index=time_index)


# --- 3. ANOMALY GENERATOR (Supports Spike and Drift) ---

def add_multivariate_anomaly(df, anomaly_type, start_time, duration_min, **magnitudes):
    """Fügt Sudden Spikes und Gradual Drifts hinzu."""

    mag_traffic = magnitudes.get('magnitude_traffic', 0.0)

    start_ts = pd.to_datetime(start_time)
    end_ts = start_ts + pd.Timedelta(minutes=duration_min)
    anomaly_indices = df[(df.index >= start_ts) & (df.index < end_ts)].index

    if not anomaly_indices.empty:
        if anomaly_type == 'spike':
            # ⚠️ SPIKE: Einfache Addition
            df.loc[anomaly_indices, 'traffic_volume_Tbits'] += mag_traffic

        elif anomaly_type == 'drift':
            # ⚠️ DRIFT: Lineare Zunahme (Gradual Shift)
            num_steps = len(anomaly_indices)
            drift_values = np.linspace(0, mag_traffic, num_steps)
            df.loc[anomaly_indices, 'traffic_volume_Tbits'] += drift_values

        df.loc[anomaly_indices, 'is_anomaly'] = True
        print(f"    -> {anomaly_type.capitalize()} hinzugefügt: {start_time}, Traffic Mag: {mag_traffic}")

    return df.copy()


# --- 4. MASTER TEMPLATES (1000 Flows, Mixed Anomalies) ---

master_templates = [
    # GRUPPE 1: PURE LINEAR FLOWS (Spike + Drift)
    {
        'group': 'Pure_Linear', 'pattern': 'linear',
        'template_asn': {'source': 1001, 'dest': 4001, 'handover': AS_IDS['AS200'], 'nexthop': AS_IDS['AS300']},
        'params': {
            'traffic_start': 40.0, 'traffic_slope': 0.005,
        },
        'anomalies': [
            # 1. Long Drift
            {'type': 'drift', 'start_time': '2025-01-07 04:00', 'duration_min': 360, 'magnitude_traffic': 10.0},
            # 2. Overlapping Spike (Testet, ob der Spike während des Drifts erkannt wird)
            {'type': 'spike', 'start_time': '2025-01-07 06:00', 'duration_min': 30, 'magnitude_traffic': 25.0},
        ],
        'count': NUM_FLOWS_PER_GROUP
    },

    # GRUPPE 2: SINUS FLOWS (Drift only)
    {
        'group': 'Pure_Sinus', 'pattern': 'sinus',
        'template_asn': {'source': 5001, 'dest': 5002, 'handover': AS_IDS['AS300'], 'nexthop': AS_IDS['AS100']},
        'params': {
            'traffic_offset': 20.0, 'traffic_amplitude': 10.0, 'traffic_daily_freq': 1, 'traffic_phase': 0,
        },
        'anomalies': [
            {'type': 'drift', 'start_time': '2025-01-07 12:00', 'duration_min': 180, 'magnitude_traffic': 8.0},
        ],
        'count': NUM_FLOWS_PER_GROUP
    },

    # GRUPPE 3: JUMPS FLOWS (Spike only)
    {
        'group': 'Jump_Patterns', 'pattern': 'jumps',
        'template_asn': {'source': 6001, 'dest': 6002, 'handover': AS_IDS['AS100'], 'nexthop': AS_IDS['AS400']},
        'params': {
            'traffic_start': 10.0, 'traffic_slope': 0.0001, 'jump_interval_min': 240, 'jump_amplitude': 5.0,
        },
        'anomalies': [
            {'type': 'spike', 'start_time': '2025-01-07 20:00', 'duration_min': 15, 'magnitude_traffic': 20.0},
        ],
        'count': NUM_FLOWS_PER_GROUP
    },
]


# --- 5. MAIN EXECUTION LOOP ---

def generate_full_master_dataset():
    """Generates the full deterministic dataset by iterating over master templates."""

    time_index = pd.date_range(start=START_DATE, end=END_DATE, freq=FREQUENCY, inclusive='left')
    all_flows_data = []
    group_mappings = []

    print(f"Starting generation of {sum(t['count'] for t in master_templates)} simplified master flows.")

    for template in master_templates:

        # We generate 'count' unique FlowKeys based on the template
        for i in range(template['count']):

            # --- Unique FlowKey Generation ---
            source_asn = template['template_asn']['source'] + i
            dest_asn = template['template_asn']['dest'] + i
            flow_id = f"{source_asn}-{dest_asn}_{template['pattern']}"

            # --- Flow Generation ---
            params_copy = template['params'].copy()
            params_copy['traffic_start'] = params_copy.get('traffic_start', 0.0) + (i * 0.01)

            flow_df = generate_univariate_data(time_index, {'pattern': template['pattern'], 'params': params_copy})
            flow_df['flow_key_id'] = flow_id
            flow_df['is_anomaly'] = False

            # --- Anomalies (Apply Spikes and Drifts) ---
            flow_df['timestamp'] = flow_df.index  # Add timestamp column for anomaly lookup
            for anomaly in template['anomalies']:
                anomaly_copy = anomaly.copy()
                anomaly_type = anomaly_copy.pop('type')
                flow_df = add_multivariate_anomaly(flow_df, anomaly_type=anomaly_type, **anomaly_copy)

            # --- Metadaten ---
            flow_df['sourceAS'] = source_asn
            flow_df['destinationAS'] = dest_asn
            flow_df['handoverAS'] = template['template_asn']['handover']
            flow_df['nexthopAS'] = template['template_asn']['nexthop']

            # --- Group Mapping ---
            group_mappings.append({
                'flow_key_id': flow_id,
                'flow_group': template['group']
            })

            all_flows_data.append(flow_df.reset_index(drop=True))

    # --- Final Save ---
    final_df = pd.concat(all_flows_data, ignore_index=True)
    final_df['timestamp'] = pd.to_datetime(final_df['timestamp'])

    final_df.to_csv(OUTPUT_FILE, index=False)
    pd.DataFrame(group_mappings).to_csv(FLOW_GROUP_FILE, index=False)

    print(f"\n=========================================================")
    print(f"MASTER DATASET GENERATED: {len(final_df)} total rows.")
    print(f"TOTAL ANOMALY POINTS: {final_df['is_anomaly'].sum()}")
    print(f"=========================================================")
    return final_df

def visualize_flows(df, flow_group_file):
    """
    Visualizes the Traffic Volume and Ground Truth for one representative flow
    from each defined flow group (Linear, Sinus, Jumps).
    """

    # ⚠️ Set Matplotlib backend to Agg for deployment safety, or TkAgg for desktop viewing
    try:
        matplotlib.use('TkAgg')
    except ImportError:
        matplotlib.use('Agg')
        print("Matplotlib switched to non-interactive 'Agg' backend.")

    try:
        group_df = pd.read_csv(flow_group_file)
    except FileNotFoundError:
        print(f"Error: Flow group file '{flow_group_file}' not found.")
        return

    # Merge data to get the flow group name
    df_merged = df[['flow_key_id', 'timestamp', 'traffic_volume_Tbits', 'is_anomaly']].merge(
        group_df, on='flow_key_id', how='left'
    ).drop_duplicates(subset=['flow_key_id', 'timestamp'])

    # Select one representative flow from each group
    groups_to_plot = ['Pure_Linear', 'Pure_Sinus', 'Jump_Patterns']

    # Stratified Sampling: Pick the first generated flow from each target group
    sampled_flow_ids = [
        df_merged[df_merged['flow_group'] == group]['flow_key_id'].iloc[0]
        for group in groups_to_plot if group in df_merged['flow_group'].unique()
    ]

    num_plots = len(sampled_flow_ids)
    rows = 1 if num_plots <= 3 else 2

    fig, axes = plt.subplots(rows, int(np.ceil(num_plots / rows)), figsize=(18, 5 * rows), squeeze=False)
    axes = axes.flatten()

    print(f"\n--- STARTE VISUALISIERUNG von {num_plots} repräsentativen Flows ---")

    for i, flow_id in enumerate(sampled_flow_ids):
        flow_data = df_merged[df_merged['flow_key_id'] == flow_id].copy()
        anomalies = flow_data[flow_data['is_anomaly'] == True]
        group = flow_data['flow_group'].iloc[0]

        ax = axes[i]

        # 1. Plot Traffic Volume
        ax.plot(flow_data['timestamp'], flow_data['traffic_volume_Tbits'], label='Traffic Volume (Tbits)', color='blue',
                linewidth=1.2)

        # 2. Plot Ground Truth (Spikes)
        if not anomalies.empty:
            ax.scatter(anomalies['timestamp'], anomalies['traffic_volume_Tbits'],
                       color='red', label='Anomaly (GT)', s=40, zorder=5)

        # 3. Markierung des Training/Test-Splits (80% Ende)
        split_index = int(len(flow_data) * 0.8)
        split_time = flow_data['timestamp'].iloc[split_index]
        ax.axvline(x=split_time, color='gray', linestyle='--', linewidth=1, label='80/20 Split')

        ax.set_title(f"Flow Pattern: {flow_id.split('_')[-1].upper()} | Group: {group}", fontsize=12)
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Traffic (Tbits)")
        ax.legend(fontsize=9)
        ax.tick_params(axis='x', rotation=45, labelsize=8)

    plt.suptitle("Visualization of anomalous traffic flow", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


# --- 3. Updated Main Execution Block ---

if __name__ == "__main__":
    final_df = generate_full_master_dataset()

    #visualize_flows(final_df, FLOW_GROUP_FILE)
