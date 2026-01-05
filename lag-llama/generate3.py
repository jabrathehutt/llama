import pandas as pd
import numpy as np
import os

# Configuration
START_DATE = '2025-01-01 00:00'
END_DATE = '2025-01-08 00:00'
FREQUENCY = '5min'
OUTPUT_FILE = 'network_metrics_data.csv'
NUM_REPETITIONS = 20 

def get_configs():
    # We use a variety of phases and high noise to make it a "Stealth" test
    base_configs = [
        {
            'sourceAS': 100,
            'destinationAS': 400,
            'pattern': 'linear',
            'params': {
                'traffic_start': 50.0, 'traffic_slope': 0.0001, 'traffic_noise': 2.5, 
                'flow_base': 20000, 'flow_noise': 500, 'flow_correlation': 0.8,
                'error_base': 0.001, 'error_noise': 0.0005
            },
            'anomalies': [
                {'type': 'drift', 'start_time': '2025-01-06 10:00', 'duration_min': 200,
                 'magnitude_traffic': 4.0}
            ]
        },
        {
            'sourceAS': 500,
            'destinationAS': 600,
            'pattern': 'sinus',
            'params': {
                'traffic_offset': 30.0, 'traffic_amplitude': 12.0, 'traffic_daily_freq': 1, 
                'traffic_phase': np.random.uniform(0, 24), 
                'traffic_noise': 3.0, 
                'traffic_slope': 0.002,
                'flow_base': 15000, 'flow_noise': 800, 'flow_correlation': 0.9,
                'error_base': 0.005, 'error_noise': 0.001
            },
            'anomalies': [
                {'type': 'spike', 'start_time': '2025-01-07 04:00', 'duration_min': 30,
                 'magnitude_traffic': 5.0}
            ]
        }
    ]
    return base_configs

def generate_univariate_pattern(time_index, pattern, params):
    t = np.arange(len(time_index))
    total_minutes = (time_index - time_index[0]).total_seconds() / 60

    if pattern == 'linear':
        traffic = params['traffic_start'] + t * params['traffic_slope']
    elif pattern == 'sinus':
        period = 24 * 60 / params['traffic_daily_freq']
        shift = params['traffic_phase'] * 60
        traffic = params['traffic_offset'] + params['traffic_amplitude'] * np.sin(
            2 * np.pi * (total_minutes - shift) / period)
    
    noise = np.random.normal(0, params['traffic_noise'], len(time_index))
    # Return as float32 immediately to maintain consistency
    return (traffic + noise).astype(np.float32)

def generate_multivariate_sdata(time_index, config):
    params = config['params']
    length = len(time_index)
    traffic_volume = generate_univariate_pattern(time_index, config['pattern'], params)
    
    flow_count = params['flow_base'] + (traffic_volume * 10) + np.random.normal(0, params['flow_noise'], length)
    error_rate = params['error_base'] + (traffic_volume * 0.0001) + np.random.normal(0, params['error_noise'], length)

    return pd.DataFrame({
        'traffic_volume_Tbits': np.maximum(0, traffic_volume).astype(np.float32),
        'flow_count_per_5min': np.maximum(1, flow_count).astype(np.int32),
        'error_rate_perc': np.clip(error_rate * 100, 0, 100).astype(np.float32)
    }, index=time_index)

def add_anomaly(df, anomaly):
    start_ts = pd.to_datetime(anomaly['start_time'])
    duration = pd.Timedelta(minutes=anomaly['duration_min'])
    mask = (df.index >= start_ts) & (df.index < start_ts + duration)
    
    if mask.any():
        if anomaly['type'] == 'spike':
            # Explicitly cast addition to float32
            df.loc[mask, 'traffic_volume_Tbits'] += np.float32(anomaly['magnitude_traffic'])
        else: # drift
            num_steps = mask.sum()
            # Generate drift and cast to float32 before addition
            drift_values = np.linspace(0, anomaly['magnitude_traffic'], num_steps).astype(np.float32)
            df.loc[mask, 'traffic_volume_Tbits'] += drift_values
            
        df.loc[mask, 'is_anomaly'] = True
    return df

def generate_stealth_dataset():
    time_index = pd.date_range(start=START_DATE, end=END_DATE, freq=FREQUENCY, inclusive='left')
    all_flows = []

    print(f"Generating Stealth Dataset with {len(get_configs()) * NUM_REPETITIONS} flows...")

    for i in range(NUM_REPETITIONS):
        configs = get_configs()
        for config in configs:
            flow_id = f"{config['sourceAS']+i}-{config['destinationAS']+i}_{config['pattern']}"
            df = generate_multivariate_sdata(time_index, config)
            df['flow_key_id'] = flow_id
            df['is_anomaly'] = False
            
            for anomaly in config['anomalies']:
                df = add_anomaly(df, anomaly)
            
            df['timestamp'] = df.index
            all_flows.append(df)

    final_df = pd.concat(all_flows, ignore_index=True)
    final_df = final_df.rename(columns={'is_anomaly': 'ground_truth'})
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully saved stealth data to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_stealth_dataset()
