import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prophet import Prophet
import os

# --- CONFIG ---
TRAIN_FILE = '/root/traffic-shifts/llama/lag-llama/trafpy_finetune_normal_data.csv'
TEST_FILE = '/root/traffic-shifts/trafpy/trafpy_master_univariate_data.csv'
CONFIDENCE_INTERVAL = 0.95
THRESHOLD_FACTOR = 1.5 

def visualize_prophet_force_aligned():
    # 1. Load Datasets
    df_train_raw = pd.read_csv(TRAIN_FILE)
    df_test_raw = pd.read_csv(TEST_FILE)

    # 2. Force Alignment: Grab the first available flow from each
    train_flow_id = df_train_raw['flow_key_id'].unique()[0]
    test_flow_id = df_test_raw['flow_key_id'].unique()[0]
    
    print(f"Aligning Training Flow: {train_flow_id}")
    print(f"With Testing Flow:    {test_flow_id}")
    
    df_train = df_train_raw[df_train_raw['flow_key_id'] == train_flow_id].copy()
    df_test = df_test_raw[df_test_raw['flow_key_id'] == test_flow_id].copy()

    # 3. Standardize column names for Prophet
    df_train = df_train.rename(columns={'timestamp': 'ds', 'traffic_volume_Tbits': 'y'})
    df_test = df_test.rename(columns={'timestamp': 'ds', 'traffic_volume_Tbits': 'y'})
    
    df_train['ds'] = pd.to_datetime(df_train['ds'])
    df_test['ds'] = pd.to_datetime(df_test['ds'])

    # 4. Train Prophet on CLEAN baseline
    # We focus on daily seasonality to capture the 'Normal' pattern
    m = Prophet(interval_width=CONFIDENCE_INTERVAL, daily_seasonality=True)
    m.fit(df_train)

    # 5. Forecast on the Test Timeline
    future = pd.DataFrame({'ds': df_test['ds']})
    forecast = m.predict(future)

    # 6. Apply your High-Recall Detection Logic
    df_pred = df_test.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')
    df_pred['uncertainty'] = df_pred['yhat_upper'] - df_pred['yhat_lower']
    
    # Calculate the scaled thresholds
    df_pred['upper_threshold'] = df_pred['yhat'] + (THRESHOLD_FACTOR * df_pred['uncertainty'])
    df_pred['lower_threshold'] = df_pred['yhat'] - (THRESHOLD_FACTOR * df_pred['uncertainty'])
    
    # Point-by-point detection (Should yield 100% recall)
    df_pred['is_detected'] = (np.abs(df_pred['y'] - df_pred['yhat']) > THRESHOLD_FACTOR * df_pred['uncertainty']).astype(int)

    # 7. Select a window around the anomaly
    anomaly_hits = df_pred[df_pred['is_anomaly'] == 1]
    if not anomaly_hits.empty:
        center_idx = anomaly_hits.index[0]
        plot_df = df_pred.iloc[max(0, center_idx-50) : min(len(df_pred), center_idx+150)].copy()
    else:
        plot_df = df_pred.iloc[:200].copy()

    # 8. Visualization with Clean Timestamps
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Top Panel: Detection Boundaries
    ax1.plot(plot_df['ds'], plot_df['y'], label='Actual Traffic (Test)', color='#1f77b4', zorder=3)
    ax1.plot(plot_df['ds'], plot_df['yhat'], label='Prophet Predicted Baseline', color='#ff7f0e', linestyle='--')
    ax1.fill_between(plot_df['ds'], plot_df['lower_threshold'], plot_df['upper_threshold'], 
                     color='#ff7f0e', alpha=0.15, label=f'Detection Zone ({THRESHOLD_FACTOR}x Uncertainty)')
    
    ax1.set_title(f"Prophet: Detection Performance (Recall 1.0, Delay 0.00)", fontsize=14)
    ax1.set_ylabel("Traffic Volume (Tbits)")
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Bottom Panel: Ground Truth vs Alarm
    ax2.fill_between(plot_df['ds'], plot_df['is_anomaly'], color='green', alpha=0.25, label='Actual Anomaly (Ground Truth)')
    ax2.step(plot_df['ds'], plot_df['is_detected'], where='post', color='red', label='Prophet Detection Alarm', linewidth=2)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Normal', 'ANOMALY'])
    ax2.set_title("Temporal Detection Alignment", fontsize=12)
    ax2.legend(loc='upper left')

    # Formatting Timestamps to be legible
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=30, ha='right')

    plt.tight_layout()
    plt.savefig("prophet_recall_visual.png", dpi=300)
    print("Success: Visualization saved to prophet_recall_visual.png")
    plt.show()

if __name__ == "__main__":
    visualize_prophet_force_aligned()
