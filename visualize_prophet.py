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
    m = Prophet(interval_width=CONFIDENCE_INTERVAL, daily_seasonality=True)
    m.fit(df_train)

    # 5. Forecast on the Test Timeline
    future = pd.DataFrame({'ds': df_test['ds']})
    forecast = m.predict(future)

    # 6. Apply High-Recall Detection Logic
    df_pred = df_test.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')
    df_pred['uncertainty'] = df_pred['yhat_upper'] - df_pred['yhat_lower']

    df_pred['upper_threshold'] = df_pred['yhat'] + (THRESHOLD_FACTOR * df_pred['uncertainty'])
    df_pred['lower_threshold'] = df_pred['yhat'] - (THRESHOLD_FACTOR * df_pred['uncertainty'])

    # 7. Select a window around the anomaly
    anomaly_hits = df_pred[df_pred['is_anomaly'] == 1]
    if not anomaly_hits.empty:
        center_idx = anomaly_hits.index[0]
        plot_df = df_pred.iloc[max(0, center_idx-50) : min(len(df_pred), center_idx+150)].copy()
    else:
        plot_df = df_pred.iloc[:200].copy()

    # 8. Visualization (Single Panel Only)
    # Note: subplot(1,1,1) or just subplots() works here
    fig, ax = plt.subplots(figsize=(15, 7)) 

    # Plotting Boundaries and Traffic
    ax.plot(plot_df['ds'], plot_df['y'], label='Actual Traffic', color='#1f77b4', zorder=3)
    ax.plot(plot_df['ds'], plot_df['yhat'], label='Prophet Predicted Baseline', color='#ff7f0e', linestyle='--')
    ax.fill_between(plot_df['ds'], plot_df['lower_threshold'], plot_df['upper_threshold'],
                     color='#ff7f0e', alpha=0.15, label=f'Detection Zone')

    # Labels and Titles
    ax.set_title(f"Prophet Traffic Detection", fontsize=14)
    ax.set_ylabel("Traffic Volume (Tbits)")
    ax.set_xlabel("Timestamp")
    ax.legend(loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.6)

    # --- TIMESTAMP FORMATTING ---
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=30, ha='right')

    plt.tight_layout()
    plt.savefig("prophet.png", dpi=300)
    print("Success: Visualization saved to prophet.png")
    plt.show()

if __name__ == "__main__":
    visualize_prophet_force_aligned()
