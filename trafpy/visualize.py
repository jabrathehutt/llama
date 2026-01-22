import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

def visualize_stochastic_traffic_single_panel(csv_file):
    # 1. Load and Prepare Data
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 2. Select the first flow containing anomalies
    anomalous_flows = df[df['is_anomaly'] == True]['flow_key_id'].unique()
    if len(anomalous_flows) == 0:
        print("Error: No anomalies found in the dataset.")
        return

    flow_id = anomalous_flows[0]
    flow_data = df[df['flow_key_id'] == flow_id].sort_values('timestamp')

    # 3. Create Single-Panel Plot
    fig, ax1 = plt.subplots(figsize=(15, 7))

    # Plot the time series
    ax1.plot(flow_data['timestamp'], flow_data['traffic_volume_Tbits'],
             label='Stochastic Traffic (TrafPy)', color='#1f77b4', alpha=0.8, linewidth=1.2)

    # Highlight the anomaly window
    anomalies = flow_data[flow_data['is_anomaly'] == True]
    if not anomalies.empty:
        ax1.axvspan(anomalies['timestamp'].min(), anomalies['timestamp'].max(),
                    color='red', alpha=0.15, label='Anomaly Window')

    # 4. Clean up Annotations and Formatting
    ax1.set_title(f"Anomalous Inter-domain Traffic Data", fontsize=14, pad=15)
    ax1.set_ylabel("Traffic Volume (Tbits)", fontsize=12)
    ax1.set_xlabel("Timestamp", fontsize=12)
    
    # Use AutoDateFormatter for decipherable timestamps
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=30, ha='right')
    
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig('trafpy_traffic.png', dpi=300)
    print("Single-panel plot saved as trafpy_traffic.png")
    plt.show()

if __name__ == "__main__":
    visualize_stochastic_traffic_single_panel('trafpy_master_univariate_data.csv')
