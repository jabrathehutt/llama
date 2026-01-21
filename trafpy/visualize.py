import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_stochastic_traffic(csv_file):
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Target column
    col = 'traffic_volume_Tbits'
    
    # Pick first anomalous flow
    flow_id = df[df['is_anomaly'] == True]['flow_key_id'].unique()[0]
    flow_data = df[df['flow_key_id'] == flow_id]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # PANEL 1: Time Series
    ax1.plot(flow_data['timestamp'], flow_data[col], color='#1f77b4', label='Traffic (Tbits)', alpha=0.8)
    ax1.axvspan(flow_data[flow_data['is_anomaly']]['timestamp'].min(), 
                flow_data[flow_data['is_anomaly']]['timestamp'].max(), 
                color='red', alpha=0.3, label='Anomaly Window')
    ax1.set_title(f"Univariate Tbit Traffic: {flow_id}", fontsize=14)
    ax1.set_ylabel("Traffic Volume (Tbits)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # PANEL 2: Distributions
    normal_data = flow_data[flow_data['is_anomaly'] == False][col]
    anomaly_data = flow_data[flow_data['is_anomaly'] == True][col]
    
    sns.kdeplot(normal_data, ax=ax2, fill=True, label='Normal (Lognormal)')
    if anomaly_data.var() > 1e-9:
        sns.kdeplot(anomaly_data, ax=ax2, fill=True, color='red', label='Anomaly (High-Variance Surge)')
    else:
        ax2.hist(anomaly_data, bins=10, alpha=0.5, color='red', label='Anomaly (Surge)', density=True)
        
    ax2.set_title("Statistical Signature Shift", fontsize=14)
    ax2.set_xlabel("Traffic Volume (Tbits)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('trafpy_visualization.png')
    print("Tbit visualization saved as trafpy_visualization.png")

if __name__ == "__main__":
    visualize_stochastic_traffic('trafpy_master_univariate_data.csv')
