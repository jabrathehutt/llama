import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_stochastic_traffic(csv_file):
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Select the first flow that actually contains anomalies
    anomalous_flows = df[df['is_anomaly'] == True]['flow_key_id'].unique()
    if len(anomalous_flows) == 0:
        print("Error: No anomalies found in the dataset. Check your generation script.")
        return
        
    flow_id = anomalous_flows[0]
    flow_data = df[df['flow_key_id'] == flow_id]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [1, 1]})
    
    # --- PANEL 1: Time Series ---
    ax1.plot(flow_data['timestamp'], flow_data['traffic_volume_Tbits'], 
             label='Stochastic Traffic (TrafPy)', color='#1f77b4', alpha=0.7)
    
    anomalies = flow_data[flow_data['is_anomaly'] == True]
    ax1.axvspan(anomalies['timestamp'].min(), anomalies['timestamp'].max(), 
                color='red', alpha=0.15, label='Anomaly Window')

    ax1.set_title(f"Univariate Traffic Flow: {flow_id}", fontsize=14)
    ax1.set_ylabel("Traffic Volume (Tbits)")
    ax1.legend()

    # --- PANEL 2: Distribution Comparison ---
    normal_vol = flow_data[flow_data['is_anomaly'] == False]['traffic_volume_Tbits']
    anomaly_vol = flow_data[flow_data['is_anomaly'] == True]['traffic_volume_Tbits']
    
    # Plot Normal Distribution
    sns.kdeplot(normal_vol, ax=ax2, fill=True, color='blue', label='Normal (Lognormal)')
    
    # Plot Anomaly with Variance Check
    if anomaly_vol.var() > 0:
        sns.kdeplot(anomaly_vol, ax=ax2, fill=True, color='red', label='Anomaly (Exponential)')
    else:
        # Fallback to histogram if variance is zero (all values identical)
        ax2.hist(anomaly_vol, bins=5, alpha=0.5, color='red', label='Anomaly (Fixed Magnitude)', density=True)
    
    ax2.set_title("Statistical Signature: Normal vs. Anomaly", fontsize=14)
    ax2.set_xlabel("Traffic Volume (Tbits)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig('trafpy_visualization.png')
    print("Plot saved as trafpy_visualization.png")
    plt.show()

if __name__ == "__main__":
    visualize_stochastic_traffic('trafpy_master_univariate_data.csv')
