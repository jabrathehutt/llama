import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_stochastic_traffic(csv_file):
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    col = 'traffic_volume_Gbits'
    
    flow_id = df[df['is_anomaly'] == True]['flow_key_id'].unique()[0]
    flow_data = df[df['flow_key_id'] == flow_id]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # PANEL 1: Time Series
    ax1.plot(flow_data['timestamp'], flow_data[col], color='#1f77b4', label='Traffic (Gbits)')
    ax1.axvspan(flow_data[flow_data['is_anomaly']]['timestamp'].min(), 
                flow_data[flow_data['is_anomaly']]['timestamp'].max(), 
                color='red', alpha=0.3, label='Anomaly Surge')
    ax1.set_title(f"Traffic Flow Surge: {flow_id}")
    ax1.set_ylabel("Volume (Gbits)")
    ax1.set_ylim(0, flow_data[col].max() * 1.1)
    ax1.legend()

    # PANEL 2: Distributions
    sns.kdeplot(flow_data[flow_data['is_anomaly']==False][col], ax=ax2, fill=True, label='Normal')
    sns.kdeplot(flow_data[flow_data['is_anomaly']==True][col], ax=ax2, fill=True, color='red', label='Anomaly')
    ax2.set_title("Statistical Distribution Shift")
    ax2.set_xlabel("Volume (Gbits)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig('trafpy_visualization.png')
    print("Plot updated with visible Gbit surge.")

if __name__ == "__main__":
    visualize_stochastic_traffic('trafpy_master_univariate_data.csv')
