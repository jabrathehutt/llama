import pandas as pd
import numpy as np

def generate_focused_datasets(input_csv):
    df = pd.read_csv(input_csv)
    traffic_col = 'TotBytes' if 'TotBytes' in df.columns else 'SrcBytes'
    
    # Keyword-based mapping we established
    def label_logic(x):
        label_str = str(x).lower()
        if 'botnet' in label_str: return 1
        if 'background' in label_str or 'normal' in label_str: return 0
        return 1

    df['is_anomaly'] = df['Label'].apply(label_logic)
    df = df.rename(columns={traffic_col: 'traffic_volume_Tbits'})
    
    # Simulation for Lag-Llama
    n = len(df)
    df['timestamp'] = pd.date_range(start="2025-01-01", periods=n, freq="5min")
    df['flow_key_id'] = "kaggle_net_01"

    # 1. Take a focused subset of Normal data for Pretraining (e.g., 100k rows)
    # Training on 1 million rows of noise makes the model too "relaxed"
    normal_df = df[df['is_anomaly'] == 0]
    focused_clean = normal_df.sample(n=min(100000, len(normal_df)), random_state=42).sort_index()
    
    focused_clean.to_csv("clean_training_traffic_data.csv", index=False)
    df.to_csv("network_metrics_data.csv", index=False)
    
    print(f"Focused Pretraining Set: {len(focused_clean)} rows.")
    print(f"Full Detection Set: {len(df)} rows.")

if __name__ == "__main__":
    generate_focused_datasets('hello_1.csv')
