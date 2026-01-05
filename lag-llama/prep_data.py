import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path

def prepare_network_for_official_run():
    dataset_name = "network_telemetry"
    input_csv = "clean_training_traffic_data.csv"
    save_root = Path("datasets") / dataset_name
    save_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    dataset_list = []
    for flow_id, group in df.groupby("flow_key_id"):
        # Sort to ensure chronological order
        group = group.sort_values("timestamp")
        target_values = group['traffic_volume_Tbits'].values.astype(np.float32)
        
        # Create an array of 1s to indicate all values are observed (no missing data)
        observed_values = np.ones_like(target_values)
        
        dataset_list.append({
            "start": pd.Period(group["timestamp"].min(), freq="5min"),
            "target": target_values,
            "observed_values": observed_values, # Explicitly added here
            "item_id": str(flow_id)
        })

    save_path = save_root / "processed_data.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(dataset_list, f)

    print(f"✅ Success! Processed dataset with observed_values ready at {save_path}")

if __name__ == "__main__":
    prepare_network_for_official_run()
