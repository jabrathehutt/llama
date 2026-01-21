import pandas as pd
import numpy as np
import torch
from gluonts.dataset.pandas import PandasDataset
from lag_llama.gluon.estimator import LagLlamaEstimator

# --- CONFIGURATION ---
INPUT_FILE = "/root/network_pretrain.csv"
CKPT_PATH = "lag-llama-backbone.ckpt"

def run_extensive_pretraining():
    print(f"Loading {INPUT_FILE} for backbone pretraining...")
    df = pd.read_csv(INPUT_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # ⚠️ CRITICAL FIX: Cast traffic_volume to float32
    # This resolves the 'Double and Float' dtype mismatch error.
    df['traffic_volume'] = df['traffic_volume'].astype('float32')
    
    # Lag-Llama expects a 'target' column. 
    dataset = PandasDataset(
        df, 
        target="traffic_volume", 
        timestamp="timestamp", 
        freq="5min"
    )

    print("Initializing Lag-Llama Estimator...")
    # Parameters optimized for backbone traffic cycles
    estimator = LagLlamaEstimator(
        prediction_length=24,   # 2-hour forecast horizon
        context_length=128,     # ~10-hour historical context
        batch_size=64,
        num_parallel_samples=100,
        trainer_kwargs={
            "max_epochs": 50, 
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": 1 # Use 1 device to avoid DDP overhead for this size
        }
    )

    print("Starting Stage 1: Extensive Unsupervised Pretraining...")
    predictor = estimator.train(dataset)
    
    # Save the trained backbone
    torch.save(predictor.network.state_dict(), CKPT_PATH)
    print(f"Successfully saved pretrained backbone to {CKPT_PATH}")

if __name__ == "__main__":
    run_extensive_pretraining()
