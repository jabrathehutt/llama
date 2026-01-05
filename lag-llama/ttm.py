import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# ==================== PATH FIX BLOCK ====================
# This forces Python to find the tsfm_public folder if pip failed to link it
possible_paths = [
    os.getcwd(),
    os.path.join(os.getcwd(), "granite-tsfm"),
    "/root/lag-llama/granite-tsfm"
]
for p in possible_paths:
    if p not in sys.path:
        sys.path.append(p)

try:
    from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
except ImportError:
    try:
        from tsfm_public import TinyTimeMixerForPrediction
    except ImportError:
        print("❌ CRITICAL: Could not find 'tsfm_public'.")
        print("Please run: git clone https://github.com/ibm-granite/granite-tsfm.git")
        sys.exit(1)

warnings.filterwarnings("ignore")

# --- Configuration ---
DATA_FILE = "/root/lag-llama/network_metrics_data.csv"
MODEL_ID = "ibm-granite/granite-timeseries-ttm-v1" 
TARGET_COLUMNS = ['traffic_volume_Tbits', 'packet_count', 'byte_count'] 
CONTEXT_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_ttm():
    print(f"🚀 Initializing TTM Multivariate on {DEVICE}...")

    # 1. Load and Scale Data
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    df = pd.read_csv(DATA_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Ensure all target columns exist
    available_cols = [c for c in TARGET_COLUMNS if c in df.columns]
    print(f"Using columns: {available_cols}")

    scaler = StandardScaler()
    df[available_cols] = scaler.fit_transform(df[available_cols])

    # 2. Load Model
    print(f"Loading weights from Hugging Face: {MODEL_ID}...")
    model = TinyTimeMixerForPrediction.from_pretrained(
        MODEL_ID,
        num_input_channels=len(available_cols),
        prediction_filter_length=1,
        ignore_mismatched_sizes=True
    ).to(DEVICE).eval()

    # 3. Find Anomaly for Validation
    anom_indices = df.index[df['is_anomaly'] == 1].tolist()
    if not anom_indices:
        print("No anomalies found in CSV.")
        return
    
    idx = anom_indices[0]
    # TTM needs a long history (CONTEXT_LENGTH) to start predicting
    start_idx = max(0, idx - 600)
    end_idx = min(len(df), idx + 200)
    sample_df = df.iloc[start_idx : end_idx].copy()

    # 4. Inference
    print(f"Analyzing window (Size: {len(sample_df)})...")
    errors = []
    y_true = []

    for i in tqdm(range(CONTEXT_LENGTH, len(sample_df))):
        window = sample_df.iloc[i-CONTEXT_LENGTH : i][available_cols].values
        actual = sample_df.iloc[i][available_cols].values
        
        # TTM expects: [Batch, Time, Channels]
        input_tensor = torch.tensor(window).float().unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(past_values=input_tensor)
            prediction = output.prediction_outputs[0, -1, :].cpu().numpy()

        # Multivariate Anomaly Score (L2 Norm)
        dist = np.linalg.norm(actual - prediction)
        errors.append(dist)
        y_true.append(sample_df.iloc[i]['is_anomaly'])

    # 5. Metrics
    # We use a very high percentile because botnets are extreme outliers
    threshold = np.percentile(errors, 99.5) 
    y_pred = [1 if e > threshold else 0 for e in errors]

    print("\n" + "="*35)
    print("   TTM MULTIVARIATE RESULTS")
    print("="*35)
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")

if __name__ == "__main__":
    run_ttm()
