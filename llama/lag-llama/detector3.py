import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import make_evaluation_predictions
from lag_llama.gluon.estimator import LagLlamaEstimator
from lag_llama.gluon.lightning_module import LagLlamaLightningModule
from gluonts.torch.distributions import StudentTOutput
from sklearn.metrics import f1_score, precision_score, recall_score

# --- OFFICIAL ARCHITECTURE CONSTANTS ---
# Lags used for tokenization across frequencies [cite: 98, 102]
LAG_INDICES = [1, 2, 3, 4, 5, 6, 7, 12, 24, 48, 72, 168]
CONTEXT_LENGTH = 32 # Standard context window size [cite: 522]
NUM_SAMPLES = 100    # Samples for probabilistic uncertainty [cite: 209]

def run_full_history_inference():
    # 1. LOAD DATA
    df = pd.read_csv("/root/traffic-shifts/trafpy/trafpy_master_univariate_data.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['traffic_volume_Tbits'] = df['traffic_volume_Tbits'].astype(np.float32)
    
    # 2. INITIALIZE OFFICIAL MODULE [cite: 108, 112]
    module = LagLlamaLightningModule(
        context_length=CONTEXT_LENGTH,
        prediction_length=1,
        model_kwargs={
            "input_size": 1,
            "context_length": CONTEXT_LENGTH,
            "max_context_length": 2048,
            "distr_output": StudentTOutput(), # Student's t-distribution 
            "n_layer": 8,
            "n_embd_per_head": 32,
            "n_head": 8,
            "scaling": "mean", # Standard magnitude scaling 
            "time_feat": False,
            "lags_seq": LAG_INDICES,
        }
    )

    # 3. LOAD SPECIALIZED WEIGHTS
    checkpoint = torch.load("specialized_v11_supervised.pt", map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    module.load_state_dict(state_dict, strict=False)
    module.eval()

    # 4. PREPARE ROLLING WINDOWS (Predicting every point in history)
    unique_ids = df['flow_key_id'].unique()[:5]
    rolling_windows = []
    
    print("Preparing full history windows...")
    for flow_id in unique_ids:
        flow_df = df[df['flow_key_id'] == flow_id].sort_values('timestamp')
        target_vals = flow_df['traffic_volume_Tbits'].values
        
        # We need at least 32 points of context before we can predict [cite: 106]
        for i in range(CONTEXT_LENGTH, len(flow_df)):
            rolling_windows.append({
                "start": flow_df['timestamp'].iloc[0],
                "target": target_vals[:i],
                "item_id": f"{flow_id}|{i}" # Store index to map back to truth
            })

    dataset = ListDataset(rolling_windows, freq="10min")

    # 5. EXECUTE OFFICIAL PREDICTOR
    estimator = LagLlamaEstimator(prediction_length=1, context_length=CONTEXT_LENGTH, batch_size=64)
    predictor = estimator.create_predictor(
        transformation=estimator.create_transformation(),
        module=module
    )

    print(f"Analyzing {len(rolling_windows)} total historical points...")
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset,
        predictor=predictor,
        num_samples=NUM_SAMPLES
    )

    forecasts = list(tqdm(forecast_it, total=len(dataset), desc="Forecasting History"))
    
    # 6. CALCULATE PERFORMANCE
    all_results = []
    for forecast in forecasts:
        flow_id, idx_str = forecast.item_id.split('|')
        idx = int(idx_str)
        
        # Threshold: Value exceeds the 95th percentile [cite: 125, 543]
        q95 = np.quantile(forecast.samples, 0.95, axis=0)[0]
        
        actual_row = df[(df['flow_key_id'] == flow_id)].iloc[idx]
        actual_val = actual_row['traffic_volume_Tbits']
        
        all_results.append({
            "ground_truth": int(actual_row['is_anomaly']),
            "prediction": 1 if actual_val > q95 else 0
        })

    res_df = pd.DataFrame(all_results)
    
    # 7. FINAL REPORT
    f1 = f1_score(res_df['ground_truth'], res_df['prediction'], zero_division=0)
    print("\n" + "="*40)
    print("FULL HISTORY EVALUATION")
    print("-" * 40)
    print(f"F1-Score:  {f1:.4f}")
    print(f"Precision: {precision_score(res_df['ground_truth'], res_df['prediction'], zero_division=0):.4f}")
    print(f"Recall:    {recall_score(res_df['ground_truth'], res_df['prediction'], zero_division=0):.4f}")
    print(f"Total Points: {len(res_df)}")
    print("="*40)

if __name__ == "__main__":
    run_full_history_inference()
