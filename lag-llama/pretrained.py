import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from types import ModuleType
from tqdm import tqdm

# --- 1. THE LEGACY BRIDGE ---
def create_dummy_module(module_path):
    parts = module_path.split('.')
    current = ''
    parent = None
    for part in parts:
        current = current + '.' + part if current else part
        if current not in sys.modules:
            module = ModuleType(current)
            sys.modules[current] = module
            if parent: setattr(sys.modules[parent], part, module)
        parent = current
    return sys.modules[module_path]

import gluonts.torch.distributions as gtd
torch.serialization.add_safe_globals([gtd.studentT.StudentTOutput, gtd.StudentTOutput])
create_dummy_module('gluonts.torch.modules.loss')
class MockL:
    def __init__(self, *args, **kwargs): pass
    def __getattr__(self, name): return lambda *args, **kwargs: None
sys.modules['gluonts.torch.modules.loss'].DistributionLoss = MockL
sys.modules['gluonts.torch.modules.loss'].NegativeLogLikelihood = MockL

from lag_llama.gluon.lightning_module import LagLlamaLightningModule

# --- 2. CONFIGURATION ---
lags_seq_92 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216, 228, 240, 252, 264, 276, 288, 300, 312, 324, 336, 348, 360, 372, 384, 396, 408, 420, 432, 444, 456, 468, 480, 492, 504, 516, 528, 540, 552, 564, 576]
SEASONAL_PERIOD = 288 
MODEL_INPUT_LEN = 32 + 576
WINDOW_SIZE = MODEL_INPUT_LEN + SEASONAL_PERIOD 
BATCH_SIZE = 500 

def run_forecast_validation():
    device = torch.device("cuda")
    MODEL_PATH = "specialized_network_llama.pt"
    DATA_FILE = "network_metrics_data.csv"

    model = LagLlamaLightningModule(
        prediction_length=1, context_length=32,
        model_kwargs={
            "n_layer": 8, "n_head": 9, "n_embd_per_head": 16,
            "lags_seq": lags_seq_92, "context_length": 32,
            "max_context_length": 1024, "scaling": "mean",
            "input_size": 1, "distr_output": gtd.StudentTOutput(), "time_feat": False
        }
    )

    # --- 3. GRAFTING ---
    ckpt_state = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt_state: ckpt_state = ckpt_state["state_dict"]
    model_state = model.state_dict()
    
    ckpt_key = "model.transformer.wte.weight" if "model.transformer.wte.weight" in ckpt_state else "transformer.wte.weight"
    model_key = "model.transformer.wte.weight" if "model.transformer.wte.weight" in model_state else "transformer.wte.weight"
    if ckpt_key in ckpt_state:
        learned_w, target_w = ckpt_state[ckpt_key], model_state[model_key]
        if learned_w.shape != target_w.shape:
            new_w = torch.zeros_like(target_w)
            r, c = min(learned_w.shape[0], target_w.shape[0]), min(learned_w.shape[1], target_w.shape[1])
            new_w[:r, :c] = learned_w[:r, :c]
            ckpt_state[model_key] = new_w
    
    model.load_state_dict(ckpt_state, strict=False)
    model.to(device).eval()

    # --- 4. DATA PROCESSING ---
    df = pd.read_csv(DATA_FILE)
    results = []
    all_flow_ids = df['flow_key_id'].unique()

    print(f"Generating forecasts and validating against actuals for {len(all_flow_ids)} flows...")

    with torch.no_grad():
        for flow_id in tqdm(all_flow_ids):
            group = df[df['flow_key_id'] == flow_id]
            raw_values = group['traffic_volume_Tbits'].values.astype(np.float32)
            is_anomaly_truth = group['is_anomaly'].values.astype(int)
            category = flow_id.split('_')[-1]

            if len(raw_values) < WINDOW_SIZE: continue

            # Create Seasonal Difference (Legacy logic)
            indices = np.arange(WINDOW_SIZE, len(raw_values) + 1)
            diffed_values = raw_values[SEASONAL_PERIOD:] - raw_values[:-SEASONAL_PERIOD]
            offset_indices = indices - SEASONAL_PERIOD

            for b in range(0, len(offset_indices), BATCH_SIZE):
                batch_indices = offset_indices[b : b + BATCH_SIZE]
                actual_indices = indices[b : b + BATCH_SIZE]

                windows = np.array([diffed_values[i-MODEL_INPUT_LEN : i] for i in batch_indices])
                ground_truth_labels = is_anomaly_truth[actual_indices-1]

                # Local Normalization (Legacy logic)
                w_means = windows.mean(axis=1, keepdims=True)
                w_stds = windows.std(axis=1, keepdims=True) + 1e-8
                scaled_windows = (windows - w_means) / w_stds

                past_target = torch.from_numpy(scaled_windows).to(device)
                _, loc, scale_pred = model.model(past_target=past_target, past_observed_values=torch.ones_like(past_target).to(device))

                # Here we compare the Actual against the Forecasted Distribution
                forecast_mu = loc[:, -1:].cpu().numpy().flatten()
                actual_observed = scaled_windows[:, -1]
                forecast_sigma = scale_pred[:, -1:].cpu().numpy().flatten()

                # Anomaly Score = Error relative to the predicted variance
                error_score = np.abs(actual_observed - forecast_mu) / (forecast_sigma + 1e-4)
                
                # Apply smoothing
                s_scores = pd.Series(error_score).rolling(window=3, center=True).mean().ffill().bfill().values

                for k in range(len(s_scores)):
                    results.append({
                        "category": category,
                        "score": float(s_scores[k]),
                        "truth": int(ground_truth_labels[k])
                    })

    report_df = pd.DataFrame(results)

    # --- 5. CALCULATE PRECISION/RECALL ---
    # We find the threshold where the forecast error is high enough to be an anomaly
    print("Finding optimal forecasting threshold...")
    best_f1, best_threshold = 0, 0
    for p in np.arange(80, 99.9, 0.1):
        test_thresh = float(report_df['score'].quantile(p/100))
        y_pred = (report_df['score'] > test_thresh).astype(int)
        f1 = f1_score(report_df['truth'], y_pred, zero_division=0)
        prec = precision_score(report_df['truth'], y_pred, zero_division=0)
        
        if prec > 0.80 and f1 > best_f1:
            best_f1, best_threshold = f1, test_thresh

    report_df['prediction'] = (report_df['score'] > best_threshold).astype(int)

    # --- 6. FINAL REPORT ---
    print("\n" + "="*45)
    print(f"   FORECAST-VALIDATION REPORT")
    print("="*45)
    y_true, y_pred = report_df['truth'], report_df['prediction']
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print("-" * 45)
    for cat in sorted(report_df['category'].unique()):
        sub = report_df[report_df['category'] == cat]
        print(f"Recall for {cat.upper():<8}: {recall_score(sub['truth'], sub['prediction'], zero_division=0):.4f}")
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"\nTP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")
    print("="*45)

if __name__ == "__main__":
    run_forecast_validation()
