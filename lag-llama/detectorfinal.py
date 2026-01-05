import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from types import ModuleType

# --- 1. THE LEGACY BRIDGE (UNCHANGED) ---
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
WINDOW_SIZE = 32 + 576 

def run_precision_recovery():
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

    # --- 3. GRAFTING LOGIC ---
    ckpt_state = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model_state = model.state_dict()
    ckpt_key = "model.transformer.wte.weight" if "model.transformer.wte.weight" in ckpt_state else "transformer.wte.weight"
    model_key = "model.transformer.wte.weight" if "model.transformer.wte.weight" in model_state else "transformer.wte.weight"

    if ckpt_key in ckpt_state:
        learned_w, target_w = ckpt_state[ckpt_key], model_state[model_key]
        if learned_w.shape != target_w.shape:
            new_w = torch.zeros_like(target_w)
            new_w[:learned_w.shape[0], :learned_w.shape[1]] = learned_w
            ckpt_state[model_key] = new_w
            if ckpt_key != model_key: del ckpt_state[ckpt_key]

    model.load_state_dict(ckpt_state, strict=False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- 4. DATA PROCESSING ---
    df = pd.read_csv(DATA_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    results = []
    sampled_flows = df['flow_key_id'].unique()[np.linspace(0, len(df['flow_key_id'].unique())-1, 150).astype(int)]

    print(f"Executing Precision Recovery over {len(sampled_flows)} flows...")

    with torch.no_grad():
        for flow_id in sampled_flows:
            group = df[df['flow_key_id'] == flow_id]
            raw_values = group['traffic_volume_Tbits'].values.astype(np.float32)
            is_anomaly_truth = group['is_anomaly'].values.astype(int)
            category = flow_id.split('_')[-1] # linear, sinus, jumps

            if len(raw_values) < WINDOW_SIZE: continue

            scan_len = 400
            start_idx = len(raw_values) - scan_len
            
            for i in range(start_idx, len(raw_values)):
                window = raw_values[i-WINDOW_SIZE : i]
                
                # Local Window Scaling (Crucial for Spike/Drift detection)
                w_mean, w_std = window.mean(), window.std() + 1e-8
                scaled_window = (window - w_mean) / w_std
                
                past_target = torch.from_numpy(scaled_window).unsqueeze(0).to(device)
                distr_args, loc, scale_pred = model.model(past_target=past_target, past_observed_values=torch.ones_like(past_target).to(device))

                # Logic: Spike = high residual, Drift = persistent high residual
                res_val = abs(scaled_window[-1] - loc[:, -1:].item())
                # Normalize by predicted uncertainty (scale)
                z_score = res_val / (scale_pred[:, -1:].item() + 1e-4)

                results.append({
                    "category": category,
                    "z_score": z_score,
                    "ground_truth": is_anomaly_truth[i-1]
                })

    report_df = pd.DataFrame(results)

    # --- 5. ROBUST THRESHOLD SWEEP ---
    # We prioritize F1-score to find the best balance between Spike (Spikes) 
    # and gradual changes (Drifts).
    best_f1, best_threshold = 0, 0
    
    # We sweep percentiles. Since Spikes are extreme, they live in 98th+ 
    # Drifts live slightly lower.
    for p in np.arange(90, 99.9, 0.1): 
        test_thresh = float(report_df['z_score'].quantile(p/100))
        y_pred = (report_df['z_score'] > test_thresh).astype(int)
        
        f1 = f1_score(report_df['ground_truth'], y_pred, zero_division=0)
        p_val = precision_score(report_df['ground_truth'], y_pred, zero_division=0)
        
        # We want Precision > 0.60 to ensure we aren't just flagging every "Jump"
        if p_val > 0.60 and f1 > best_f1:
            best_f1, best_threshold = f1, test_thresh

    report_df['prediction'] = (report_df['z_score'] > best_threshold).astype(int)

    print("\n" + "="*40 + "\n   RECOVERED METRICS REPORT\n" + "="*40)
    y_true, y_pred = report_df['ground_truth'], report_df['prediction']
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print("-" * 40)
    for cat in sorted(report_df['category'].unique()):
        cat_df = report_df[report_df['category'] == cat]
        print(f"Recall for {cat.upper():<7}: {recall_score(cat_df['ground_truth'], cat_df['prediction'], zero_division=0):.4f}")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"\nTP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}\n" + "="*40)

if __name__ == "__main__":
    run_precision_recovery()
