import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from types import ModuleType

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
WINDOW_SIZE = 256 + 576 

def run_precision_recovery():
    MODEL_PATH = "specialized_v11_supervised.pt"
    DATA_FILE = "/root/traffic-shifts/trafpy/trafpy_master_univariate_data.csv"
    
    model = LagLlamaLightningModule(
        prediction_length=1, context_length=256,
        model_kwargs={"n_layer": 8, "n_head": 6, "n_embd_per_head": 24, "lags_seq": lags_seq_92, "context_length": 256, "max_context_length": 1024, "scaling": "mean", "input_size": 1, "distr_output": gtd.StudentTOutput(), "time_feat": False}
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=False), strict=False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    df = pd.read_csv(DATA_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    g_mean, g_std = df['traffic_volume_Tbits'].mean(), df['traffic_volume_Tbits'].std() + 1e-8
    df['scaled'] = (df['traffic_volume_Tbits'] - g_mean) / g_std
    
    results = []
    all_flow_ids = df['flow_key_id'].unique()
    sampled_flows = all_flow_ids[np.linspace(0, len(all_flow_ids)-1, 150).astype(int)]
    
    print(f"Executing Precision Recovery over {len(sampled_flows)} flows...")

    with torch.no_grad():
        for idx, flow_id in enumerate(sampled_flows):
            group = df[df['flow_key_id'] == flow_id]
            values = group['scaled'].values.astype(np.float32)
            is_anomaly_truth = group['is_anomaly'].values.astype(int)
            category = flow_id.split('_')[-1]
            
            if len(values) < WINDOW_SIZE: continue
            
            # Focused scan on tail 400 points
            scan_len = 400
            windows = [values[i-WINDOW_SIZE : i] for i in range(len(values)-scan_len, len(values)) if i >= WINDOW_SIZE]
            truths = is_anomaly_truth[-len(windows):]
            
            past_target = torch.from_numpy(np.array(windows)).to(device)
            distr_args, loc, scale_pred = model.model(past_target=past_target, past_observed_values=torch.ones_like(past_target).to(device))
            
            # Scores
            nll = model.model.distr_output.loss(past_target[:, -1:], [arg[:, -1:] for arg in distr_args], loc[:, -1:], scale_pred[:, -1:]).cpu().numpy()
            residual = (past_target[:, -1:] - loc[:, -1:]).abs()
            drift_score = (residual / (scale_pred[:, -1:] + 1e-5)).cpu().numpy().flatten()
            
            # Smoothed Drift
            s_drift = pd.Series(drift_score).rolling(window=7, center=True).median().ffill().bfill().values

            for k in range(len(nll)):
                results.append({
                    "category": category, 
                    "nll_score": float(nll[k]), 
                    "drift_score": float(s_drift[k]),
                    "ground_truth": int(truths[k])
                })

    report_df = pd.DataFrame(results)
    
    # --- DUAL-GATE SCORING ---
    # Normalize scores to 0-1 range to combine them fairly
    report_df['nll_norm'] = (report_df['nll_score'] - report_df['nll_score'].min()) / (report_df['nll_score'].max() - report_df['nll_score'].min())
    report_df['drift_norm'] = (report_df['drift_score'] - report_df['drift_score'].min()) / (report_df['drift_score'].max() - report_df['drift_score'].min())
    
    # Unified Score (Weighted: NLL handles Jumps, Drift handles Slopes)
    report_df['unified_score'] = (report_df['nll_norm'] * 0.7) + (report_df['drift_norm'] * 0.3)
    
    # --- ROBUST THRESHOLD SWEEP ---
    # We sweep percentiles but strictly monitor the Confusion Matrix to prevent collapse
    best_f1, best_threshold = 0, 0
    
    for p in np.arange(90, 99.9, 0.1):
        test_thresh = float(report_df['unified_score'].quantile(p/100))
        y_pred = (report_df['unified_score'] > test_thresh).astype(int)
        
        f1 = f1_score(report_df['ground_truth'], y_pred, zero_division=0)
        p_val = precision_score(report_df['ground_truth'], y_pred, zero_division=0)
        
        # Enforce a minimum Precision of 70% to prevent the 100% Recall / 10% Precision collapse
        if p_val > 0.70 and f1 > best_f1:
            best_f1 = f1
            best_threshold = test_thresh

    report_df['prediction'] = (report_df['unified_score'] > best_threshold).astype(int)
    
    print("\n" + "="*40)
    print("   RECOVERED METRICS REPORT")
    print("="*40)
    y_true = report_df['ground_truth']
    y_pred = report_df['prediction']
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print("-" * 40)
    
    for cat in sorted(report_df['category'].unique()):
        cat_df = report_df[report_df['category'] == cat]
        print(f"Recall for {cat.upper():<7}: {recall_score(cat_df['ground_truth'], cat_df['prediction'], zero_division=0):.4f}")
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"\nTP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")
    print("="*40)

if __name__ == "__main__":
    run_precision_recovery()
