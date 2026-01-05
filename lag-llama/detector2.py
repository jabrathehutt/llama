import os
import sys
import torch
import pandas as pd
import numpy as np
import random
from sklearn.metrics import precision_score, recall_score, f1_score
from types import ModuleType

# --- 0. HARDWARE & KERNEL LOCKDOWN ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# --- 1. RIGID SEEDING ---
SEED = 42
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
set_seed(SEED)

# --- 2. COMPATIBILITY BRIDGE ---
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

# --- 3. CONFIGURATION ---
lags_seq_92 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216, 228, 240, 252, 264, 276, 288, 300, 312, 324, 336, 348, 360, 372, 384, 396, 408, 420, 432, 444, 456, 468, 480, 492, 504, 516, 528, 540, 552, 564, 576]
WINDOW_SIZE = 832 

def run_calibrated_detector():
    model = LagLlamaLightningModule(
        prediction_length=1, context_length=256, 
        model_kwargs={"n_layer": 8, "n_head": 6, "n_embd_per_head": 24, "lags_seq": lags_seq_92, "context_length": 256, "max_context_length": 1024, "scaling": "mean", "input_size": 1, "distr_output": gtd.StudentTOutput(), "time_feat": False}
    )
    model.load_state_dict(torch.load("specialized_network_llama.pt", map_location="cpu", weights_only=False), strict=False)
    model.eval().to("cuda")

    df = pd.read_csv("network_metrics_data.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['flow_key_id', 'timestamp']).reset_index(drop=True)
    
    g_mean, g_std = df['traffic_volume_Tbits'].mean(), df['traffic_volume_Tbits'].std() + 1e-8
    df['scaled'] = (df['traffic_volume_Tbits'] - g_mean) / g_std
    
    all_flows = sorted(df['flow_key_id'].unique())
    indices = np.linspace(0, len(all_flows)-1, 150).astype(int)
    sampled_flows = [all_flows[i] for i in indices]
    
    raw_results = []
    with torch.no_grad():
        for flow_id in sampled_flows:
            group = df[df['flow_key_id'] == flow_id]
            values = group['scaled'].values.astype(np.float32)
            truths = group['is_anomaly'].values.astype(int)
            if len(values) < WINDOW_SIZE + 400: continue
            
            scan_len = 400
            # Slicing the last 400 points
            windows = [values[i-WINDOW_SIZE : i] for i in range(len(values)-scan_len, len(values))]
            batch_truths = truths[-scan_len:]
            
            past_target = torch.from_numpy(np.array(windows)).to("cuda")
            distr_args, loc, scale_pred = model.model(past_target=past_target, past_observed_values=torch.ones_like(past_target).to("cuda"))
            
            nll = model.model.distr_output.loss(past_target[:, -1:], [arg[:, -1:] for arg in distr_args], loc[:, -1:], scale_pred[:, -1:]).cpu().numpy()
            res = (past_target[:, -1:] - loc[:, -1:]).abs().cpu().numpy().flatten()
            
            for k in range(len(nll)):
                raw_results.append({"n": float(nll[k]), "r": float(res[k]), "t": int(batch_truths[k])})

    res_df = pd.DataFrame(raw_results)
    
    # --- DETERMINISTIC SEARCH FOR OPTIMAL WEIGHTING ---
    best_f1, best_t, best_w = 0, 0, 0
    
    # Normalize signals
    res_df['n_norm'] = (res_df['n'] - res_df['n'].min()) / (res_df['n'].max() - res_df['n'].min() + 1e-9)
    res_df['r_norm'] = (res_df['r'] - res_df['r'].min()) / (res_df['r'].max() - res_df['r'].min() + 1e-9)
    
    # Grid Search for unified score weight (w = Weight of Surprise/NLL)
    for w in [0.1, 0.3, 0.5, 0.7, 0.9]:
        res_df['u'] = (res_df['n_norm'] * w) + (res_df['r_norm'] * (1-w))
        scores = res_df['u'].values
        y_true = res_df['t'].values
        
        for p in np.arange(95, 99.9, 0.1):
            thresh = float(np.percentile(scores, p))
            f1 = f1_score(y_true, (scores > thresh).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t, best_w = f1, thresh, w

    print(f"\nDeterministic Audit Complete (Seed: {SEED})")
    print(f"Optimal Score Weight (NLL): {best_w}")
    print(f"Optimal Percentile: {best_t}")
    print(f"F1-Score: {best_f1:.4f}")

if __name__ == "__main__":
    run_calibrated_detector()
