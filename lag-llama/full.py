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
MODEL_INPUT_LEN = 32 + 576

def run_comparative_audit(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_FILE = "network_metrics_data.csv"

    # CRITICAL: We set scaling to "none" in the config to force the model to use the weights
    model = LagLlamaLightningModule(
        prediction_length=1, context_length=32,
        model_kwargs={
            "n_layer": 8, "n_head": 9, "n_embd_per_head": 16,
            "lags_seq": lags_seq_92, "context_length": 32,
            "max_context_length": 1024, "scaling": "none", # DISABLE AUTO-SCALING
            "input_size": 1, "distr_output": gtd.StudentTOutput(), "time_feat": False
        }
    )

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    m_state = model.state_dict()
    m_k = "model.transformer.wte.weight" if "model.transformer.wte.weight" in m_state else "transformer.wte.weight"
    c_k = "model.transformer.wte.weight" if "model.transformer.wte.weight" in state else "transformer.wte.weight"
    
    if c_k in state:
        l_w, t_w = state[c_k], m_state[m_k]
        new_w = torch.zeros_like(t_w)
        r, c = min(l_w.shape[0], t_w.shape[0]), min(l_w.shape[1], t_w.shape[1])
        new_w[:r, :c] = l_w[:r, :c]
        state[m_k] = new_w
    
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    df = pd.read_csv(DATA_FILE)
    all_scores = []
    
    print(f"\nScanning with Raw Internal Features: {model_path}")
    with torch.no_grad():
        for flow_id in tqdm(df['flow_key_id'].unique()[:20]): # Test first 20 for speed
            group = df[df['flow_key_id'] == flow_id]
            vals = group['traffic_volume_Tbits'].values.astype(np.float32)
            truth_col = 'ground_truth' if 'ground_truth' in group.columns else 'is_anomaly'
            truths = group[truth_col].values.astype(int)

            if len(vals) < MODEL_INPUT_LEN + 1: continue

            for i in range(MODEL_INPUT_LEN, len(vals)):
                window = vals[i-MODEL_INPUT_LEN : i]
                # Manual scaling to a fixed global range so the model weights must do the work
                s_win = torch.from_numpy(window / 100.0).unsqueeze(0).to(device)
                
                _, loc, scale = model.model(past_target=s_win, past_observed_values=torch.ones_like(s_win))
                
                f_mu = loc[0, -1].item() * 100.0
                f_std = scale[0, -1].item() * 100.0
                
                z = np.abs(vals[i] - f_mu) / (f_std + 1e-4)
                all_scores.append({"z": z, "y": truths[i]})

    report = pd.DataFrame(all_scores)
    
    # Calculate metrics at a fixed high-sensitivity threshold
    threshold = report['z'].quantile(0.95)
    y_pred = (report['z'] > threshold).astype(int)
    
    print(f"\n--- Results for {model_path} ---")
    print(f"Mean Z-Error: {report['z'].mean():.4f}")
    print(f"F1-Score:    {f1_score(report['y'], y_pred):.4f}")
    print(f"Precision:   {precision_score(report['y'], y_pred):.4f}")
    print(f"Recall:      {recall_score(report['y'], y_pred):.4f}")

if __name__ == "__main__":
    run_comparative_audit("specialized_network_llama.pt")
    run_comparative_audit("lag-llama.ckpt")
