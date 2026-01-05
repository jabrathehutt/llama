import os
import sys
import torch
import hashlib
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
PREDICTION_LEN = 48
WINDOW_SIZE = MODEL_INPUT_LEN + SEASONAL_PERIOD + PREDICTION_LEN
BATCH_SIZE = 256

def get_model_hash(model):
    """Calculates a hash of all model parameters to verify weight loading."""
    hash_gen = hashlib.sha256()
    for param in model.parameters():
        hash_gen.update(param.data.cpu().numpy().tobytes())
    return hash_gen.hexdigest()[:16]

def run_audit_report():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "lag-llama.ckpt" # Ensure this filename is correct for each run
    DATA_FILE = "network_metrics_data.csv"

    # --- 3. MODEL INITIALIZATION ---
    model = LagLlamaLightningModule(
        prediction_length=PREDICTION_LEN, context_length=32,
        model_kwargs={
            "n_layer": 8, "n_head": 9, "n_embd_per_head": 16,
            "lags_seq": lags_seq_92, "context_length": 32,
            "max_context_length": 1024, "scaling": "mean",
            "input_size": 1, "distr_output": gtd.StudentTOutput(), "time_feat": False
        }
    )

    print(f"Initial Model Hash: {get_model_hash(model)}")

    if os.path.exists(MODEL_PATH):
        print(f"Loading weights from: {MODEL_PATH}")
        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        
        # Grafting for WTE mismatch
        model_dict = model.state_dict()
        key = "model.transformer.wte.weight" if "model.transformer.wte.weight" in state_dict else "transformer.wte.weight"
        m_key = "model.transformer.wte.weight" if "model.transformer.wte.weight" in model_dict else "transformer.wte.weight"
        
        if key in state_dict:
            l_w = state_dict[key]
            t_w = model_dict[m_key]
            if l_w.shape != t_w.shape:
                print(f"Grafting {key} {l_w.shape} -> {t_w.shape}")
                new_w = torch.zeros_like(t_w)
                new_w[:l_w.shape[0], :l_w.shape[1]] = l_w
                state_dict[m_key] = new_w

        # THE FIX: Force correct keys if necessary
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Load Status: {msg}")
    else:
        print("WARNING: Checkpoint file not found! Using random weights.")

    model.to(device).eval()
    print(f"Post-Load Model Hash: {get_model_hash(model)}")

    # --- 4. DATA ANALYSIS ---
    df = pd.read_csv(DATA_FILE)
    results = []
    all_flow_ids = df['flow_key_id'].unique()

    with torch.no_grad():
        for flow_id in tqdm(all_flow_ids, desc="Audit Scanning"):
            group = df[df['flow_key_id'] == flow_id]
            vals = group['traffic_volume_Tbits'].values.astype(np.float32)
            truths = group['is_anomaly'].values.astype(int)
            category = flow_id.split('_')[-1]
            if len(vals) < WINDOW_SIZE: continue

            diffed = vals[SEASONAL_PERIOD:] - vals[:-SEASONAL_PERIOD]
            indices = np.arange(MODEL_INPUT_LEN, len(diffed) - PREDICTION_LEN + 1)

            for b in range(0, len(indices), BATCH_SIZE):
                batch_idx = indices[b : b + BATCH_SIZE]
                windows = np.array([diffed[i-MODEL_INPUT_LEN : i] for i in batch_idx])
                target_actuals = np.array([diffed[i + PREDICTION_LEN - 1] for i in batch_idx])
                
                # Label is 1 if ANY point in the future window is an anomaly
                y_true = np.array([1 if any(truths[SEASONAL_PERIOD + i : SEASONAL_PERIOD + i + PREDICTION_LEN]) else 0 for i in batch_idx])

                w_mu, w_std = windows.mean(axis=1, keepdims=True), windows.std(axis=1, keepdims=True) + 1e-8
                s_win = (windows - w_mu) / w_std
                s_target = (target_actuals - w_mu.flatten()) / w_std.flatten()

                past_target = torch.from_numpy(s_win).to(device)
                _, loc, scale = model.model(past_target=past_target, past_observed_values=torch.ones_like(past_target).to(device))

                f_mu, f_sigma = loc[:, -1].cpu().numpy(), scale[:, -1].cpu().numpy()
                z = np.abs(s_target - f_mu) / (f_sigma + 1e-4)

                for k in range(len(z)):
                    results.append({"cat": category, "z": float(z[k]), "y": int(y_true[k])})

    # --- 5. REPORT ---
    report_df = pd.DataFrame(results)
    best_f1, best_t = 0, 0
    for p in np.arange(85, 99.9, 0.1):
        t = float(report_df['z'].quantile(p/100))
        y_p = (report_df['z'] > t).astype(int)
        f1 = f1_score(report_df['y'], y_p, zero_division=0)
        if precision_score(report_df['y'], y_p, zero_division=0) > 0.85 and f1 > best_f1:
            best_f1, best_t = f1, t

    report_df['pred'] = (report_df['z'] > best_t).astype(int)
    print("\n" + "="*45 + f"\nREPORT FOR {MODEL_PATH}\n" + "="*45)
    print(f"F1: {f1_score(report_df['y'], report_df['pred']):.4f} | Hash: {get_model_hash(model)}")
    for c in sorted(report_df['cat'].unique()):
        sub = report_df[report_df['cat'] == c]
        print(f"Recall {c.upper():<10}: {recall_score(sub['y'], sub['pred'], zero_division=0):.4f}")

if __name__ == "__main__":
    run_audit_report()
