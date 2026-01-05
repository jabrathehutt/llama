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
PREDICTION_LEN = 48  # The "Stress Test" Horizon
WINDOW_SIZE = MODEL_INPUT_LEN + SEASONAL_PERIOD + PREDICTION_LEN
BATCH_SIZE = 200 

def run_long_horizon_report():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "specialized_network_llama.pt"
    DATA_FILE = "network_metrics_data.csv"

    # --- 3. MODEL SETUP ---
    model = LagLlamaLightningModule(
        prediction_length=PREDICTION_LEN, context_length=32,
        model_kwargs={
            "n_layer": 8, "n_head": 9, "n_embd_per_head": 16,
            "lags_seq": lags_seq_92, "context_length": 32,
            "max_context_length": 1024, "scaling": "mean",
            "input_size": 1, "distr_output": gtd.StudentTOutput(), "time_feat": False
        }
    )

    if os.path.exists(MODEL_PATH):
        ckpt_state = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        if "state_dict" in ckpt_state: ckpt_state = ckpt_state["state_dict"]
        model_state = model.state_dict()
        if "model.transformer.wte.weight" in ckpt_state:
            lw, tw = ckpt_state["model.transformer.wte.weight"], model_state["model.transformer.wte.weight"]
            if lw.shape != tw.shape:
                new_w = torch.zeros_like(tw)
                new_w[:lw.shape[0], :lw.shape[1]] = lw
                ckpt_state["model.transformer.wte.weight"] = new_w
        model.load_state_dict(ckpt_state, strict=False)

    model.to(device).eval()

    # --- 4. LONG-HORIZON ANALYSIS ---
    df = pd.read_csv(DATA_FILE)
    results = []
    all_flow_ids = df['flow_key_id'].unique()

    print(f"Analyzing {len(all_flow_ids)} flows at Step {PREDICTION_LEN} horizon...")

    with torch.no_grad():
        for flow_id in tqdm(all_flow_ids):
            group = df[df['flow_key_id'] == flow_id]
            raw_values = group['traffic_volume_Tbits'].values.astype(np.float32)
            is_anomaly_truth = group['is_anomaly'].values.astype(int)
            category = flow_id.split('_')[-1]

            if len(raw_values) < WINDOW_SIZE: continue

            diffed_values = raw_values[SEASONAL_PERIOD:] - raw_values[:-SEASONAL_PERIOD]
            # Adjust indices to account for the prediction horizon
            indices = np.arange(MODEL_INPUT_LEN, len(diffed_values) - PREDICTION_LEN + 1)

            for b in range(0, len(indices), BATCH_SIZE):
                batch_idx = indices[b : b + BATCH_SIZE]
                
                # Context windows
                windows = np.array([diffed_values[i-MODEL_INPUT_LEN : i] for i in batch_idx])
                
                # The "Truth" is now PREDICTION_LEN steps ahead of the window end
                target_actuals = np.array([diffed_values[i + PREDICTION_LEN - 1] for i in batch_idx])
                # Find if an anomaly exists ANYWHERE in that 48-step future window
                truths = np.array([1 if any(is_anomaly_truth[SEASONAL_PERIOD + i : SEASONAL_PERIOD + i + PREDICTION_LEN]) else 0 for i in batch_idx])

                # Normalization
                w_means, w_stds = windows.mean(axis=1, keepdims=True), windows.std(axis=1, keepdims=True) + 1e-8
                scaled_windows = (windows - w_means) / w_stds
                scaled_targets = (target_actuals - w_means.flatten()) / w_stds.flatten()

                past_target = torch.from_numpy(scaled_windows).to(device)
                _, loc, scale_pred = model.model(past_target=past_target, past_observed_values=torch.ones_like(past_target).to(device))

                # Extract only the LAST step of the multi-step forecast
                f_mu = loc[:, -1].cpu().numpy()
                f_sigma = scale_pred[:, -1].cpu().numpy()

                # Anomaly score based on long-range error
                z = np.abs(scaled_targets - f_mu) / (f_sigma + 1e-4)

                for k in range(len(z)):
                    results.append({"cat": category, "score": float(z[k]), "y": int(truths[k])})

    report_df = pd.DataFrame(results)

    # --- 5. REPORTING ---
    print("\nCalculating optimal horizon threshold...")
    best_f1, best_t = 0, 0
    for p in np.arange(80, 99.9, 0.1):
        test_t = float(report_df['score'].quantile(p/100))
        y_p = (report_df['score'] > test_t).astype(int)
        f1 = f1_score(report_df['y'], y_p, zero_division=0)
        if precision_score(report_df['y'], y_p, zero_division=0) > 0.70 and f1 > best_f1:
            best_f1, best_t = f1, test_t

    report_df['pred'] = (report_df['score'] > best_t).astype(int)

    print("\n" + "="*45)
    print(f"   LONG-HORIZON REPORT (H={PREDICTION_LEN})")
    print("="*45)
    y_t, y_p = report_df['y'], report_df['pred']
    print(f"Precision: {precision_score(y_t, y_p, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_t, y_p, zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_t, y_p, zero_division=0):.4f}")
    print("-" * 45)
    for c in sorted(report_df['cat'].unique()):
        sub = report_df[report_df['cat'] == c]
        print(f"Recall for {c.upper():<10}: {recall_score(sub['y'], sub['pred'], zero_division=0):.4f}")
    print("="*45)

if __name__ == "__main__":
    run_long_horizon_report()
