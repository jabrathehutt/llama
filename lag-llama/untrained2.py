import os
import sys
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# --- 1. THE LEGACY BRIDGE ---
from types import ModuleType
import gluonts.torch.distributions as gtd

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

torch.serialization.add_safe_globals([gtd.studentT.StudentTOutput, gtd.StudentTOutput])
create_dummy_module('gluonts.torch.modules.loss')
class MockL:
    def __init__(self, *args, **kwargs): pass
    def __getattr__(self, name): return lambda *args, **kwargs: None
sys.modules['gluonts.torch.modules.loss'].DistributionLoss = MockL
sys.modules['gluonts.torch.modules.loss'].NegativeLogLikelihood = MockL

from lag_llama.gluon.lightning_module import LagLlamaLightningModule

# --- 2. CONFIG ---
OFFICIAL_CKPT = "lag-llama.ckpt" # Ensure this file is in your directory
DATA_FILE = "network_metrics_data.csv"
CONTEXT_LENGTH = 32
# Official weights use a specific lag sequence. We match your 33-lag config.
LAG_SEQ = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 48, 72, 96, 120, 144, 288, 576]
LAG_BUFFER = 576 + CONTEXT_LENGTH

def test_official():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize fresh structure
    model = LagLlamaLightningModule(
        prediction_length=1, context_length=CONTEXT_LENGTH,
        model_kwargs={
            "n_layer": 8, "n_head": 9, "n_embd_per_head": 16,
            "lags_seq": LAG_SEQ, 
            "context_length": CONTEXT_LENGTH, "max_context_length": 1024, 
            "scaling": "mean", "input_size": 1, "distr_output": gtd.StudentTOutput(), "time_feat": False
        }
    )

    # 2. Load and Graft Official Weights
    print(f"Loading official weights from {OFFICIAL_CKPT}...")
    ckpt = torch.load(OFFICIAL_CKPT, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    
    # Fix Size Mismatch for the Lag Embedding layer (WTE)
    wte_key = "model.transformer.wte.weight"
    if wte_key in state_dict:
        learned_w = state_dict[wte_key]
        target_w = model.state_dict()[wte_key]
        if learned_w.shape != target_w.shape:
            print(f"Grafting WTE: {learned_w.shape} -> {target_w.shape}")
            new_w = torch.zeros_like(target_w)
            # Copy only the overlapping weights
            rows = min(learned_w.shape[0], target_w.shape[0])
            cols = min(learned_w.shape[1], target_w.shape[1])
            new_w[:rows, :cols] = learned_w[:rows, :cols]
            state_dict[wte_key] = new_w

    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    # 3. Test on a slice of your data
    df = pd.read_csv(DATA_FILE)
    df_slice = df.head(LAG_BUFFER + 10).copy()
    
    results = []
    with torch.no_grad():
        raw_values = np.log1p(df_slice['traffic_volume_Tbits'].values.astype(np.float32))
        
        for i in range(LAG_BUFFER, len(raw_values)):
            past_target = raw_values[i - LAG_BUFFER : i]
            m = past_target.mean() + 1e-8
            input_tensor = torch.from_numpy(past_target / m).unsqueeze(0).to(device)
            
            _, loc, scale = model.model(past_target=input_tensor, past_observed_values=torch.ones_like(input_tensor).to(device))
            
            mu, sigma = loc[0, -1].item(), scale[0, -1].item()
            results.append({
                "Actual_Sc": round(raw_values[i]/m, 4),
                "Pred_Sc": round(mu, 4),
                "Sigma": round(sigma, 4)
            })

    print("\n--- RESULTS WITH OFFICIAL WEIGHTS ---")
    report = pd.DataFrame(results)
    print(report.to_string(index=False))
    
    # Analyze results
    if report['Pred_Sc'].sum() == 0:
        print("\n[!] CRITICAL: Even official weights predict 0.0. The issue is likely in the input tensor or model initialization logic.")
    else:
        print("\n[+] SUCCESS: Official weights are producing predictions. Your previous custom weights are the issue.")

if __name__ == "__main__":
    test_official()
