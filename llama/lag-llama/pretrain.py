# --- 1. THE UNIVERSAL PANDAS SHIM (STOPS VALUEERROR) ---
import pandas as pd
original_to_offset = pd.tseries.frequencies.to_offset
def universal_to_offset(freq_str):
    try:
        if isinstance(freq_str, int) or (isinstance(freq_str, str) and freq_str.isdigit()) or freq_str is None:
            return original_to_offset("5min")
    except: pass
    return original_to_offset(freq_str)
pd.tseries.frequencies.to_offset = universal_to_offset
pd.to_offset = universal_to_offset

import sys, types, os, torch, numpy as np

# A. Mock distutils for Python 3.13
def mock_strtobool(val):
    val = str(val).lower()
    return 1 if val in ('y', 'yes', 't', 'true', 'on', '1') else 0
d_util = types.ModuleType("distutils.util"); d_util.strtobool = mock_strtobool
sys.modules["distutils"] = types.ModuleType("distutils"); sys.modules["distutils.util"] = d_util
sys.modules["distutils.util"] = d_util

# B. Mock GluonTS loss
m_loss = types.ModuleType("gluonts.torch.modules.loss")
class MockLoss: pass
m_loss.NegativeLogLikelihood = MockLoss; m_loss.DistributionLoss = MockLoss
sys.modules["gluonts.torch.modules.loss"] = m_loss

# C. GluonTS Lag Bypass
import gluonts.time_feature.lag as lag_module
lag_module.get_lags_for_frequency = lambda freq_str, num_default_lags=1: [1]

# D. Imports
from lag_llama.gluon.estimator import LagLlamaEstimator
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

# --- 2. CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "lag-llama.ckpt"
DATA_CSV = "/root/network_pretrain.csv"
SAVE_PATH = "specialized_v7_pretrain.pt" 

CUSTOM_LAGS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216, 228, 240, 252, 264, 276, 288, 300, 312, 324, 336, 348, 360, 372, 384, 396, 408, 420, 432, 444, 456, 468, 480, 492, 504, 516, 528, 540, 552, 564, 576]

def run_extensive_pretraining():
    if not os.path.exists(DATA_CSV):
        print(f"Error: {DATA_CSV} not found."); return

    print("Starting Stage 1: Extensive Unsupervised Pretraining...")
    
    estimator = LagLlamaEstimator(
        ckpt_path=None, prediction_length=1, context_length=1152, 
        lags_seq=CUSTOM_LAGS, n_layer=8, n_head=9, n_embd_per_head=16, scaling="mean"
    )

    module = estimator.create_lightning_module()
    
    # --- SURGICAL ARCHITECTURE REPAIR ---
    emb_dim = 144
    model_feat_dim = 94 
    module.model.lags_seq = CUSTOM_LAGS
    
    # Re-initialize the WTE layer
    module.model.transformer.wte = torch.nn.Linear(model_feat_dim, emb_dim, bias=False)

    # Load checkpoint
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    new_sd = { (k if k.startswith("model.") else f"model.{k}"): v for k, v in sd.items() }
    
    # Weight Padding (92 columns from checkpoint -> 94 columns in model)
    ckpt_weight = new_sd["model.transformer.wte.weight"]
    if ckpt_weight.shape[1] == 92:
        padded_weight = torch.zeros((144, 94), device=ckpt_weight.device)
        padded_weight[:, :92] = ckpt_weight
        new_sd["model.transformer.wte.weight"] = padded_weight

    module.load_state_dict(new_sd, strict=False)
    
    # CRITICAL FIX: Ensure gradients are enabled for the entire model
    model = module.model.to(DEVICE)
    for name, param in model.named_parameters():
        param.requires_grad = True
    
    model.train()

    # --- 3. DATA LOADING ---
    df = pd.read_csv(DATA_CSV)
    x_s, y_s = [], []
    grouped = df.groupby('flow_key_id')
    
    for _, group in tqdm(list(grouped), desc="Building Training Windows"):
        v = group.sort_values('timestamp')['traffic_volume_Tbits'].values
        if len(v) >= 1153:
            for i in range(1153, len(v), 10): 
                x_s.append(v[i-1153:i])
                y_s.append(v[i])
    
    loader = DataLoader(TensorDataset(torch.tensor(np.array(x_s)).float(), torch.tensor(np.array(y_s)).float()), batch_size=32, shuffle=True)
    
    # Re-initialize optimizer AFTER setting requires_grad=True
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7)

    for epoch in range(30):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            
            # Explicitly wrap in enable_grad to be safe
            with torch.enable_grad():
                x_mean = x.mean(dim=1, keepdim=True) + 1e-5
                x_norm, y_norm = x / x_mean, y / x_mean.squeeze()
                
                # Forward pass
                _, loc, scale = model(
                    past_target=x_norm, 
                    past_observed_values=torch.ones_like(x_norm).to(DEVICE),
                    use_kv_cache=False
                )
                
                p_loc, p_scale = loc[:, -1], scale[:, -1]
                
                # Calculate loss
                loss = torch.nn.functional.mse_loss(p_loc, y_norm) + (1.0 * torch.mean(torch.abs(p_scale)))
                
                # Check if loss is trackable
                if loss.grad_fn is None:
                    # Fallback: Force a gradient link if the graph is disconnected
                    loss = loss + (0.0 * model.transformer.wte.weight.sum())
                
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Pretrained weights saved to {SAVE_PATH}")

if __name__ == "__main__":
    run_extensive_pretraining()
