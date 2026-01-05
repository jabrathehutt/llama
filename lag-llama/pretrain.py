# --- 1. THE SHIMS ---
import sys, types, torch, pickle, numpy as np, pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from lag_llama.gluon.estimator import LagLlamaEstimator
import gluonts.torch.distributions as gtd

m = types.ModuleType("gluonts.torch.modules.loss"); sys.modules["gluonts.torch.modules.loss"] = m
class MockLoss: pass
m.NegativeLogLikelihood = MockLoss; m.DistributionLoss = MockLoss
d = types.ModuleType("distutils"); sys.modules["distutils"] = d
u = types.ModuleType("distutils.util"); sys.modules["distutils.util"] = u
setattr(d, "util", u); u.strtobool = lambda v: 1 if str(v).lower() in ("y", "yes", "t", "true", "1") else 0

# --- CONFIG ---
DEVICE = torch.device("cuda")
CKPT_PATH = "lag-llama.ckpt"
DATA_PKL = "datasets/network_telemetry/processed_data.pkl"
SAVE_PATH = "specialized_v6_refined.pt"

def train_refined():
    print("Initializing Refined Architecture...")
    estimator = LagLlamaEstimator(
        ckpt_path=None, prediction_length=1, context_length=32,
        lags_seq=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216, 228, 240, 252, 264, 276, 288, 300, 312, 324, 336, 348, 360, 372, 384, 396, 408, 420, 432, 444, 456, 468, 480, 492, 504, 516, 528, 540, 552, 564, 576],
        n_layer=8, n_head=9, n_embd_per_head=16, scaling="mean", trainer_kwargs=dict(accelerator="gpu", devices=1)
    )

    lightning_module = estimator.create_lightning_module()
    
    # Load Foundation Weights (using surgical alignment)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    if sd["model.transformer.wte.weight"].shape[1] == 92:
        new_wte = torch.zeros((144, 94), device=DEVICE)
        new_wte[:, :92] = sd["model.transformer.wte.weight"]
        sd["model.transformer.wte.weight"] = new_wte
    lightning_module.load_state_dict(sd, strict=False)
    
    model = lightning_module.model.to(DEVICE)
    for p in model.parameters(): p.requires_grad = True
    model.train()

    # Load Data
    with open(DATA_PKL, "rb") as f: data_list = pickle.load(f)
    x_s, y_s = [], []
    for entry in data_list:
        v = entry['target']
        if len(v) > 608:
            for i in range(608, len(v), 10): # Even wider steps to prevent overfitting
                x_s.append(v[i-608:i]); y_s.append(v[i])
    
    loader = DataLoader(TensorDataset(torch.tensor(np.array(x_s)).float(), torch.tensor(np.array(y_s)).float()), batch_size=128, shuffle=True)
    
    # LOWER LEARNING RATE (1e-6) to maintain stability
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    distr_output = gtd.StudentTOutput() # Use the proper distribution head

    print("Starting Refined Training (Student-T Loss)...")
    for epoch in range(3):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            
            with torch.enable_grad():
                x_mean = x.mean(dim=1, keepdim=True) + 1e-5
                x_norm, y_norm = x / x_mean, y / x_mean.squeeze()
                
                # Forward Pass
                distr_args, loc, scale = model(past_target=x_norm, past_observed_values=torch.ones_like(x_norm), use_kv_cache=False)
                
                # Proper Distributional Loss
                distr = distr_output.distribution(distr_args)
                loss = -distr.log_prob(y_norm.unsqueeze(-1)).mean()
                
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=loss.item())

    print(f"Refined weights saved to {SAVE_PATH}")
    torch.save(model.state_dict(), SAVE_PATH)

if __name__ == "__main__":
    train_refined()
