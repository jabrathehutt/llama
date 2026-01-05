import sys
from types import ModuleType

# --- 1. THE SURGICAL LEGACY BRIDGE ---
# We only create the path for the missing loss modules to satisfy the unpickler
def patch_gluonts():
    if 'gluonts' not in sys.modules:
        sys.modules['gluonts'] = ModuleType('gluonts')
    
    # Create torch as a sub-package of gluonts if it doesn't exist
    if 'gluonts.torch' not in sys.modules:
        torch_mod = ModuleType('gluonts.torch')
        sys.modules['gluonts.torch'] = torch_mod
        setattr(sys.modules['gluonts'], 'torch', torch_mod)
        
    # Create the specific path the unpickler needs
    if 'gluonts.torch.modules' not in sys.modules:
        m_mod = ModuleType('gluonts.torch.modules')
        sys.modules['gluonts.torch.modules'] = m_mod
        setattr(sys.modules['gluonts.torch'], 'modules', m_mod)
        
    if 'gluonts.torch.modules.loss' not in sys.modules:
        l_mod = ModuleType('gluonts.torch.modules.loss')
        sys.modules['gluonts.torch.modules.loss'] = l_mod
        setattr(sys.modules['gluonts.torch.modules'], 'loss', l_mod)
        
        # Inject the mock classes into the newly created path
        class MockLoss:
            def __init__(self, *args, **kwargs): pass
            def __call__(self, *args, **kwargs): return 0.0
        l_mod.NegativeLogLikelihood = MockLoss
        l_mod.DistributionLoss = MockLoss

patch_gluonts()

# --- 2. NOW IMPORT REAL LIBRARIES ---
import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import gluonts.torch.distributions as gtd
from lag_llama.gluon.lightning_module import LagLlamaLightningModule

# Register for PyTorch 2.6+
torch.serialization.add_safe_globals([gtd.studentT.StudentTOutput, gtd.StudentTOutput])

def run_real_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_FILE = "clean_training_traffic_data.csv"
    BASE_MODEL = "lag-llama.ckpt"
    SAVE_PATH = "actually_specialized_llama.pt"

    print(f"Using device: {device}")

    lags_seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216, 228, 240, 252, 264, 276, 288, 300, 312, 324, 336, 348, 360, 372, 384, 396, 408, 420, 432, 444, 456, 468, 480, 492, 504, 516, 528, 540, 552, 564, 576]
    
    model = LagLlamaLightningModule(
        prediction_length=1, context_length=32,
        model_kwargs={
            "n_layer": 8, "n_head": 9, "n_embd_per_head": 16,
            "lags_seq": lags_seq, "context_length": 32, "max_context_length": 1024,
            "scaling": "mean", "input_size": 1, "distr_output": gtd.StudentTOutput(), "time_feat": False
        }
    )

    print(f"Loading {BASE_MODEL}...")
    ckpt = torch.load(BASE_MODEL, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    m_state = model.state_dict()
    
    wte_key = "model.transformer.wte.weight" if "model.transformer.wte.weight" in m_state else "transformer.wte.weight"
    if wte_key in state:
        new_w = torch.zeros_like(m_state[wte_key])
        l_w = state[wte_key]
        r, c = min(l_w.shape[0], new_w.shape[0]), min(l_w.shape[1], new_w.shape[1])
        new_w[:r, :c] = l_w[:r, :c]
        state[wte_key] = new_w

    model.load_state_dict(state, strict=False)
    model.to(device).train()

    # --- 3. DATASET ---
    df = pd.read_csv(DATA_FILE)
    
    class TrafficDataset(Dataset):
        def __init__(self, dataframe, window_len=608):
            self.samples = []
            for _, group in dataframe.groupby('flow_key_id'):
                v = group['traffic_volume_Tbits'].values.astype(np.float32)
                # Apply same global scaling used in previous turn
                mu, std = v.mean(), v.std() + 1e-8
                v = (v - mu) / std
                if len(v) > window_len + 1:
                    for i in range(window_len, len(v)):
                        self.samples.append((v[i-window_len:i], v[i]))
        def __len__(self): return len(self.samples)
        def __getitem__(self, idx):
            x, y = self.samples[idx]
            return torch.tensor(x), torch.tensor(y)

    loader = DataLoader(TrafficDataset(df), batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # --- 4. TRAINING LOOP ---
    print(f"Training on {len(loader)} batches...")
    for epoch in range(3):
        epoch_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            _, loc, _ = model.model(past_target=x, past_observed_values=torch.ones_like(x))
            loss = torch.mean((loc[:, -1] - y)**2)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"SUCCESS: Model saved to {SAVE_PATH}")

if __name__ == "__main__":
    run_real_training()
