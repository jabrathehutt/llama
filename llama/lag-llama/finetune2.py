import sys, types, torch, numpy as np, pandas as pd, random, os
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from lag_llama.gluon.estimator import LagLlamaLightningModule
from gluonts.torch.distributions import StudentTOutput

# --- 1. ENVIRONMENT SHIMS & SEEDING ---
def mock_strtobool(val):
    val = str(val).lower()
    return 1 if val in ('y', 'yes', 't', 'true', 'on', '1') else 0
d_util = types.ModuleType("distutils.util"); d_util.strtobool = mock_strtobool
sys.modules["distutils"] = types.ModuleType("distutils"); sys.modules["distutils.util"] = d_util

m = types.ModuleType("gluonts.torch.modules.loss")
class MockLoss: pass
m.NegativeLogLikelihood = MockLoss; m.DistributionLoss = MockLoss
sys.modules["gluonts.torch.modules.loss"] = m

def seed_everything(seed=42):
    random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(42)

# --- 2. CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOAD_PATH = "lag-llama-backbone.ckpt"
METRICS_CSV = "/root/network_finetune.csv"
FINAL_PATH = "specialized_v11_supervised.pt"

# Lags/Arch matching your validated 86-dim backbone
CONTEXT_LEN = 96 
N_LAYER = 1      
N_HEAD = 8       
LAGS_SEQ = list(range(1, 85)) 

class DualMarginDataset(Dataset):
    def __init__(self, csv_path, context_len):
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        flow0 = df[df['flow_key_id'] == '0'].sort_values('timestamp')
        
        v = flow0['traffic_volume'].values.astype('float32')
        l = flow0['is_anomaly'].astype('float32').values
        
        self.samples = []
        self.context_len = context_len
        if len(v) > self.context_len:
            for i in range(self.context_len, len(v)):
                self.samples.append({'x': v[i-self.context_len:i], 'y': v[i], 'label': l[i]})

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        return (torch.tensor(self.samples[i]['x']),
                torch.tensor(self.samples[i]['y']),
                torch.tensor(self.samples[i]['label']))

# --- 3. RECALL-OPTIMIZED TUNING ---
def run_recall_boost_tuning():
    full_dataset = DualMarginDataset(METRICS_CSV, CONTEXT_LEN)

    module = LagLlamaLightningModule(
        context_length=CONTEXT_LEN, prediction_length=1,
        model_kwargs={
            "context_length": CONTEXT_LEN, "max_context_length": 1024, "input_size": 1,
            "distr_output": StudentTOutput(), "n_layer": N_LAYER, "n_head": N_HEAD,
            "n_embd_per_head": 16, "lags_seq": LAGS_SEQ, "scaling": "mean", "time_feat": False,
        }
    )

    if os.path.exists(LOAD_PATH):
        sd = torch.load(LOAD_PATH, map_location=DEVICE, weights_only=False)
        new_sd = { (k if k.startswith("model.") else f"model.{k}"): v for k, v in sd.items() }
        module.load_state_dict(new_sd, strict=True) 

    model = module.model.to(DEVICE); model.train()
    # Lower LR for finer control over the anomaly boundary
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-7)
    loader = DataLoader(full_dataset, batch_size=64, shuffle=True)

    # Increased epochs to help model find the "drift" patterns
    for epoch in range(15):
        pbar = tqdm(loader, desc=f"Recall Boost Epoch {epoch+1}")
        for x, y, label in pbar:
            x, y, label = x.to(DEVICE), y.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            with torch.enable_grad():
                scale_f = x.mean(dim=1, keepdim=True) + 1e-5
                distr_args, _, _ = model(past_target=x/scale_f, past_observed_values=torch.ones_like(x).to(DEVICE))

                p_loc, p_scale = distr_args[0][:, -1], torch.exp(distr_args[1][:, -1])
                y_norm = y / scale_f.squeeze()
                z = torch.abs(p_loc - y_norm) / (p_scale + 1e-10)

                # --- LOSS ADJUSTMENTS ---
                # 1. Reduce normal pressure (50.0 -> 15.0) to allow more sensitivity
                loss_normal = ((1 - label) * (torch.pow(z, 2) + 15.0 * p_scale)).mean()
                
                # 2. Lower margin (250 -> 100) to pull smaller deviations into anomalous territory
                loss_anomaly = (label * (torch.clamp(100.0 - z, min=0) + 10.0 * p_scale)).mean()
                
                # 3. Increase anomaly weight slightly to 60x
                total_loss = (1.0 * loss_normal) + (60.0 * loss_anomaly)

                total_loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=f"{total_loss.item():.2f}")

    torch.save(model.state_dict(), FINAL_PATH)
    print(f"Weights saved to {FINAL_PATH}")

if __name__ == "__main__":
    run_recall_boost_tuning()
