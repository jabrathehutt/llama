import sys, types, torch, numpy as np, pandas as pd, random, os
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# Core Lag-Llama and GluonTS imports
from lag_llama.gluon.estimator import LagLlamaLightningModule
from gluonts.torch.distributions import StudentTOutput

# --- 1. ENVIRONMENT SHIMS ---
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
LOAD_PATH = "specialized_v7_pretrain.pt"
METRICS_CSV = "/root/network_finetune.csv"
FINAL_PATH = "specialized_v11_supervised.pt"
CONTEXT_LEN = 256 

def get_train_test_split(csv_path, split_ratio=0.8):
    df = pd.read_csv(csv_path)
    unique_flows = sorted(df['flow_key_id'].unique())
    random.seed(42)
    random.shuffle(unique_flows)
    split_idx = int(len(unique_flows) * split_ratio)
    return set(unique_flows[:split_idx]), set(unique_flows[split_idx:])

class DualMarginDataset(Dataset):
    def __init__(self, csv_path, allowed_flows, context_len):
        df = pd.read_csv(csv_path)
        time_col = [c for c in df.columns if 'time' in c.lower()][0]
        df[time_col] = pd.to_datetime(df[time_col])
        df = df[df['flow_key_id'].isin(allowed_flows)]
        self.samples = []
        self.context_len = context_len
        
        for flow_id, group in tqdm(df.groupby('flow_key_id'), desc="Dataset Build"):
            flow = group.sort_values(time_col)
            v = flow['traffic_volume_Tbits'].values
            l = flow['is_anomaly'].astype(float).values
            for i in range(self.context_len, len(flow), 5):
                self.samples.append({'x': v[i-self.context_len:i], 'y': v[i], 'label': l[i]})

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        return (torch.tensor(self.samples[i]['x']).float(),
                torch.tensor(self.samples[i]['y']).float(),
                torch.tensor(self.samples[i]['label']).float())

# --- 3. AGGRESSIVE TUNING LOOP ---
def run_aggressive_recall_tuning():
    train_flows, _ = get_train_test_split(METRICS_CSV)
    raw_lags = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216, 228, 240, 252, 264, 276, 288, 300, 312, 324, 336, 348, 360, 372, 384, 396, 408, 420, 432, 444, 456, 468, 480, 492, 504, 516, 528, 540, 552, 564, 576]
    lags_seq = [l for l in raw_lags if l < CONTEXT_LEN]

    module = LagLlamaLightningModule(
        context_length=CONTEXT_LEN, prediction_length=1,
        model_kwargs={
            "context_length": CONTEXT_LEN, "max_context_length": 1024, "input_size": 1,
            "distr_output": StudentTOutput(), "n_layer": 8, "n_head": 9,
            "n_embd_per_head": 16, "lags_seq": lags_seq, "scaling": "mean", "time_feat": False,
        }
    )

    if os.path.exists(LOAD_PATH):
        sd = torch.load(LOAD_PATH, map_location=DEVICE, weights_only=False)
        new_sd = { (k if k.startswith("model.") else f"model.{k}"): v for k, v in sd.items() }
        current = module.state_dict()
        loadable = {k: v for k, v in new_sd.items() if k in current and v.size() == current[k].size()}
        current.update(loadable)
        module.load_state_dict(current)

    model = module.model.to(DEVICE); model.train()
    # Slightly higher learning rate to move the weights for the anomaly class
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    loader = DataLoader(DualMarginDataset(METRICS_CSV, train_flows, CONTEXT_LEN), batch_size=64, shuffle=True)

    for epoch in range(5):
        pbar = tqdm(loader, desc=f"Aggressive Epoch {epoch+1}")
        for x, y, label in pbar:
            x, y, label = x.to(DEVICE), y.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            with torch.enable_grad():
                scale_f = x.mean(dim=1, keepdim=True) + 1e-5
                distr_args, _, _ = model(past_target=x/scale_f, past_observed_values=torch.ones_like(x).to(DEVICE))
                
                p_loc, p_scale = distr_args[0][:, -1], torch.exp(distr_args[1][:, -1])
                y_norm = y / scale_f.squeeze()
                z = torch.abs(p_loc - y_norm) / (p_scale + 1e-10)

                # 1. NORMAL LOSS: Tighten uncertainty
                loss_normal = ((1 - label) * (torch.pow(z, 2) + 50.0 * p_scale)).mean()
                
                # 2. ANOMALY LOSS: Force Z way higher (250 margin)
                # We add a p_scale penalty to prevent the model from expanding uncertainty to lower Z
                loss_anomaly = (label * (torch.clamp(250.0 - z, min=0) + 10.0 * p_scale)).mean()
                
                # 3. BALANCED TOTAL: Double the weight of the anomaly penalty
                total_loss = (1.0 * loss_normal) + (50.0 * loss_anomaly)

                total_loss.backward()
                optimizer.step()
                
                z_anom_val = z[label==1].mean().item() if (label==1).any() else 0
                pbar.set_postfix(loss=f"{total_loss.item():.2f}", z_anom=f"{z_anom_val:.1f}")

    torch.save(model.state_dict(), FINAL_PATH)
    print(f"Aggressive Supervised weights saved: {FINAL_PATH}")

if __name__ == "__main__":
    run_aggressive_recall_tuning()
