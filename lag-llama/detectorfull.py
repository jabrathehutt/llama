# --- 1. THE SHIMS ---
import sys, types, torch, numpy as np, pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import make_evaluation_predictions
from lag_llama.gluon.estimator import LagLlamaEstimator

m = types.ModuleType("gluonts.torch.modules.loss"); sys.modules["gluonts.torch.modules.loss"] = m
class MockLoss: pass
m.NegativeLogLikelihood = MockLoss; m.DistributionLoss = MockLoss
d = types.ModuleType("distutils"); sys.modules["distutils"] = d
u = types.ModuleType("distutils.util"); sys.modules["distutils.util"] = u
setattr(d, "util", u); u.strtobool = lambda v: 1 if str(v).lower() in ("y", "yes", "t", "true", "1") else 0

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "lag-llama.ckpt"
SPECIALIZED_PATH = "specialized_v5_final.pt"
DATA_FILE = "network_metrics_data.csv"

def get_aligned_predictor(mode="foundation"):
    estimator = LagLlamaEstimator(
        ckpt_path=None, prediction_length=1, context_length=32,
        lags_seq=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216, 228, 240, 252, 264, 276, 288, 300, 312, 324, 336, 348, 360, 372, 384, 396, 408, 420, 432, 444, 456, 468, 480, 492, 504, 516, 528, 540, 552, 564, 576],
        n_layer=8, n_head=9, n_embd_per_head=16, scaling="mean", trainer_kwargs=dict(accelerator="gpu", devices=1)
    )

    lightning_module = estimator.create_lightning_module()

    if mode in ["foundation", "specialized"]:
        # Load foundation or specialized weights
        path = CKPT_PATH if mode == "foundation" else SPECIALIZED_PATH
        print(f"Loading {mode} weights from {path}...")
        state_dict = torch.load(path, map_location=DEVICE, weights_only=False)
        if "state_dict" in state_dict: state_dict = state_dict["state_dict"]

        # Surgical WTE fix (only for the original ckpt, specialized is already 94)
        wte_key = "model.transformer.wte.weight"
        if wte_key in state_dict and state_dict[wte_key].shape[1] == 92:
            new_wte = torch.zeros((144, 94), device=DEVICE)
            new_wte[:, :92] = state_dict[wte_key]
            state_dict[wte_key] = new_wte
        
        # Load (Specialized might be raw model state, so we handle both naming styles)
        try:
            lightning_module.load_state_dict(state_dict, strict=False)
        except:
            lightning_module.model.load_state_dict(state_dict, strict=False)
            
    else:
        print("Initializing Untrained (Random) Model...")
        for p in lightning_module.parameters():
            if p.dim() > 1: torch.nn.init.xavier_uniform_(p)

    return estimator.create_predictor(estimator.create_transformation(), lightning_module)

def run_final_audit():
    df = pd.read_csv(DATA_FILE)
    # Use 100 flows to get a meaningful F1 score
    unique_flows = df['flow_key_id'].unique()[:100]
    
    dataset_list = []
    for flow_id in unique_flows:
        group = df[df['flow_key_id'] == flow_id].sort_values('timestamp')
        dataset_list.append({
            "start": pd.Period(group.timestamp.iloc[0], freq="5min"),
            "target": group.traffic_volume_Tbits.values.astype(np.float32),
            "item_id": str(flow_id),
            "label": group.ground_truth.values[-1]
        })
    dataset = ListDataset(dataset_list, freq="5min")
    
    audit_results = {}
    for mode in ["foundation", "untrained", "specialized"]:
        predictor = get_aligned_predictor(mode)
        forecast_it, ts_it = make_evaluation_predictions(dataset=dataset, predictor=predictor, num_samples=100)
        
        forecasts = list(tqdm(forecast_it, total=len(dataset), desc=f"Audit: {mode}"))
        tss = list(ts_it)
        
        z_scores, labels = [], []
        for i in range(len(forecasts)):
            p_mean, p_std = np.mean(forecasts[i].samples), np.std(forecasts[i].samples)
            actual = tss[i].values[-1]
            z = np.abs(actual - p_mean) / (p_std + 1e-6)
            z_scores.append(z)
            labels.append(dataset_list[i]["label"])
            
        p, r, t = precision_recall_curve(labels, z_scores)
        f1 = 2 * (p * r) / (p + r + 1e-8)
        audit_results[mode] = {"Max F1": np.max(f1), "Mean Z": np.mean(z_scores), "Max Z": np.max(z_scores)}

    print("\n" + "="*60)
    print("FINAL ARCHITECTURAL AUDIT")
    print("="*60)
    print(pd.DataFrame(audit_results).T)

if __name__ == "__main__":
    run_final_audit()
