# --- 1. THE SHIMS ---
import sys, types, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from lag_llama.gluon.estimator import LagLlamaEstimator
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import make_evaluation_predictions

m = types.ModuleType("gluonts.torch.modules.loss"); sys.modules["gluonts.torch.modules.loss"] = m
class MockLoss: pass
m.NegativeLogLikelihood = MockLoss; m.DistributionLoss = MockLoss
d = types.ModuleType("distutils"); sys.modules["distutils"] = d
u = types.ModuleType("distutils.util"); sys.modules["distutils.util"] = u
setattr(d, "util", u); u.strtobool = lambda v: 1 if str(v).lower() in ("y", "yes", "t", "true", "1") else 0

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "lag-llama.ckpt"
SPECIALIZED_PATH = "specialized_v6_refined.pt"

DATA_FILE = "network_metrics_data.csv"

def get_aligned_predictor(mode="foundation"):
    estimator = LagLlamaEstimator(
        ckpt_path=None, prediction_length=1, context_length=32,
        lags_seq=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216, 228, 240, 252, 264, 276, 288, 300, 312, 324, 336, 348, 360, 372, 384, 396, 408, 420, 432, 444, 456, 468, 480, 492, 504, 516, 528, 540, 552, 564, 576],
        n_layer=8, n_head=9, n_embd_per_head=16, scaling="mean", trainer_kwargs=dict(accelerator="gpu", devices=1)
    )
    lightning_module = estimator.create_lightning_module()
    path = CKPT_PATH if mode == "foundation" else SPECIALIZED_PATH
    state_dict = torch.load(path, map_location=DEVICE, weights_only=False)
    if "state_dict" in state_dict: state_dict = state_dict["state_dict"]
    
    # 92 -> 94 adjustment for original ckpt
    wte_key = "model.transformer.wte.weight"
    if wte_key in state_dict and state_dict[wte_key].shape[1] == 92:
        new_wte = torch.zeros((144, 94), device=DEVICE)
        new_wte[:, :92] = state_dict[wte_key]
        state_dict[wte_key] = new_wte
    
    lightning_module.load_state_dict(state_dict, strict=False)
    return estimator.create_predictor(estimator.create_transformation(), lightning_module)

def visualize_anomaly():
    df = pd.read_csv(DATA_FILE)
    # Find a flow that actually has an anomaly (ground_truth == 1)
    anomaly_flows = df[df['ground_truth'] == 1]['flow_key_id'].unique()
    if len(anomaly_flows) == 0:
        print("No anomalies found in dataset to visualize!")
        return
    
    flow_id = anomaly_flows[0]
    group = df[df['flow_key_id'] == flow_id].sort_values('timestamp')
    
    # Dataset for prediction
    test_ds = ListDataset([{"start": pd.Period(group.timestamp.iloc[0], freq="5min"), 
                            "target": group.traffic_volume_Tbits.values.astype(np.float32)}], freq="5min")
    
    plt.figure(figsize=(12, 6))
    plt.plot(group.traffic_volume_Tbits.values[-50:], label="Actual Traffic", color='black', linewidth=2)
    
    for mode, color in [("foundation", "blue"), ("specialized", "red")]:
        predictor = get_aligned_predictor(mode)
        forecast_it, ts_it = make_evaluation_predictions(dataset=test_ds, predictor=predictor, num_samples=100)
        forecast = list(forecast_it)[0]
        
        low = np.quantile(forecast.samples, 0.05)
        high = np.quantile(forecast.samples, 0.95)
        mean = np.mean(forecast.samples)
        
        plt.plot([len(group)-1], [mean], marker='o', color=color, label=f"{mode} Prediction")
        plt.fill_between([len(group)-1], [low], [high], color=color, alpha=0.2)

    # Mark the ground truth anomaly
    if group['ground_truth'].iloc[-1] == 1:
        plt.axvline(x=49, color='orange', linestyle='--', label="TRUE ANOMALY")

    plt.title(f"Foundation vs Specialized: Flow {flow_id}")
    plt.legend()
    plt.savefig("comparison_plot.png")
    print("Plot saved to comparison_plot.png")

if __name__ == "__main__":
    visualize_anomaly()
