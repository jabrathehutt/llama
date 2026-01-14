import sys, types, torch, numpy as np, pandas as pd, os
from lag_llama.gluon.estimator import LagLlamaLightningModule
from gluonts.torch.distributions import StudentTOutput

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METRICS_CSV = "network_metrics_data.csv"
MODEL_PATH = "specialized_v11_supervised.pt"
CONTEXT_LEN = 256
THRESHOLD = 4.9012  # Your best found threshold

def run_production_detector():
    df = pd.read_csv(METRICS_CSV)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Setup Lags
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
    
    # Partial weight loading logic for stability
    sd = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    clean_sd = {k.replace("model.", ""): v for k, v in sd.items()}
    curr = module.model.state_dict()
    curr.update({k: v for k, v in clean_sd.items() if k in curr and v.size() == curr[k].size()})
    module.model.load_state_dict(curr)
    model = module.model.to(DEVICE).eval()

    print(f"Model ready. Monitoring {METRICS_CSV} with Z-Threshold {THRESHOLD}...")

    # Example: Run on the last flow in the file
    flow_id = df['flow_key_id'].iloc[-1]
    flow_data = df[df['flow_key_id'] == flow_id].sort_values('timestamp').tail(CONTEXT_LEN + 1)
    
    if len(flow_data) <= CONTEXT_LEN:
        print("Not enough data for inference.")
        return

    x_raw = flow_data.iloc[:CONTEXT_LEN]['traffic_volume_Tbits'].values
    target = flow_data.iloc[CONTEXT_LEN]['traffic_volume_Tbits']
    
    x = torch.tensor(x_raw).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        scale_f = x.mean(dim=1, keepdim=True) + 1e-5
        distr_args, _, _ = model(past_target=x/scale_f, past_observed_values=torch.ones_like(x).to(DEVICE))
        p_loc = (distr_args[0][0, -1] * scale_f.squeeze()).item()
        p_scale = (torch.exp(distr_args[1][0, -1]) * scale_f.squeeze()).item()
        
        z = abs(p_loc - target) / (p_scale + 1e-10)
        is_anom = z > THRESHOLD

        print(f"\nAnalysis for Flow {flow_id}:")
        print(f"Observed: {target:.4f} Tbits | Predicted: {p_loc:.4f} Tbits")
        print(f"Z-Score:  {z:.4f} {'[!!! ANOMALY DETECTED !!!]' if is_anom else '[Normal]'}")

if __name__ == "__main__":
    run_production_detector()
