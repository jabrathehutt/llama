import torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from lag_llama.gluon.estimator import LagLlamaLightningModule
from gluonts.torch.distributions import StudentTOutput

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METRICS_CSV = "/root/network_forecast_anomalous.csv"
MODEL_PATH = "specialized_v11_supervised.pt"
CONTEXT_LEN = 96
LAGS_SEQ = list(range(1, 85))

def visualize_results():
    df = pd.read_csv(METRICS_CSV)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    flow0 = df[df['flow_key_id'] == '0'].sort_values('timestamp')
    
    # Load Model
    module = LagLlamaLightningModule(
        context_length=CONTEXT_LEN, prediction_length=1,
        model_kwargs={"context_length": CONTEXT_LEN, "max_context_length": 1024, "input_size": 1,
                     "distr_output": StudentTOutput(), "n_layer": 1, "n_head": 8,
                     "n_embd_per_head": 16, "lags_seq": LAGS_SEQ, "scaling": "mean", "time_feat": False}
    )
    sd = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    module.model.load_state_dict({k.replace("model.", ""): v for k, v in sd.items()}, strict=True)
    model = module.model.to(DEVICE).eval()

    v = flow0['traffic_volume'].values.astype('float32')
    labels = flow0['is_anomaly'].values
    timestamps = flow0['timestamp'].values[CONTEXT_LEN:]
    
    z_scores = []
    for i in range(CONTEXT_LEN, len(v)):
        x = torch.tensor(v[i-CONTEXT_LEN:i]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            scale_f = x.mean() + 1e-5
            distr_args, _, _ = model(past_target=x/scale_f, past_observed_values=torch.ones_like(x))
            p_loc = (distr_args[0][0, -1] * scale_f).cpu().item()
            p_scale = (torch.exp(distr_args[1][0, -1]) * scale_f).cpu().item()
            z_scores.append(abs(p_loc - v[i]) / (p_scale + 1e-10))

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    ax1.plot(timestamps, v[CONTEXT_LEN:], label='Traffic Volume', color='blue')
    ax1.scatter(timestamps[labels[CONTEXT_LEN:]==1], v[CONTEXT_LEN:][labels[CONTEXT_LEN:]==1], color='red', label='Ground Truth Anomaly')
    ax1.set_title("Network Traffic & Anomalies")
    ax1.legend()

    ax2.plot(timestamps, z_scores, label='Z-Score (Model Sensitivity)', color='orange')
    ax2.axhline(y=2.5, color='green', linestyle='--', label='Current Threshold')
    ax2.set_title("Model Z-Scores")
    ax2.legend()
    
    plt.savefig("anomaly_diagnostic.png")
    print("Diagnostic plot saved as anomaly_diagnostic.png")

if __name__ == "__main__":
    visualize_results()
