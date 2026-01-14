import torch, numpy as np, pandas as pd, os
import matplotlib.pyplot as plt
from lag_llama.gluon.estimator import LagLlamaLightningModule
from gluonts.torch.distributions import StudentTOutput

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METRICS_CSV = "realistic_traffic_data.csv" # Using your saved CSV name
MODEL_PATH = "specialized_v11_supervised.pt"
CONTEXT_LEN = 256
THRESHOLD = 10.1020 # Using your new optimal threshold

def visualize_case(df, index, title, filename, p_loc, p_scale, z_score):
    """Plots the context leading up to the detection/miss."""
    plt.figure(figsize=(10, 5))
    
    # Get window
    context = df.iloc[index-CONTEXT_LEN:index]
    target_val = df.iloc[index]['traffic_volume_Tbits']
    is_true_anomaly = df.iloc[index]['is_anomaly']
    
    # Plot Context
    plt.plot(range(CONTEXT_LEN), context['traffic_volume_Tbits'], label='Context (Input)', color='blue', alpha=0.6)
    
    # Plot Target and Prediction
    plt.scatter(CONTEXT_LEN, target_val, color='red' if is_true_anomaly else 'green', 
                label=f'Actual Value ({"Anomaly" if is_true_anomaly else "Normal"})', s=100, zorder=5)
    
    # Visualize the "Normal Range" predicted by Lag-Llama (Student-T)
    # 3 standard deviations is roughly where a Z-score of 10 would sit relative to scale
    plt.errorbar(CONTEXT_LEN, p_loc, yerr=p_scale * THRESHOLD, fmt='o', color='black', 
                 label=f'Predicted Range (Z={THRESHOLD})', capsize=5)

    plt.title(f"{title}\nActual: {target_val:.2f} | Predicted: {p_loc:.2f} | Z-Score: {z_score:.2f}")
    plt.ylabel("Tbits")
    plt.xlabel("Time Steps")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")

def run_visual_detection():
    df = pd.read_csv(METRICS_CSV)
    
    # Load Model (Strictly univariate Lag-Llama)
    raw_lags = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216, 228, 240, 252, 264, 276, 288, 300, 312, 324, 336, 348, 360, 372, 384, 396, 408, 420, 432, 444, 456, 468, 480, 492, 504, 516, 528, 540, 552, 564, 576]
    lags_seq = [l for l in raw_lags if l < CONTEXT_LEN]

    module = LagLlamaLightningModule(
        context_length=CONTEXT_LEN, prediction_length=1,
        model_kwargs={"context_length": CONTEXT_LEN, "max_context_length": 1024, "input_size": 1,
                      "distr_output": StudentTOutput(), "n_layer": 8, "n_head": 9,
                      "n_embd_per_head": 16, "lags_seq": lags_seq, "scaling": "mean", "time_feat": False}
    )

    sd = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    clean_sd = {k.replace("model.", ""): v for k, v in sd.items()}
    module.model.load_state_dict(clean_sd, strict=False)
    model = module.model.to(DEVICE).eval()

    detected_index = None
    missed_index = None

    print("Searching for specific visualization cases...")
    # We iterate through the dataframe to find one 'Success' and one 'Failure'
    for i in range(CONTEXT_LEN, len(df)):
        if detected_index and missed_index: break
            
        x_raw = df.iloc[i-CONTEXT_LEN:i]['traffic_volume_Tbits'].values
        target = df.iloc[i]['traffic_volume_Tbits']
        is_anom_true = df.iloc[i]['is_anomaly']

        x = torch.tensor(x_raw).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            scale_f = x.mean(dim=1, keepdim=True) + 1e-5
            distr_args, _, _ = model(past_target=x/scale_f, past_observed_values=torch.ones_like(x).to(DEVICE))
            p_loc = (distr_args[0][0, -1] * scale_f.squeeze()).item()
            p_scale = (torch.exp(distr_args[1][0, -1]) * scale_f.squeeze()).item()
            z = abs(p_loc - target) / (p_scale + 1e-10)
            is_anom_pred = z > THRESHOLD

            # Case 1: True Positive (Detected)
            if is_anom_true and is_anom_pred and not detected_index:
                detected_index = i
                visualize_case(df, i, "Case: Detected Anomaly (True Positive)", "detected_anomaly.png", p_loc, p_scale, z)

            # Case 2: False Negative (Missed)
            if is_anom_true and not is_anom_pred and not missed_index:
                missed_index = i
                visualize_case(df, i, "Case: Missed Anomaly (False Negative)", "missed_anomaly.png", p_loc, p_scale, z)

if __name__ == "__main__":
    run_visual_detection()
