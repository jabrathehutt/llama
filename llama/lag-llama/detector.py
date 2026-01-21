import torch, numpy as np, pandas as pd, os
from lag_llama.gluon.estimator import LagLlamaLightningModule
from gluonts.torch.distributions import StudentTOutput
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METRICS_CSV = "/root/network_forecast_anomalous.csv"
MODEL_PATH = "specialized_v11_supervised.pt"
CONTEXT_LEN = 256
BATCH_SIZE = 128 

def run_threshold_optimization():
    if not os.path.exists(METRICS_CSV):
        print("Data not found."); return

    df = pd.read_csv(METRICS_CSV)
    
    # 1. Load Model
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

    sd = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    clean_sd = {k.replace("model.", ""): v for k, v in sd.items()}
    module.model.load_state_dict(clean_sd, strict=False)
    model = module.model.to(DEVICE).eval()

    # 2. Collect Data Windows
    windows_x, targets_y, y_true = [], [], []
    for _, flow_group in df.groupby('flow_key_id'):
        v = flow_group['traffic_volume_Tbits'].values
        labels = flow_group['is_anomaly'].values
        for i in range(CONTEXT_LEN, len(v)):
            windows_x.append(v[i-CONTEXT_LEN:i])
            targets_y.append(v[i])
            y_true.append(1 if labels[i] else 0)

    # 3. Generate Z-Scores
    num_windows = len(windows_x)
    all_z_scores = []
    
    print(f"Calculating Z-Scores for {num_windows} points...")
    for i in tqdm(range(0, num_windows, BATCH_SIZE)):
        batch_end = min(i + BATCH_SIZE, num_windows)
        x_batch = torch.tensor(np.array(windows_x[i:batch_end])).float().to(DEVICE)
        y_batch = np.array(targets_y[i:batch_end])
        
        with torch.no_grad():
            scale_f = x_batch.mean(dim=1, keepdim=True) + 1e-5
            distr_args, _, _ = model(past_target=x_batch/scale_f, past_observed_values=torch.ones_like(x_batch).to(DEVICE))
            p_loc = (distr_args[0][:, -1] * scale_f.squeeze(-1)).cpu().numpy()
            p_scale = (torch.exp(distr_args[1][:, -1]) * scale_f.squeeze(-1)).cpu().numpy()
            
            z = np.abs(p_loc - y_batch) / (p_scale + 1e-10)
            all_z_scores.extend(z)

    # 4. Search for Best Threshold
    thresholds = np.linspace(3.0, 15.0, 50)
    best_f1, best_t = 0, 0
    results = []

    print("Searching for optimal threshold...")
    for t in thresholds:
        y_pred = (np.array(all_z_scores) > t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1, best_t = f1, t
        results.append((t, p, r, f1))

    # 5. Output results
    res_df = pd.DataFrame(results, columns=['Threshold', 'Precision', 'Recall', 'F1'])
    print("\n" + "="*45)
    print(f"BEST THRESHOLD FOUND: {best_t:.4f}")
    print(f"MAX F1-SCORE:        {best_f1:.4f}")
    print("="*45)
    print(res_df.sort_values('F1', ascending=False).head(10).to_string(index=False))

if __name__ == "__main__":
    run_threshold_optimization()
