import torch, numpy as np, pandas as pd, os
from lag_llama.gluon.estimator import LagLlamaLightningModule
from gluonts.torch.distributions import StudentTOutput
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METRICS_CSV = "/root/network_forecast_anomalous.csv"
MODEL_PATH = "specialized_v11_supervised.pt"

# ⚠️ MATCHING ARCHITECTURE (from finetune2.py)
CONTEXT_LEN = 96
N_LAYER = 1
N_HEAD = 8
LAGS_SEQ = list(range(1, 85)) # 84 lags to result in 86 total features
BATCH_SIZE = 128

def run_final_evaluation():
    if not os.path.exists(METRICS_CSV):
        print("Data not found."); return

    df = pd.read_csv(METRICS_CSV)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 1. Initialize Module and Model Structure
    module = LagLlamaLightningModule(
        context_length=CONTEXT_LEN, 
        prediction_length=1,
        model_kwargs={
            "context_length": CONTEXT_LEN, 
            "max_context_length": 1024, 
            "input_size": 1,
            "distr_output": StudentTOutput(), 
            "n_layer": N_LAYER, 
            "n_head": N_HEAD,
            "n_embd_per_head": 16, 
            "lags_seq": LAGS_SEQ, 
            "scaling": "mean", 
            "time_feat": False,
        }
    )

    # 2. Load the supervised weights
    print(f"Loading supervised weights from {MODEL_PATH}...")
    sd = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    # Clean keys if saved with prefix
    clean_sd = {k.replace("model.", ""): v for k, v in sd.items()}
    module.model.load_state_dict(clean_sd, strict=True)
    model = module.model.to(DEVICE).eval()

    # 3. Collect Data Windows (Filtering for Backbone Flow 0)
    windows_x, targets_y, y_true = [], [], []
    flow0 = df[df['flow_key_id'] == '0'].sort_values('timestamp')
    
    v = flow0['traffic_volume'].values.astype('float32')
    labels = flow0['is_anomaly'].values
    
    for i in range(CONTEXT_LEN, len(v)):
        windows_x.append(v[i-CONTEXT_LEN:i])
        targets_y.append(v[i])
        y_true.append(1 if labels[i] else 0)

    # 4. Generate Z-Scores
    num_windows = len(windows_x)
    all_z_scores = []

    print(f"Calculating Z-Scores for {num_windows} intervals...")
    for i in tqdm(range(0, num_windows, BATCH_SIZE)):
        batch_end = min(i + BATCH_SIZE, num_windows)
        x_batch = torch.tensor(np.array(windows_x[i:batch_end])).to(DEVICE)
        y_batch = np.array(targets_y[i:batch_end])

        with torch.no_grad():
            scale_f = x_batch.mean(dim=1, keepdim=True) + 1e-5
            distr_args, _, _ = model(
                past_target=x_batch/scale_f, 
                past_observed_values=torch.ones_like(x_batch).to(DEVICE)
            )
            
            # Prediction mean and scale (standard deviation)
            p_loc = (distr_args[0][:, -1] * scale_f.squeeze(-1)).cpu().numpy()
            p_scale = (torch.exp(distr_args[1][:, -1]) * scale_f.squeeze(-1)).cpu().numpy()

            # Z-Score represents the distance from the expected backbone volume
            z = np.abs(p_loc - y_batch) / (p_scale + 1e-10)
            all_z_scores.extend(z)

    # 5. Search for Best Threshold (2.0 to 100.0)
    thresholds = np.linspace(2.0, 100.0, 200)
    best_f1, best_t = 0, 0
    results = []

    print("Searching for optimal detection threshold...")
    for t in thresholds:
        y_pred = (np.array(all_z_scores) > t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)

        if f1 > best_f1:
            best_f1, best_t = f1, t
        results.append((t, p, r, f1))

    # 6. Final Report
    res_df = pd.DataFrame(results, columns=['Threshold', 'Precision', 'Recall', 'F1'])
    print("\n" + "="*45)
    print(f"BEST THRESHOLD: {best_t:.4f}")
    print(f"FINAL F1-SCORE: {best_f1:.4f}")
    print("="*45)
    print(res_df.sort_values('F1', ascending=False).head(5).to_string(index=False))

if __name__ == "__main__":
    run_final_evaluation()
