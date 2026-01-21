import sys, types, torch, numpy as np, pandas as pd, random, os
from tqdm import tqdm
from lag_llama.gluon.estimator import LagLlamaLightningModule
from gluonts.torch.distributions import StudentTOutput
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# --- 1. ENVIRONMENT SHIMS ---
def mock_strtobool(val):
    val = str(val).lower()
    return 1 if val in ('y', 'yes', 't', 'true', 'on', '1') else 0
d_util = types.ModuleType("distutils.util"); d_util.strtobool = mock_strtobool
sys.modules["distutils"] = types.ModuleType("distutils"); sys.modules["distutils.util"] = d_util

# --- 2. CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METRICS_CSV = "network_metrics_data.csv"
MODEL_PATH = "specialized_v11_supervised.pt"
CONTEXT_LEN = 256
THRESHOLD = 4.9012  # Your optimized threshold

def get_test_flows(csv_path):
    df = pd.read_csv(csv_path)
    unique_flows = sorted(df['flow_key_id'].unique())
    random.seed(42)
    random.shuffle(unique_flows)
    test_ids = set(unique_flows[int(len(unique_flows) * 0.8):])
    return df[df['flow_key_id'].isin(test_ids)].copy()

def run_balanced_audit():
    # Load and identify columns
    df_test = get_test_flows(METRICS_CSV)
    label_col = 'is_anomaly' if 'is_anomaly' in df_test.columns else 'ground_truth'
    time_col = 'timestamp' if 'timestamp' in df_test.columns else 'timestamp'
    df_test[time_col] = pd.to_datetime(df_test[time_col])

    # Model Setup
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
    
    # Weight adaptation logic
    sd = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    clean_sd = {k.replace("model.", ""): v for k, v in sd.items()}
    curr_dict = module.model.state_dict()
    loadable = {k: v for k, v in clean_sd.items() if k in curr_dict and v.size() == curr_dict[k].size()}
    curr_dict.update(loadable)
    module.model.load_state_dict(curr_dict)
    model = module.model.to(DEVICE).eval()

    y_true, y_pred, z_scores = [], [], []

    # Stratified Sampling: Ensure we have exactly 100 normal and 100 anomaly points
    for label in [0, 1]:
        eligible = df_test[df_test[label_col] == label].index
        n_samples = min(100, len(eligible))
        sampled_indices = np.random.choice(eligible, n_samples, replace=False)

        for idx in tqdm(sampled_indices, desc=f"Testing Label={label}"):
            row = df_test.loc[idx]
            flow_id = row['flow_key_id']
            flow_full = df_test[df_test['flow_key_id'] == flow_id].sort_values(time_col)
            pos = flow_full.index.get_loc(idx)
            
            if pos < CONTEXT_LEN: continue
                
            x_raw = flow_full.iloc[pos-CONTEXT_LEN:pos]['traffic_volume_Tbits'].values
            target = row['traffic_volume_Tbits']
            
            x_tensor = torch.tensor(x_raw).float().unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                scale_f = x_tensor.mean(dim=1, keepdim=True) + 1e-5
                distr_args, _, _ = model(past_target=x_tensor/scale_f, past_observed_values=torch.ones_like(x_tensor).to(DEVICE))
                
                p_loc = (distr_args[0][0, -1] * scale_f.squeeze()).item()
                p_scale = (torch.exp(distr_args[1][0, -1]) * scale_f.squeeze()).item()
                
                z = abs(p_loc - target) / (p_scale + 1e-10)
                
                y_true.append(int(row[label_col]))
                y_pred.append(1 if z > THRESHOLD else 0)
                z_scores.append(z)

    # Output Metrics
    print("\n" + "="*50)
    print(f"BALANCED DETECTION REPORT (Z > {THRESHOLD})")
    print("="*50)
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred):.4f}")
    print("-" * 50)
    print("Detailed Stats:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))
    print("="*50)

if __name__ == "__main__":
    run_balanced_audit()
