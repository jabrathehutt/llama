import torch, numpy as np, pandas as pd, os
import matplotlib.pyplot as plt
from lag_llama.gluon.estimator import LagLlamaLightningModule
from gluonts.torch.distributions import StudentTOutput
from tqdm import tqdm

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_CSV = "/root/traffic-shifts/llama/lag-llama/trafpy_finetune_normal_data.csv"
TEST_CSV = "/root/traffic-shifts/trafpy/trafpy_master_univariate_data.csv"
MODEL_PATH = "specialized_v11_supervised.pt"
CONTEXT_LEN = 256  

def run_and_plot_anomaly():
    # 1. Load Stats for Thresholding
    df_train = pd.read_csv(TRAIN_CSV)
    GLOBAL_MEAN = np.mean(df_train['traffic_volume_Tbits'].values)
    GLOBAL_STD = np.std(df_train['traffic_volume_Tbits'].values)
    THRESHOLD = 4.5 * GLOBAL_STD

    # 2. Load Model
    module = LagLlamaLightningModule(
        context_length=CONTEXT_LEN, prediction_length=1,
        model_kwargs={
            "context_length": CONTEXT_LEN, "max_context_length": 1024, "input_size": 1,
            "distr_output": StudentTOutput(), "n_layer": 1, "n_head": 8,
            "n_embd_per_head": 16, "lags_seq": list(range(1, 85)), "scaling": "mean", "time_feat": False,
        }
    )
    sd = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    module.model.load_state_dict({k.replace("model.", ""): v for k, v in sd.items()}, strict=True)
    model = module.model.to(DEVICE).eval()

    # 3. Get Test Data
    df_test = pd.read_csv(TEST_CSV)
    test_flow = df_test['flow_key_id'].unique()[0]
    flow_df = df_test[df_test['flow_key_id'] == test_flow].sort_values('timestamp').reset_index(drop=True)
    
    # 4. Inference on the Anomaly Window
    # Centering around the actual anomaly start
    anomaly_start = flow_df[flow_df['is_anomaly'] == 1].index[0]
    plot_slice = flow_df.iloc[anomaly_start-50 : anomaly_start+150].copy()
    
    v = flow_df['traffic_volume_Tbits'].values.astype('float32')
    preds, actuals, times = [], [], []

    for i in tqdm(plot_slice.index):
        window = torch.tensor(v[i-CONTEXT_LEN:i]).unsqueeze(0).to(DEVICE).float()
        with torch.no_grad():
            scale_f = torch.tensor([[GLOBAL_MEAN + 1e-5]]).to(DEVICE).float()
            distr_args, _, _ = model(past_target=window/scale_f, past_observed_values=torch.ones_like(window).to(DEVICE).float())
            y_pred = (distr_args[0][:, -1] * scale_f.squeeze(-1)).cpu().item()
            preds.append(y_pred)
            actuals.append(v[i])
            times.append(flow_df.loc[i, 'timestamp'])

    # 5. Visualization
    residuals = np.abs(np.array(actuals) - np.array(preds))
    detections = (residuals > THRESHOLD).astype(int)
    
    plt.figure(figsize=(14, 8))
    
    # Top Plot: Traffic Volume
    plt.subplot(2, 1, 1)
    plt.plot(times, actuals, label='Actual Traffic', color='blue', linewidth=1.5)
    plt.plot(times, preds, label='Lag-Llama Forecast', color='orange', linestyle='--', alpha=0.8)
    plt.axvline(x=times[50], color='red', linestyle=':', label='Anomaly Start') # 50 is the relative start
    plt.title(f"Detected Anomaly: {test_flow}")
    plt.ylabel("Tbits")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Bottom Plot: Residuals (The Decision Maker)
    plt.subplot(2, 1, 2)
    plt.fill_between(times, residuals, color='red', alpha=0.3, label='Model Residual')
    plt.axhline(y=THRESHOLD, color='black', linestyle='--', label=f'Threshold ({THRESHOLD:.4f})')
    plt.ylabel("Absolute Error")
    plt.xlabel("Time")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("anomaly_detection_plot.png")
    print("Plot saved as anomaly_detection_plot.png")
    plt.show()

if __name__ == "__main__":
    run_and_plot_anomaly()
