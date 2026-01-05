import pandas as pd
import numpy as np
import torch
import warnings
import sys
from types import ModuleType
from tqdm import tqdm
from gluonts.dataset.common import ListDataset
from lag_llama.gluon.estimator import LagLlamaEstimator
from sklearn.metrics import precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")

# --- DUMMY MODULE BRIDGE ---
def create_dummy_module(module_path):
    parts = module_path.split('.')
    current = ''
    parent = None
    for part in parts:
        current = current + '.' + part if current else part
        if current not in sys.modules:
            module = ModuleType(current)
            sys.modules[current] = module
            if parent: setattr(sys.modules[parent], part, module)
        parent = current
    return sys.modules[module_path]

create_dummy_module('gluonts.torch.modules.loss')
class DistributionLoss:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, *args, **kwargs): return 0.0
class NegativeLogLikelihood:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, *args, **kwargs): return 0.0
sys.modules['gluonts.torch.modules.loss'].DistributionLoss = DistributionLoss
sys.modules['gluonts.torch.modules.loss'].NegativeLogLikelihood = NegativeLogLikelihood

from gluonts.torch.distributions.studentT import StudentTOutput
torch.serialization.add_safe_globals([StudentTOutput, NegativeLogLikelihood, DistributionLoss])

# --- Configuration ---
DATA_FILE = "/root/lag-llama/network_metrics_data.csv"
FINETUNED_CKPT = "/root/lag-llama/real_world_network_llama.pt"
CONTEXT_LENGTH = 128
BUFFER_STEPS = 6 

def run_feature_detector():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(DATA_FILE)
    
    anom_indices = df.index[df['is_anomaly'] == 1].tolist()
    idx = anom_indices[0]
    sample_df = df.iloc[max(0, idx - 300) : min(len(df), idx + 300)].copy()

    # Load Weights
    ckpt = torch.load(FINETUNED_CKPT, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    args = ckpt.get('hyper_parameters', {}).get('model_kwargs', {"input_size": 1, "n_layer": 8, "n_head": 9, "n_embd_per_head": 16, "scaling": "mean", "time_feat": 0})

    estimator = LagLlamaEstimator(ckpt_path=None, prediction_length=1, context_length=CONTEXT_LENGTH, **args)
    module = estimator.create_lightning_module()
    
    # Grafting
    model_state = module.state_dict()
    if "model.transformer.wte.weight" in state_dict:
        s_w = state_dict["model.transformer.wte.weight"]
        m_w = model_state["model.transformer.wte.weight"]
        if s_w.shape != m_w.shape:
            new_w = torch.zeros_like(m_w)
            new_w[:min(s_w.shape[0], m_w.shape[0]), :min(s_w.shape[1], m_w.shape[1])] = s_w[:min(s_w.shape[0], m_w.shape[0]), :min(s_w.shape[1], m_w.shape[1])]
            state_dict["model.transformer.wte.weight"] = new_w

    module.load_state_dict(state_dict, strict=False)
    predictor = estimator.create_predictor(estimator.create_transformation(), module.to(device).eval())

    # 1. Compute Raw MAD-Scores
    print("Extracting Temporal Surprise Features...")
    scores = []
    indices = []
    for i in tqdm(range(len(sample_df))):
        orig_idx = sample_df.index[i]
        if orig_idx < CONTEXT_LENGTH: continue
        
        history = np.log1p(df.iloc[orig_idx - CONTEXT_LENGTH : orig_idx]['traffic_volume_Tbits'].values)
        actual = np.log1p(df.iloc[orig_idx]['traffic_volume_Tbits'])
        
        ds = ListDataset([{"start": pd.to_datetime(sample_df['timestamp'].iloc[i]), "target": history}], freq='5min')
        forecast = list(predictor.predict(ds, num_samples=100))[0]
        
        samples = forecast.samples.flatten()
        median = np.median(samples)
        mad = np.median(np.abs(samples - median)) + 1e-6
        scores.append(np.abs(actual - median) / mad)
        indices.append(orig_idx)

    # 2. FEATURE ENGINEERING: Relative Volatility
    # We compare current surprise to the rolling median surprise
    s_series = pd.Series(scores)
    rolling_baseline = s_series.rolling(window=20).median().fillna(method='bfill')
    relative_surprise = s_series / (rolling_baseline + 0.1)

    # 3. Final Sweep
    print(f"\n{'Rel-Surprise Thresh':<20} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 70)
    
    true_anom_indices = sample_df.index[sample_df['is_anomaly'] == 1].tolist()
    
    for thresh in [1.5, 2.0, 3.0, 5.0, 10.0]:
        detections = [indices[i] for i, val in enumerate(relative_surprise) if val > thresh]
        
        y_true, y_pred = [], []
        for current_idx in indices:
            is_truth = 1 if current_idx in true_anom_indices else 0
            nearby = any(abs(current_idx - d_idx) <= BUFFER_STEPS for d_idx in detections)
            y_true.append(is_truth)
            y_pred.append(1 if nearby else 0)

        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"{thresh:<20.1f} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f}")

if __name__ == "__main__":
    run_feature_detector()
