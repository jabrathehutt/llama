import pandas as pd
import numpy as np
import torch
import torch.serialization
from gluonts.dataset.common import ListDataset
from itertools import islice


import sys
from types import ModuleType
from lag_llama.gluon.estimator import LagLlamaEstimator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create dummy module hierarchy
def create_dummy_module(module_path):
    """
    Create a dummy module hierarchy for the given path.
    Returns the leaf module.
    """
    parts = module_path.split('.')
    current = ''
    parent = None

    for part in parts:
        current = current + '.' + part if current else part
        if current not in sys.modules:
            module = ModuleType(current)
            sys.modules[current] = module
            if parent:
                setattr(sys.modules[parent], part, module)
        parent = current

    return sys.modules[module_path]

# Create the dummy gluonts module hierarchy
gluonts_module = create_dummy_module('gluonts.torch.modules.loss')

# Create dummy classes for the specific loss functions
class DistributionLoss:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return 0.0

    def __getattr__(self, name):
        return lambda *args, **kwargs: None

class NegativeLogLikelihood:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return 0.0

    def __getattr__(self, name):
        return lambda *args, **kwargs: None

# Add the specific classes to the module
gluonts_module.DistributionLoss = DistributionLoss
gluonts_module.NegativeLogLikelihood = NegativeLogLikelihood




# --- Configuration ---
TARGET_COLUMN = 'traffic_volume_Tbits'
SERIES_ID_COLUMN = 'flow_key_id'
PREDICTION_LENGTH = 1
CONTEXT_LENGTH = 64 # The amount of history used to predict the next point
LOWER_QUANTILE = 0.005 # 99% Prediction Interval (0.5% tail)
UPPER_QUANTILE = 0.995
NUM_SAMPLES = 20
FREQ = '5min'
CHECKPOINT_PATH = "/root/lag-llama/lag-llama.ckpt"

# --- PyTorch Security Fix (Required to load the old checkpoint) ---
#torch.serialization.add_safe_globals([StudentTOutput, NegativeLogLikelihood])
# ---------------------------

# --- Step 1: Data Loading and Model Initialization ---

df = pd.read_csv("/root/lag-llama/network_metrics_data.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

print("Loading model configuration from checkpoint...")
# Load configuration arguments from the checkpoint file (Fixes size mismatch)
ckpt = torch.load(CHECKPOINT_PATH, weights_only=False)
estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

# Instantiate Estimator with DYNAMIC arguments from checkpoint (Fixes size mismatch and TypeErrors)
estimator = LagLlamaEstimator(
    ckpt_path=CHECKPOINT_PATH,
    prediction_length=PREDICTION_LENGTH,
    context_length=CONTEXT_LENGTH,
    device=DEVICE,
    # Dynamic model arguments from checkpoint
    input_size=estimator_args["input_size"],
    n_layer=estimator_args["n_layer"],
    n_embd_per_head=estimator_args["n_embd_per_head"],
    n_head=estimator_args["n_head"],
    scaling=estimator_args["scaling"],
    time_feat=estimator_args["time_feat"],

    # All non-essential GluonTS arguments are omitted to prevent TypeErrors
)

# Generate the predictor using the correct demo pattern
lightning_module = estimator.create_lightning_module()
transformation = estimator.create_transformation()

predictor = estimator.create_predictor(
    transformation=transformation,
    module=lightning_module
)

# --- Step 2: Sliding Window Anomaly Detection ---

all_anomaly_results = []
unique_series_ids = df[SERIES_ID_COLUMN].unique()

print(f"\nStarting full historical anomaly analysis across {len(unique_series_ids)} flows...")

for series_id in unique_series_ids:
    series_df = df[df[SERIES_ID_COLUMN] == series_id].set_index('timestamp')
    series_target = series_df[TARGET_COLUMN].values.astype(np.float32)
    series_start_time = series_df.index[0]

    # We start the sliding window after the context length is satisfied
    # The anomaly check is performed on the last point of the window.
    for end_index in range(CONTEXT_LENGTH, len(series_target)):

        # The data window contains the context + the point to check (length = CONTEXT_LENGTH + 1)
        window_target = series_target[end_index - CONTEXT_LENGTH : end_index + 1]

        # The time and value of the point being checked (the last point in the window)
        anomaly_time = series_df.index[end_index]
        actual_value = window_target[-1]

        # 1. Create a single-item ListDataset for the current window
        test_ds = ListDataset([{
            "start": anomaly_time - (CONTEXT_LENGTH * pd.to_timedelta(FREQ)),
            "target": window_target,
        }], freq=FREQ, one_dim_target=True)

        # 2. Predict the distribution for the next step (which is the last point in our window)
        # Note: We only generate 1 forecast, so we take the first item in the forecasts list
        forecasts_generator = predictor.predict(test_ds, num_samples=NUM_SAMPLES)
        forecast = next(forecasts_generator)

        samples = forecast.samples[:, 0]

        # 3. Anomaly Check
        lower_bound = np.quantile(samples, LOWER_QUANTILE)
        upper_bound = np.quantile(samples, UPPER_QUANTILE)
        is_anomaly = (actual_value < lower_bound) or (actual_value > upper_bound)

        if is_anomaly:
            all_anomaly_results.append({
                "series_id": series_id,
                "timestamp": anomaly_time,
                "actual_value": actual_value,
                f"lower_bound_{LOWER_QUANTILE*100:.1f}": lower_bound,
                f"upper_bound_{UPPER_QUANTILE*100:.1f}": upper_bound,
                "is_anomaly": is_anomaly
            })

# --- Step 3: Output Results ---

anomaly_df = pd.DataFrame(all_anomaly_results)

print(f"\nAnalysis Complete. Total points checked: {len(series_target) * len(unique_series_ids)}")

if anomaly_df.empty:
    print("NO ANOMALIES DETECTED in the historical data for any flow.")
else:
    print(f"ANOMALIES DETECTED: {len(anomaly_df)} total anomalies found.")

    # --- Save output to CSV ---
    output_file = "lag_llama_full_history_anomalies.csv"
    anomaly_df.to_csv(output_file, index=False)

    print("\n--- Detailed Anomaly List ---")
    print(anomaly_df.to_string())
    print(f"\nFull results saved to {output_file}")
