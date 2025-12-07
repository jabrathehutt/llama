import pandas as pd
import numpy as np
import torch
import torch.serialization
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
import pytorch_lightning as pl
import os
import sys
from types import ModuleType
from lag_llama.gluon.estimator import LagLlamaEstimator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create dummy module hierarchy
def create_dummy_module(module_path):
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

gluonts_module.DistributionLoss = DistributionLoss
gluonts_module.NegativeLogLikelihood = NegativeLogLikelihood

# --- Configuration ---
TARGET_COLUMN = 'traffic_volume_Tbits'
SERIES_ID_COLUMN = 'flow_key_id'
PREDICTION_LENGTH = 1
CONTEXT_LENGTH = 96
FREQ = '5min'
CHECKPOINT_PATH = "/root/llama/lag-llama/lag-llama.ckpt"
FINETUNED_CHECKPOINT_PATH = "/root/llama/lag-llama/lag-llama-finetuned.ckpt"

# Fine-tuning configuration
FINETUNE_EPOCHS = 10
FINETUNE_BATCH_SIZE = 16
TRAIN_VAL_SPLIT = 0.8
MAX_SERIES = 10  # Limit number of series to process

# Use ground truth
HAS_GROUND_TRUTH = True
GROUND_TRUTH_COLUMN = 'is_anomaly'

def analyze_univariate_data(df):
    """Analyze the univariate dataset efficiently"""
    print("Analyzing univariate dataset...")
    
    # Get unique series and limit for processing
    unique_series_ids = df[SERIES_ID_COLUMN].unique()
    print(f"Total unique series: {len(unique_series_ids)}")
    
    if len(unique_series_ids) > MAX_SERIES:
        print(f"Limiting to first {MAX_SERIES} series for processing")
        unique_series_ids = unique_series_ids[:MAX_SERIES]
    
    for series_id in unique_series_ids:
        # Use more efficient filtering
        series_mask = df[SERIES_ID_COLUMN] == series_id
        series_data = df.loc[series_mask, TARGET_COLUMN].values
        
        print(f"\nSeries {series_id}:")
        print(f"  Length: {len(series_data)}")
        print(f"  Mean: {np.mean(series_data):.4f}")
        print(f"  Std: {np.std(series_data):.4f}")
        print(f"  Min: {np.min(series_data):.4f}")
        print(f"  Max: {np.max(series_data):.4f}")
        print(f"  Range: {np.ptp(series_data):.4f}")
        
        # Check ground truth distribution
        if GROUND_TRUTH_COLUMN in df.columns:
            anomaly_count = df.loc[series_mask, GROUND_TRUTH_COLUMN].sum()
            print(f"  Anomalies: {anomaly_count}/{len(series_data)} ({anomaly_count/len(series_data)*100:.2f}%)")
    
    return unique_series_ids

def prepare_univariate_data_efficient(df, unique_series_ids):
    """Prepare univariate data efficiently with limited series"""
    train_data = []
    test_data = []
    
    print(f"Preparing data for {len(unique_series_ids)} series...")
    
    for i, series_id in enumerate(unique_series_ids):
        if i % 10 == 0:  # Progress indicator
            print(f"  Processing series {i+1}/{len(unique_series_ids)}...")
        
        # Efficient filtering
        series_mask = df[SERIES_ID_COLUMN] == series_id
        series_df = df[series_mask].sort_values('timestamp')
        
        if len(series_df) == 0:
            continue
            
        series_target = series_df[TARGET_COLUMN].values.astype(np.float32)
        start_time = pd.Timestamp(series_df['timestamp'].iloc[0])
        
        # Simple scaling to [0, 1] range
        series_min = np.min(series_target)
        series_max = np.max(series_target)
        series_range = series_max - series_min
        
        if series_range > 0:
            scaled_target = (series_target - series_min) / series_range
        else:
            scaled_target = series_target - series_min
        
        # Split each series into train/validation
        split_idx = int(len(scaled_target) * TRAIN_VAL_SPLIT)
        
        train_target = scaled_target[:split_idx]
        test_target = scaled_target[split_idx:]
        
        if len(train_target) >= CONTEXT_LENGTH + PREDICTION_LENGTH:
            train_data.append({
                FieldName.START: start_time,
                FieldName.TARGET: train_target,
                FieldName.ITEM_ID: str(series_id)
            })
        
        if len(test_target) >= CONTEXT_LENGTH + PREDICTION_LENGTH:
            test_start = start_time + pd.Timedelta(len(train_target) * 5, 'm')
            test_data.append({
                FieldName.START: test_start,
                FieldName.TARGET: test_target,
                FieldName.ITEM_ID: str(series_id)
            })
    
    print(f"Prepared {len(train_data)} training series, {len(test_data)} test series")
    return train_data, test_data

def get_model_kwargs_from_checkpoint(checkpoint_path):
    """Extract model kwargs from checkpoint"""
    try:
        ckpt = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
        
        if 'hyper_parameters' in ckpt and 'model_kwargs' in ckpt['hyper_parameters']:
            return ckpt['hyper_parameters']['model_kwargs']
        elif 'model_kwargs' in ckpt:
            return ckpt['model_kwargs']
        else:
            print("Warning: Using default model kwargs")
            return {
                "input_size": 1,
                "n_layer": 12,
                "n_embd_per_head": 64,
                "n_head": 6,
                "scaling": "mean",
                "time_feat": 0
            }
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        raise

def finetune_lag_llama_univariate(df):
    """Fine-tune Lag-Llama on limited univariate data"""
    print("Preparing univariate data for fine-tuning...")
    
    # Analyze data and get limited series
    unique_series_ids = analyze_univariate_data(df)
    
    # Prepare data efficiently
    train_data, test_data = prepare_univariate_data_efficient(df, unique_series_ids)
    
    if not train_data:
        raise ValueError("No valid training data found.")
    
    print(f"\nFine-tuning on {len(train_data)} training series")
    print(f"Validation on {len(test_data)} test series")
    
    # Create GluonTS datasets
    train_ds = ListDataset(train_data, freq=FREQ)
    test_ds = ListDataset(test_data, freq=FREQ) if test_data else None
    
    # Load model configuration
    print("Loading model configuration from checkpoint...")
    estimator_args = get_model_kwargs_from_checkpoint(CHECKPOINT_PATH)
    
    # Create estimator optimized for univariate data
    estimator = LagLlamaEstimator(
        ckpt_path=CHECKPOINT_PATH,
        prediction_length=PREDICTION_LENGTH,
        context_length=CONTEXT_LENGTH,
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],
        batch_size=FINETUNE_BATCH_SIZE,
        num_batches_per_epoch=50,  # Reduced for faster training
        trainer_kwargs={
            "max_epochs": FINETUNE_EPOCHS,
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "enable_progress_bar": True,
            "gradient_clip_val": 0.5,
        },
    )
    
    print(f"Starting fine-tuning for {FINETUNE_EPOCHS} epochs...")
    
    # Fine-tune the model
    try:
        if test_ds:
            predictor = estimator.train(
                training_data=train_ds,
                validation_data=test_ds,
                num_workers=0,
            )
        else:
            predictor = estimator.train(
                training_data=train_ds,
                num_workers=0,
            )
        
        # Get the trained model state dict
        print("Saving fine-tuned model...")
        model_state_dict = predictor.prediction_net.state_dict()
        
        # Also get the original checkpoint to preserve other info
        original_checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False, map_location='cpu')
        
        # Update the checkpoint with fine-tuned weights
        if 'state_dict' in original_checkpoint:
            # Update the state dict while preserving other structure
            original_checkpoint['state_dict'] = model_state_dict
            # Ensure we have the pytorch-lightning version
            if 'pytorch-lightning_version' not in original_checkpoint:
                original_checkpoint['pytorch-lightning_version'] = pl.__version__
        else:
            # Create new checkpoint structure
            original_checkpoint = {
                'state_dict': model_state_dict,
                'hyper_parameters': original_checkpoint.get('hyper_parameters', {}),
                'pytorch-lightning_version': pl.__version__,
            }
        
        # Save the updated checkpoint
        torch.save(original_checkpoint, FINETUNED_CHECKPOINT_PATH)
        
        print(f"Fine-tuning completed. Model saved to {FINETUNED_CHECKPOINT_PATH}")
        return FINETUNED_CHECKPOINT_PATH
        
    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        raise

def delete_existing_checkpoint():
    """Delete existing fine-tuned checkpoint"""
    if os.path.exists(FINETUNED_CHECKPOINT_PATH):
        os.remove(FINETUNED_CHECKPOINT_PATH)
        print("Deleted existing fine-tuned checkpoint")

# --- Main Execution ---
def main():
    print("="*60)
    print("LAG-LLAMA FINE-TUNING SCRIPT")
    print("="*60)
    
    # Load data
    data_path = "/root/llama/lag-llama/network_metrics_data.csv"
    print(f"Loading data from {data_path}...")
    
    try:
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print("Univariate dataset loaded successfully.")
        print(f"Total records: {len(df)}")
        print(f"Unique flows: {df[SERIES_ID_COLUMN].nunique()}")
        
        # Check ground truth
        if GROUND_TRUTH_COLUMN in df.columns:
            print(f"Ground truth column '{GROUND_TRUTH_COLUMN}' found.")
            ground_truth_dist = df[GROUND_TRUTH_COLUMN].value_counts().to_dict()
            print(f"Ground truth distribution: {ground_truth_dist}")
            print(f"Anomaly prevalence: {ground_truth_dist.get(True, 0)/len(df)*100:.2f}%")
        else:
            print("No ground truth column found.")
        
        # Always start fresh
        print("\nStarting fresh fine-tuning...")
        delete_existing_checkpoint()
        
        # Fine-tune from scratch
        finetuned_checkpoint = finetune_lag_llama_univariate(df)
        
        print("\n" + "="*60)
        print("FINE-TUNING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Fine-tuned model saved to: {finetuned_checkpoint}")
        print(f"Next step: Run anomaly detection script")
        print("="*60)
        
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure the CSV file exists at the specified path.")
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
