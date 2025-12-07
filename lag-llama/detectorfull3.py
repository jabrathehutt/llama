import pandas as pd
import numpy as np
import torch
import torch.serialization
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
import pytorch_lightning as pl
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import json

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
LOWER_QUANTILE = 0.05
UPPER_QUANTILE = 0.95
NUM_SAMPLES = 100
FREQ = '5min'
CHECKPOINT_PATH = "/root/lag-llama/lag-llama.ckpt"
FINETUNED_CHECKPOINT_PATH = "/root/lag-llama/lag-llama-finetuned.ckpt"

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
        
        # Get the trained model
        trained_model = predictor.prediction_net
        
        # Save the fine-tuned model
        model_checkpoint = {
            'state_dict': trained_model.state_dict(),
            'hyper_parameters': {
                'model_kwargs': estimator_args
            }
        }
        
        torch.save(model_checkpoint, FINETUNED_CHECKPOINT_PATH)
        print(f"Fine-tuning completed. Model saved to {FINETUNED_CHECKPOINT_PATH}")
        return FINETUNED_CHECKPOINT_PATH
        
    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        raise

def create_predictor_safely(checkpoint_path):
    """Create predictor with proper error handling"""
    try:
        # Load model kwargs
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
        estimator_args = checkpoint['hyper_parameters']['model_kwargs']
        
        # Create estimator
        estimator = LagLlamaEstimator(
            ckpt_path=checkpoint_path,
            prediction_length=PREDICTION_LENGTH,
            context_length=CONTEXT_LENGTH,
            input_size=estimator_args["input_size"],
            n_layer=estimator_args["n_layer"],
            n_embd_per_head=estimator_args["n_embd_per_head"],
            n_head=estimator_args["n_head"],
            scaling=estimator_args["scaling"],
            time_feat=estimator_args["time_feat"],
            batch_size=1,
        )
        
        # Create predictor
        lightning_module = estimator.create_lightning_module()
        transformation = estimator.create_transformation()
        predictor = estimator.create_predictor(
            transformation=transformation,
            module=lightning_module
        )
        
        return predictor
        
    except Exception as e:
        print(f"Error creating predictor from {checkpoint_path}: {e}")
        raise

def detect_anomalies_univariate_limited(df, checkpoint_path, max_series=10):
    """Perform anomaly detection on limited series"""
    print("Loading fine-tuned model for anomaly detection...")
    
    # Create predictor
    predictor = create_predictor_safely(checkpoint_path)
    
    # Get limited series for detection
    unique_series_ids = df[SERIES_ID_COLUMN].unique()[:max_series]
    
    print(f"Starting anomaly detection on {len(unique_series_ids)} flows...")
    
    # Perform anomaly detection (rest of the function remains the same as before)
    all_anomaly_results = []
    all_predictions = []
    all_actuals = []
    all_anomaly_flags = []
    all_ground_truth = []
    all_prediction_intervals = []
    
    for series_id in unique_series_ids:
        series_mask = df[SERIES_ID_COLUMN] == series_id
        series_df = df[series_mask].sort_values('timestamp')
        series_target = series_df[TARGET_COLUMN].values.astype(np.float32)
        series_timestamps = series_df['timestamp'].values
        
        # Get scaling parameters
        series_min = np.min(series_target)
        series_max = np.max(series_target)
        series_range = series_max - series_min
        
        # Scale the series
        if series_range > 0:
            scaled_target = (series_target - series_min) / series_range
        else:
            scaled_target = series_target - series_min
        
        # Get ground truth
        if HAS_GROUND_TRUTH and GROUND_TRUTH_COLUMN in series_df.columns:
            series_ground_truth = series_df[GROUND_TRUTH_COLUMN].values
        else:
            series_ground_truth = None
        
        # Use validation portion
        split_idx = int(len(scaled_target) * TRAIN_VAL_SPLIT)
        validation_target = scaled_target[split_idx:]
        validation_timestamps = series_timestamps[split_idx:]
        
        if series_ground_truth is not None:
            validation_ground_truth = series_ground_truth[split_idx:]
        else:
            validation_ground_truth = None
        
        if len(validation_target) < CONTEXT_LENGTH + 1:
            print(f"Skipping series {series_id} - validation data too short")
            continue
            
        print(f"Processing series {series_id} with {len(validation_target)} validation points...")
        
        anomaly_count = 0
        total_points = len(validation_target) - CONTEXT_LENGTH
        
        # Process only every 10th point for speed (adjust as needed)
        step_size = max(1, total_points // 100)  # Process at most 100 points per series
        
        for end_index in range(CONTEXT_LENGTH, len(validation_target), step_size):
            window_target = validation_target[end_index - CONTEXT_LENGTH : end_index + 1]
            actual_scaled = window_target[-1]
            actual_original = series_target[split_idx + end_index]
            anomaly_time = validation_timestamps[end_index]
            
            # Create dataset for prediction
            test_ds = ListDataset([{
                "start": pd.Timestamp(validation_timestamps[end_index - CONTEXT_LENGTH]),
                "target": window_target[:-1],
            }], freq=FREQ)
            
            # Generate forecast
            try:
                forecasts = list(predictor.predict(test_ds, num_samples=NUM_SAMPLES))
                if forecasts:
                    forecast = forecasts[0]
                    samples = forecast.samples[:, 0]
                    
                    # Descale predictions
                    if series_range > 0:
                        point_forecast = np.median(samples) * series_range + series_min
                        lower_bound = np.quantile(samples, LOWER_QUANTILE) * series_range + series_min
                        upper_bound = np.quantile(samples, UPPER_QUANTILE) * series_range + series_min
                    else:
                        point_forecast = np.median(samples) + series_min
                        lower_bound = np.quantile(samples, LOWER_QUANTILE) + series_min
                        upper_bound = np.quantile(samples, UPPER_QUANTILE) + series_min
                    
                    # Anomaly check
                    is_anomaly = (actual_original < lower_bound) or (actual_original > upper_bound)
                    
                    # Store for metrics
                    all_predictions.append(point_forecast)
                    all_actuals.append(actual_original)
                    all_anomaly_flags.append(is_anomaly)
                    all_prediction_intervals.append((lower_bound, upper_bound))
                    
                    if validation_ground_truth is not None:
                        all_ground_truth.append(validation_ground_truth[end_index])
                    
                    if is_anomaly:
                        anomaly_count += 1
                        all_anomaly_results.append({
                            "series_id": series_id,
                            "timestamp": anomaly_time,
                            "actual_value": actual_original,
                            "predicted_value": point_forecast,
                            f"lower_bound_{LOWER_QUANTILE*100:.1f}": lower_bound,
                            f"upper_bound_{UPPER_QUANTILE*100:.1f}": upper_bound,
                            "is_anomaly": is_anomaly
                        })
            except Exception as e:
                print(f"Error predicting for series {series_id} at index {end_index}: {e}")
                continue
        
        print(f"Series {series_id}: {anomaly_count}/{total_points} anomalies detected")
    
    # Calculate metrics
    ground_truth_for_metrics = all_ground_truth if all_ground_truth and len(all_ground_truth) > 0 else None
    metrics = calculate_metrics(all_predictions, all_actuals, all_anomaly_flags, 
                               all_prediction_intervals, all_ground_truth=ground_truth_for_metrics)
    
    return all_anomaly_results, metrics

def calculate_metrics(all_predictions, all_actuals, all_anomaly_flags, all_prediction_intervals, all_ground_truth=None):
    """Calculate comprehensive evaluation metrics"""
    # ... (same as before)
    predictions = np.array(all_predictions)
    actuals = np.array(all_actuals)
    anomalies = np.array(all_anomaly_flags)
    
    mae = np.mean(np.abs(predictions - actuals))
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    
    non_zero_actuals = actuals != 0
    if np.any(non_zero_actuals):
        mape = np.mean(np.abs((actuals[non_zero_actuals] - predictions[non_zero_actuals]) / actuals[non_zero_actuals])) * 100
    else:
        mape = np.nan
    
    if all_prediction_intervals and len(all_prediction_intervals) > 0:
        lower_bounds = np.array([p[0] for p in all_prediction_intervals])
        upper_bounds = np.array([p[1] for p in all_prediction_intervals])
        coverage = np.mean((actuals >= lower_bounds) & (actuals <= upper_bounds)) * 100
        interval_widths = upper_bounds - lower_bounds
        mean_interval_width = np.mean(interval_widths)
    else:
        coverage = 0.0
        mean_interval_width = 0.0
    
    metrics = {
        'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape,
        'Coverage_%': coverage, 'Mean_Interval_Width': mean_interval_width,
        'Total_Points': len(actuals),
        'Anomaly_Rate_%': (np.sum(anomalies) / len(anomalies)) * 100 if len(anomalies) > 0 else 0
    }
    
    if all_ground_truth is not None and HAS_GROUND_TRUTH and len(all_ground_truth) > 0:
        ground_truth = np.array(all_ground_truth)
        min_length = min(len(ground_truth), len(anomalies))
        ground_truth = ground_truth[:min_length]
        anomalies_subset = anomalies[:min_length]
        
        precision = precision_score(ground_truth, anomalies_subset, zero_division=0)
        recall = recall_score(ground_truth, anomalies_subset, zero_division=0)
        f1 = f1_score(ground_truth, anomalies_subset, zero_division=0)
        accuracy = accuracy_score(ground_truth, anomalies_subset)
        
        cm = confusion_matrix(ground_truth, anomalies_subset)
        if cm.size == 1:
            if ground_truth[0] == 0:
                tn, fp, fn, tp = len(ground_truth), 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, len(ground_truth)
        else:
            tn, fp, fn, tp = cm.ravel()
        
        metrics.update({
            'Precision': precision, 'Recall': recall, 'F1_Score': f1, 'Accuracy': accuracy,
            'True_Positives': tp, 'False_Positives': fp, 'True_Negatives': tn, 'False_Negatives': fn
        })
    
    return metrics

def delete_existing_checkpoint():
    """Delete existing fine-tuned checkpoint"""
    if os.path.exists(FINETUNED_CHECKPOINT_PATH):
        os.remove(FINETUNED_CHECKPOINT_PATH)
        print("Deleted existing fine-tuned checkpoint")

# --- Main Execution ---
def main():
    # Load data
    df = pd.read_csv("/root/lag-llama/network_metrics_data.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print("Univariate dataset loaded successfully.")
    print(f"Total records: {len(df)}")
    print(f"Unique flows: {df[SERIES_ID_COLUMN].nunique()}")
    
    # Check ground truth
    global HAS_GROUND_TRUTH
    if GROUND_TRUTH_COLUMN in df.columns:
        HAS_GROUND_TRUTH = True
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
    try:
        finetuned_checkpoint = finetune_lag_llama_univariate(df)
        
        # Perform anomaly detection on limited series
        print("\nStarting anomaly detection with fine-tuned model...")
        anomaly_results, metrics = detect_anomalies_univariate_limited(df, finetuned_checkpoint, max_series=5)
        
        # Output results
        anomaly_df = pd.DataFrame(anomaly_results)
        
        print(f"\nAnalysis Complete. Total anomalies detected: {len(anomaly_df)}")
        
        # Print comprehensive metrics
        print("\n" + "="*60)
        print("COMPREHENSIVE PERFORMANCE METRICS")
        print("="*60)
        
        print(f"\n--- Forecast Accuracy Metrics ---")
        print(f"MAE (Mean Absolute Error): {metrics['MAE']:.6f}")
        print(f"MSE (Mean Squared Error): {metrics['MSE']:.6f}")
        print(f"RMSE (Root Mean Squared Error): {metrics['RMSE']:.6f}")
        if not np.isnan(metrics['MAPE']):
            print(f"MAPE (Mean Absolute Percentage Error): {metrics['MAPE']:.2f}%")
        
        print(f"\n--- Prediction Interval Metrics ---")
        print(f"Coverage (% of points within prediction interval): {metrics['Coverage_%']:.2f}%")
        print(f"Mean Prediction Interval Width: {metrics['Mean_Interval_Width']:.6f}")
        print(f"Anomaly Detection Rate: {metrics['Anomaly_Rate_%']:.2f}%")
        
        if HAS_GROUND_TRUTH and 'Precision' in metrics:
            print(f"\n--- Classification Metrics ---")
            print(f"Precision: {metrics['Precision']:.4f}")
            print(f"Recall: {metrics['Recall']:.4f}")
            print(f"F1-Score: {metrics['F1_Score']:.4f}")
            print(f"Accuracy: {metrics['Accuracy']:.4f}")
            print(f"\nConfusion Matrix:")
            print(f"True Positives: {metrics['True_Positives']}")
            print(f"False Positives: {metrics['False_Positives']}")
            print(f"True Negatives: {metrics['True_Negatives']}")
            print(f"False Negatives: {metrics['False_Negatives']}")
        
        print(f"\n--- Summary ---")
        print(f"Total Points Analyzed: {metrics['Total_Points']}")
        print(f"Total Anomalies Detected: {len(anomaly_df)}")
        
        if not anomaly_df.empty:
            output_file = "lag_llama_univariate_anomalies.csv"
            anomaly_df.to_csv(output_file, index=False)
            
            metrics_file = "univariate_anomaly_metrics.json"
            with open(metrics_file, 'w') as f:
                json_metrics = {}
                for k, v in metrics.items():
                    if isinstance(v, (np.floating, np.integer)):
                        json_metrics[k] = float(v)
                    elif np.isnan(v):
                        json_metrics[k] = None
                    else:
                        json_metrics[k] = v
                json.dump(json_metrics, f, indent=2)
            
            print("\n--- Sample Anomalies ---")
            print(anomaly_df.head(10).to_string())
            print(f"\nFull results saved to {output_file}")
            print(f"Metrics saved to {metrics_file}")
            
        else:
            print("NO ANOMALIES DETECTED")
            
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
