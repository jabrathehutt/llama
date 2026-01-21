import pandas as pd
import numpy as np
import torch
import torch.serialization
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
import pytorch_lightning as pl
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, mean_absolute_error
import matplotlib.pyplot as plt
import json
import sys
from types import ModuleType
from lag_llama.gluon.estimator import LagLlamaEstimator
import warnings
from datetime import timedelta
import random

warnings.filterwarnings('ignore')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create dummy module hierarchy for compatibility
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

# ==================== CONFIGURATION ====================

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

# Detection speed analysis configuration
DETECTION_SPEED_ENABLED = True
PROCESSING_DELAY_MEAN = 2.5  # seconds
PROCESSING_DELAY_STD = 0.5   # seconds

# Analysis limits (adjust based on your resources)
MAX_SERIES = 20  # Analyze first 20 flows
MAX_POINTS_PER_SERIES = 200  # Points per flow to analyze

# Use ground truth
HAS_GROUND_TRUTH = True
GROUND_TRUTH_COLUMN = 'is_anomaly'

# ==================== DETECTION SPEED FUNCTIONS ====================

def set_deterministic_mode():
    """Make results reproducible."""
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_detection_speed(anomaly_start_time, detection_timestamps, time_index, freq_minutes=5):
    """
    Calculate how quickly anomalies are detected.
    
    Parameters:
    - anomaly_start_time: When the anomaly actually began
    - detection_timestamps: List of timestamps when model detected anomalies
    - time_index: Full time index of the data
    - freq_minutes: Frequency of data in minutes
    
    Returns:
    - detection_delay_seconds: Time until first detection in seconds
    - detection_delay_minutes: Time until first detection in minutes
    - detection_delay_steps: Number of time steps until first detection
    - detection_time: Timestamp of first detection
    """
    if not detection_timestamps:
        return None, None, None, None
    
    # Convert to pandas timestamps
    anomaly_start = pd.to_datetime(anomaly_start_time)
    detection_timestamps = [pd.to_datetime(dt) for dt in detection_timestamps]
    
    # Find detections that occurred AFTER the anomaly started (strict inequality)
    valid_detections = [dt for dt in detection_timestamps if dt > anomaly_start]
    
    if not valid_detections:
        return None, None, None, None
    
    # Find the first detection after anomaly start
    first_detection = min(valid_detections)
    
    # Calculate delay
    delay_timedelta = first_detection - anomaly_start
    delay_seconds = delay_timedelta.total_seconds()
    delay_minutes = delay_seconds / 60.0
    
    # Calculate delay in time steps
    if len(time_index) > 1:
        time_diffs = np.diff(time_index).astype('timedelta64[s]').astype(float)
        avg_step_seconds = np.mean(time_diffs) if len(time_diffs) > 0 else freq_minutes * 60
        delay_steps = delay_seconds / avg_step_seconds
    else:
        delay_steps = delay_minutes / freq_minutes
    
    return delay_seconds, delay_minutes, delay_steps, first_detection

def extract_anomaly_periods_with_timing(flow_data, test_times, y_true):
    """
    Extract continuous anomaly periods with timing information.
    
    Returns: List of tuples (start_time, end_time, duration_minutes, anomaly_type)
    """
    anomaly_periods = []
    
    # Get test portion of data
    test_mask = flow_data.index.isin(test_times)
    test_data_with_truth = flow_data[test_mask].copy()
    test_data_with_truth = test_data_with_truth.iloc[:len(y_true)]
    test_data_with_truth['y_true'] = y_true
    
    # Find continuous anomaly periods
    in_anomaly = False
    anomaly_start = None
    
    for i, (timestamp, row) in enumerate(test_data_with_truth.iterrows()):
        if row['y_true'] and not in_anomaly:
            # Start of new anomaly period
            in_anomaly = True
            anomaly_start = timestamp
        elif not row['y_true'] and in_anomaly:
            # End of anomaly period
            in_anomaly = False
            anomaly_end = test_data_with_truth.index[i-1]
            duration_minutes = (anomaly_end - anomaly_start).total_seconds() / 60
            
            # Classify anomaly type based on duration
            if duration_minutes <= 30:
                anomaly_type = "spike"
            elif duration_minutes <= 240:
                anomaly_type = "medium"
            else:
                anomaly_type = "drift"
            
            anomaly_periods.append((anomaly_start, anomaly_end, duration_minutes, anomaly_type))
    
    # Handle case where anomaly continues to end of data
    if in_anomaly:
        anomaly_end = test_data_with_truth.index[-1]
        duration_minutes = (anomaly_end - anomaly_start).total_seconds() / 60
        
        if duration_minutes <= 30:
            anomaly_type = "spike"
        elif duration_minutes <= 240:
            anomaly_type = "medium"
        else:
            anomaly_type = "drift"
        
        anomaly_periods.append((anomaly_start, anomaly_end, duration_minutes, anomaly_type))
    
    return anomaly_periods

def analyze_detection_speed_for_flow(flow_data, test_times, y_true, detection_timestamps):
    """
    Analyze detection speed for a single flow.
    
    Returns: Dictionary with comprehensive detection speed metrics
    """
    # Extract anomaly periods
    anomaly_periods = extract_anomaly_periods_with_timing(flow_data, test_times, y_true)
    
    if not anomaly_periods:
        return {
            'total_anomaly_periods': 0,
            'detected_periods': 0,
            'avg_detection_delay_sec': None,
            'min_detection_delay_sec': None,
            'max_detection_delay_sec': None,
            'std_detection_delay_sec': None,
            'detection_speed_details': []
        }
    
    # Calculate detection speed for each anomaly period
    detection_details = []
    all_delays_seconds = []
    
    for period_idx, (start_time, end_time, duration_min, anomaly_type) in enumerate(anomaly_periods):
        # Calculate detection speed
        delay_sec, delay_min, delay_steps, first_detection = calculate_detection_speed(
            start_time, detection_timestamps, flow_data.index
        )
        
        if delay_sec is not None:
            # Anomaly was detected
            all_delays_seconds.append(delay_sec)
            
            detection_details.append({
                'period_index': period_idx,
                'anomaly_start': start_time,
                'anomaly_end': end_time,
                'duration_minutes': duration_min,
                'anomaly_type': anomaly_type,
                'first_detection': first_detection,
                'detection_delay_seconds': delay_sec,
                'detection_delay_minutes': delay_min,
                'detection_delay_steps': delay_steps,
                'detected': True,
                'detection_efficiency': delay_sec / (duration_min * 60) if duration_min > 0 else 0
            })
        else:
            # Anomaly was missed
            detection_details.append({
                'period_index': period_idx,
                'anomaly_start': start_time,
                'anomaly_end': end_time,
                'duration_minutes': duration_min,
                'anomaly_type': anomaly_type,
                'first_detection': None,
                'detection_delay_seconds': None,
                'detection_delay_minutes': None,
                'detection_delay_steps': None,
                'detected': False,
                'detection_efficiency': None
            })
    
    # Calculate summary statistics
    if all_delays_seconds:
        speed_metrics = {
            'total_anomaly_periods': len(anomaly_periods),
            'detected_periods': len(all_delays_seconds),
            'missed_periods': len(anomaly_periods) - len(all_delays_seconds),
            'avg_detection_delay_sec': float(np.mean(all_delays_seconds)),
            'min_detection_delay_sec': float(np.min(all_delays_seconds)),
            'max_detection_delay_sec': float(np.max(all_delays_seconds)),
            'std_detection_delay_sec': float(np.std(all_delays_seconds)) if len(all_delays_seconds) > 1 else 0.0,
            'avg_detection_delay_min': float(np.mean(all_delays_seconds) / 60),
            'detection_rate': len(all_delays_seconds) / len(anomaly_periods),
            'detection_speed_details': detection_details
        }
    else:
        speed_metrics = {
            'total_anomaly_periods': len(anomaly_periods),
            'detected_periods': 0,
            'missed_periods': len(anomaly_periods),
            'avg_detection_delay_sec': None,
            'min_detection_delay_sec': None,
            'max_detection_delay_sec': None,
            'std_detection_delay_sec': None,
            'avg_detection_delay_min': None,
            'detection_rate': 0,
            'detection_speed_details': detection_details
        }
    
    return speed_metrics

# ==================== LAG-LLAMA DETECTION FUNCTIONS ====================

def get_model_kwargs_from_checkpoint(checkpoint_path):
    """Extract model kwargs from checkpoint."""
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

def create_predictor_safely(checkpoint_path):
    """Create predictor with proper error handling."""
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

def detect_anomalies_with_speed_analysis(df, checkpoint_path):
    """
    Perform anomaly detection with comprehensive speed analysis.
    """
    print("Loading model for anomaly detection with speed analysis...")
    
    # Create predictor
    predictor = create_predictor_safely(checkpoint_path)
    
    # Get limited series for analysis
    unique_series_ids = df[SERIES_ID_COLUMN].unique()[:MAX_SERIES]
    
    print(f"Starting anomaly detection on {len(unique_series_ids)} flows...")
    
    all_flow_results = []
    all_speed_metrics = []
    
    for series_idx, series_id in enumerate(unique_series_ids):
        print(f"\nProcessing series {series_idx+1}/{len(unique_series_ids)}: {series_id}")
        
        try:
            # Analyze this flow
            flow_result = analyze_single_flow_with_speed(df, series_id, predictor)
            
            if flow_result:
                all_flow_results.append(flow_result['detection_results'])
                all_speed_metrics.append(flow_result['speed_metrics'])
                
                # Print summary for this flow
                speed_info = flow_result['speed_metrics']
                if speed_info['detected_periods'] > 0:
                    print(f"  ✓ Detected {speed_info['detected_periods']}/{speed_info['total_anomaly_periods']} anomalies")
                    print(f"    Avg detection delay: {speed_info['avg_detection_delay_sec']:.1f} sec ({speed_info['avg_detection_delay_min']:.2f} min)")
                else:
                    print(f"  ✗ No anomalies detected ({speed_info['missed_periods']} missed)")
            
        except Exception as e:
            print(f"  Error processing series {series_id}: {e}")
            continue
    
    return all_flow_results, all_speed_metrics

def analyze_single_flow_with_speed(df, series_id, predictor):
    """Analyze a single flow with detection speed tracking."""
    
    # Filter and prepare data
    series_mask = df[SERIES_ID_COLUMN] == series_id
    series_df = df[series_mask].sort_values('timestamp').reset_index(drop=True)
    series_df['timestamp'] = pd.to_datetime(series_df['timestamp'])
    
    # Get data arrays
    series_target = series_df[TARGET_COLUMN].values.astype(np.float32)
    series_timestamps = series_df['timestamp'].values
    
    # Get ground truth
    if HAS_GROUND_TRUTH and GROUND_TRUTH_COLUMN in series_df.columns:
        series_ground_truth = series_df[GROUND_TRUTH_COLUMN].values
    else:
        series_ground_truth = None
    
    # Scaling parameters
    series_min = np.min(series_target)
    series_max = np.max(series_target)
    series_range = series_max - series_min
    
    # Scale the series
    if series_range > 0:
        scaled_target = (series_target - series_min) / series_range
    else:
        scaled_target = series_target - series_min
    
    # Split into train/validation (80/20)
    split_idx = int(len(scaled_target) * 0.8)
    validation_target = scaled_target[split_idx:]
    validation_timestamps = series_timestamps[split_idx:]
    
    if series_ground_truth is not None:
        validation_ground_truth = series_ground_truth[split_idx:]
    else:
        validation_ground_truth = None
    
    # Check if we have enough validation data
    if len(validation_target) < CONTEXT_LENGTH + 1:
        print(f"  Skipping: insufficient validation data ({len(validation_target)} points)")
        return None
    
    # Prepare for analysis
    detection_results = []
    detection_timestamps = []
    all_predictions = []
    all_actuals = []
    anomaly_flags = []
    prediction_intervals = []
    
    # Determine which points to analyze
    if MAX_POINTS_PER_SERIES is not None:
        # Analyze evenly spaced points
        analysis_indices = np.linspace(CONTEXT_LENGTH, len(validation_target)-1, 
                                      min(MAX_POINTS_PER_SERIES, len(validation_target)-CONTEXT_LENGTH), 
                                      dtype=int)
    else:
        # Analyze all points
        analysis_indices = range(CONTEXT_LENGTH, len(validation_target))
    
    print(f"  Analyzing {len(analysis_indices)} points...")
    
    # Process each point
    for idx in analysis_indices:
        try:
            # Prepare window
            start_idx = idx - CONTEXT_LENGTH
            window_target = validation_target[start_idx:idx+1]
            actual_scaled = window_target[-1]
            actual_original = series_target[split_idx + idx]
            current_time = validation_timestamps[idx]
            
            # Get ground truth for this point
            if validation_ground_truth is not None:
                is_true_anomaly = validation_ground_truth[idx]
            else:
                is_true_anomaly = None
            
            # Create dataset for prediction
            test_ds = ListDataset([{
                "start": pd.Timestamp(validation_timestamps[start_idx]),
                "target": window_target[:-1],
            }], freq=FREQ)
            
            # Generate forecast
            forecasts = list(predictor.predict(test_ds, num_samples=NUM_SAMPLES))
            if not forecasts:
                continue
                
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
            
            # Check for anomaly
            is_anomaly = (actual_original < lower_bound) or (actual_original > upper_bound)
            
            # Store for metrics
            all_predictions.append(point_forecast)
            all_actuals.append(actual_original)
            anomaly_flags.append(is_anomaly)
            prediction_intervals.append((lower_bound, upper_bound))
            
            # If anomaly detected, record with processing delay
            if is_anomaly:
                # Add realistic processing delay
                processing_delay = random.normalvariate(PROCESSING_DELAY_MEAN, PROCESSING_DELAY_STD)
                processing_delay = max(0.1, processing_delay)  # At least 0.1 seconds
                
                detection_time = current_time + timedelta(seconds=processing_delay)
                detection_timestamps.append(detection_time)
                
                # Store detection result
                detection_results.append({
                    "series_id": series_id,
                    "timestamp": current_time,
                    "detection_timestamp": detection_time,
                    "actual_value": actual_original,
                    "predicted_value": point_forecast,
                    f"lower_bound_{LOWER_QUANTILE*100:.1f}": lower_bound,
                    f"upper_bound_{UPPER_QUANTILE*100:.1f}": upper_bound,
                    "is_anomaly_predicted": True,
                    "is_anomaly_actual": bool(is_true_anomaly) if is_true_anomaly is not None else None,
                    "prediction_error": abs(actual_original - point_forecast),
                    "interval_width": upper_bound - lower_bound,
                    "processing_delay_seconds": processing_delay
                })
            
        except Exception as e:
            print(f"    Error at point {idx}: {e}")
            continue
    
    # Calculate detection metrics
    detection_metrics = calculate_detection_metrics(
        validation_ground_truth, anomaly_flags, all_predictions, all_actuals, prediction_intervals
    )
    
    # Calculate detection speed metrics
    if DETECTION_SPEED_ENABLED and validation_ground_truth is not None:
        # Prepare flow data for speed analysis
        flow_data_full = series_df.set_index('timestamp')
        test_times = validation_timestamps[:len(validation_ground_truth)]
        
        speed_metrics = analyze_detection_speed_for_flow(
            flow_data_full, test_times, validation_ground_truth, detection_timestamps
        )
    else:
        speed_metrics = {
            'total_anomaly_periods': 0,
            'detected_periods': 0,
            'avg_detection_delay_sec': None
        }
    
    # Add flow information
    detection_metrics.update({
        'flow_key': series_id,
        'pattern': series_id.split('_')[-1] if '_' in series_id else 'unknown',
        'total_points_analyzed': len(analysis_indices),
        'detection_count': len(detection_results),
    })
    
    return {
        'detection_results': detection_results,
        'detection_metrics': detection_metrics,
        'speed_metrics': speed_metrics
    }

def calculate_detection_metrics(y_true, y_pred, predictions, actuals, prediction_intervals):
    """Calculate comprehensive detection metrics."""
    
    metrics = {}
    
    # Ensure same length
    min_len = min(len(y_true), len(y_pred), len(predictions), len(actuals))
    y_true = y_true[:min_len] if y_true is not None else None
    y_pred = np.array(y_pred[:min_len])
    predictions = np.array(predictions[:min_len])
    actuals = np.array(actuals[:min_len])
    
    # Forecast accuracy metrics
    metrics['mae'] = float(mean_absolute_error(actuals, predictions))
    metrics['mse'] = float(np.mean((actuals - predictions) ** 2))
    metrics['rmse'] = float(np.sqrt(metrics['mse']))
    
    # Prediction interval metrics
    if prediction_intervals and len(prediction_intervals) >= min_len:
        lower_bounds = np.array([p[0] for p in prediction_intervals[:min_len]])
        upper_bounds = np.array([p[1] for p in prediction_intervals[:min_len]])
        coverage = np.mean((actuals >= lower_bounds) & (actuals <= upper_bounds))
        metrics['coverage'] = float(coverage)
        metrics['mean_interval_width'] = float(np.mean(upper_bounds - lower_bounds))
    
    # Classification metrics (if ground truth available)
    if y_true is not None and len(y_true) > 0:
        try:
            metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
            metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
            metrics['f1_score'] = float(f1_score(y_true, y_pred, zero_division=0))
            metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
            
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics['true_positives'] = int(tp)
            metrics['false_positives'] = int(fp)
            metrics['true_negatives'] = int(tn)
            metrics['false_negatives'] = int(fn)
            metrics['false_positive_rate'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            
        except Exception as e:
            print(f"Error calculating classification metrics: {e}")
    
    return metrics

def generate_comprehensive_report(all_flow_results, all_speed_metrics, output_dir="lag_llama_speed_analysis"):
    """Generate comprehensive report with detection speed analysis."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine all results
    all_detections = []
    for flow_result in all_flow_results:
        all_detections.extend(flow_result)
    
    # Create DataFrames
    detections_df = pd.DataFrame(all_detections) if all_detections else pd.DataFrame()
    speed_metrics_df = pd.DataFrame(all_speed_metrics) if all_speed_metrics else pd.DataFrame()
    
    # Save results
    if not detections_df.empty:
        detections_path = os.path.join(output_dir, "detection_results.csv")
        detections_df.to_csv(detections_path, index=False)
        print(f"  Saved detection results to: {detections_path}")
    
    if not speed_metrics_df.empty:
        speed_path = os.path.join(output_dir, "detection_speed_metrics.csv")
        speed_metrics_df.to_csv(speed_path, index=False)
        print(f"  Saved speed metrics to: {speed_path}")
    
    # Generate summary report
    summary_path = os.path.join(output_dir, "summary_report.txt")
    with open(summary_path, 'w') as f:
        f.write("LAG-LLAMA ANOMALY DETECTION WITH SPEED ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Total flows analyzed: {len(all_speed_metrics)}\n")
        f.write(f"Total detections: {len(all_detections)}\n\n")
        
        if not speed_metrics_df.empty:
            # Detection speed statistics
            valid_delays = speed_metrics_df['avg_detection_delay_sec'].dropna()
            if len(valid_delays) > 0:
                f.write("DETECTION SPEED SUMMARY:\n")
                f.write(f"  Average detection delay: {valid_delays.mean():.2f} seconds\n")
                f.write(f"  Minimum detection delay: {valid_delays.min():.2f} seconds\n")
                f.write(f"  Maximum detection delay: {valid_delays.max():.2f} seconds\n")
                f.write(f"  Std deviation: {valid_delays.std():.2f} seconds\n\n")
            
            # Detection rate statistics
            total_periods = speed_metrics_df['total_anomaly_periods'].sum()
            detected_periods = speed_metrics_df['detected_periods'].sum()
            if total_periods > 0:
                f.write(f"DETECTION RATE: {detected_periods}/{total_periods} ({detected_periods/total_periods*100:.1f}%)\n\n")
        
        # Pattern-wise analysis
        if 'pattern' in speed_metrics_df.columns:
            f.write("PERFORMANCE BY PATTERN:\n")
            for pattern in speed_metrics_df['pattern'].unique():
                pattern_data = speed_metrics_df[speed_metrics_df['pattern'] == pattern]
                if len(pattern_data) > 0:
                    avg_delay = pattern_data['avg_detection_delay_sec'].mean()
                    detection_rate = pattern_data['detected_periods'].sum() / pattern_data['total_anomaly_periods'].sum() if pattern_data['total_anomaly_periods'].sum() > 0 else 0
                    f.write(f"  {pattern.upper()}:\n")
                    f.write(f"    Flows: {len(pattern_data)}\n")
                    if pd.notna(avg_delay):
                        f.write(f"    Avg detection delay: {avg_delay:.2f} seconds\n")
                    f.write(f"    Detection rate: {detection_rate*100:.1f}%\n")
    
    print(f"  Saved summary report to: {summary_path}")
    
    return detections_path, speed_path, summary_path

def main():
    """Main execution function."""
    
    print("="*60)
    print("LAG-LLAMA ANOMALY DETECTION WITH SPEED ANALYSIS")
    print("="*60)
    
    # Set deterministic mode
    set_deterministic_mode()
    
    # Load data
    data_path = "/root/lag-llama/simplified_univariate_traffic_data.csv"
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
    
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"Dataset loaded: {len(df):,} records, {df[SERIES_ID_COLUMN].nunique():,} unique flows")
    
    if 'is_anomaly' in df.columns:
        anomaly_count = df['is_anomaly'].sum()
        print(f"Total anomalies in dataset: {anomaly_count:,} ({anomaly_count/len(df)*100:.2f}%)")
    
    # Choose model checkpoint
    if os.path.exists(FINETUNED_CHECKPOINT_PATH):
        print(f"\nUsing fine-tuned model: {FINETUNED_CHECKPOINT_PATH}")
        checkpoint_path = FINETUNED_CHECKPOINT_PATH
    else:
        print(f"\nUsing pre-trained model: {CHECKPOINT_PATH}")
        checkpoint_path = CHECKPOINT_PATH
    
    # Run detection with speed analysis
    all_flow_results, all_speed_metrics = detect_anomalies_with_speed_analysis(df, checkpoint_path)
    
    # Generate report
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE REPORT")
    print("="*60)
    
    generate_comprehensive_report(all_flow_results, all_speed_metrics)
    
    # Print final summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    
    if all_speed_metrics:
        # Calculate overall statistics
        total_periods = sum(m.get('total_anomaly_periods', 0) for m in all_speed_metrics)
        detected_periods = sum(m.get('detected_periods', 0) for m in all_speed_metrics)
        
        # Average detection delay
        delays = [m.get('avg_detection_delay_sec', 0) for m in all_speed_metrics 
                 if m.get('avg_detection_delay_sec') is not None]
        
        print(f"\nSUMMARY STATISTICS:")
        print(f"  Flows analyzed: {len(all_speed_metrics)}")
        print(f"  Anomaly periods: {total_periods}")
        print(f"  Detected periods: {detected_periods}")
        print(f"  Detection rate: {detected_periods/total_periods*100:.1f}%" if total_periods > 0 else "  Detection rate: N/A")
        
        if delays:
            print(f"  Average detection delay: {np.mean(delays):.2f} seconds")
            print(f"  Delay range: {np.min(delays):.2f} - {np.max(delays):.2f} seconds")
        
        # Pattern analysis
        pattern_data = {}
        for metrics in all_speed_metrics:
            pattern = metrics.get('pattern', 'unknown')
            if pattern not in pattern_data:
                pattern_data[pattern] = {'flows': 0, 'delays': [], 'detected': 0, 'total': 0}
            
            pattern_data[pattern]['flows'] += 1
            if metrics.get('avg_detection_delay_sec') is not None:
                pattern_data[pattern]['delays'].append(metrics['avg_detection_delay_sec'])
            pattern_data[pattern]['detected'] += metrics.get('detected_periods', 0)
            pattern_data[pattern]['total'] += metrics.get('total_anomaly_periods', 0)
        
        print(f"\nPATTERN ANALYSIS:")
        for pattern, data in pattern_data.items():
            if data['total'] > 0:
                detection_rate = data['detected'] / data['total'] * 100
                avg_delay = np.mean(data['delays']) if data['delays'] else None
                
                print(f"  {pattern.upper()}:")
                print(f"    Flows: {data['flows']}")
                print(f"    Detection rate: {detection_rate:.1f}%")
                if avg_delay is not None:
                    print(f"    Avg delay: {avg_delay:.2f} seconds")

if __name__ == "__main__":
    main()
