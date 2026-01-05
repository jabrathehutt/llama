import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

import os
import sys
import torch
import pandas as pd
import numpy as np
from types import ModuleType
import argparse
import lightning.pytorch as lp
from gluonts.dataset.common import ListDataset
from lag_llama.gluon.estimator import LagLlamaEstimator
import gluonts.torch.distributions as gtd

# --- 1. THE LEGACY COMPATIBILITY BRIDGE ---
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

torch.serialization.add_safe_globals([gtd.studentT.StudentTOutput, gtd.StudentTOutput])
create_dummy_module('gluonts.torch.modules.loss')
class MockL:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, *args, **kwargs): return 0.0
sys.modules['gluonts.torch.modules.loss'].DistributionLoss = MockL
sys.modules['gluonts.torch.modules.loss'].NegativeLogLikelihood = MockL

import gluonts.time_feature.lag as lag_mod
lag_mod.to_offset = lambda x: pd.tseries.frequencies.to_offset('5min')

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    lp.seed_everything(seed, workers=True)

def load_network_data(csv_path, freq="5min"):
    df = pd.read_csv(csv_path)
    # Log-scaling handles real byte variance (e.g., small packets vs large transfers)
    df['target_scaled'] = np.log1p(df['traffic_volume_Tbits'].values.astype(np.float32))

    dataset_list = []
    for flow_id, group in df.groupby("flow_key_id"):
        target_values = group.sort_values("timestamp")['target_scaled'].values
        dataset_list.append({
            "start": pd.Period(group["timestamp"].min(), freq=freq),
            "target": target_values,
            "item_id": str(flow_id)
        })
    return ListDataset(dataset_list, freq=freq)

def train(args):
    set_seed(args.seed)
    print(f"Loading CLEAN dataset from: {args.dataset_path}/clean_training_traffic_data.csv")
    network_ds = load_network_data(f"{args.dataset_path}/clean_training_traffic_data.csv")

    lags_seq_eval = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 48, 72, 96, 120, 144, 288, 576]

    estimator = LagLlamaEstimator(
        prediction_length=args.prediction_length,
        context_length=args.context_length,
        batch_size=args.batch_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd_per_head=args.n_embd_per_head,
        lr=args.lr,
        lags_seq=lags_seq_eval,
        num_batches_per_epoch=args.num_batches_per_epoch,
        distr_output="studentT",
        scaling=args.data_normalization,
        trainer_kwargs=dict(
            max_epochs=args.max_epochs, 
            accelerator="gpu", 
            devices=[args.gpu], # Uses the GPU index passed from shell
            default_root_dir=args.results_dir
        )
    )

    # --- 2. THE GRAFTING LOGIC ---
    transformation = estimator.create_transformation()
    lightning_module = estimator.create_lightning_module()
    predictor = estimator.create_predictor(transformation, lightning_module.model)

    checkpoint_path = "specialized_network_llama.pt"
    if os.path.exists(checkpoint_path):
        print(f"Grafting weights from {checkpoint_path}...")
        ckpt_state = torch.load(checkpoint_path, map_location="cpu")
        model_state = predictor.network.state_dict()
        
        # Check for key alignment
        ckpt_key = "model.transformer.wte.weight"
        model_key = "model.transformer.wte.weight" if "model.transformer.wte.weight" in model_state else "transformer.wte.weight"
        
        if ckpt_key in ckpt_state:
            learned_w = ckpt_state[ckpt_key]
            target_w = model_state[model_key]
            if learned_w.shape != target_w.shape:
                new_w = torch.zeros_like(target_w)
                new_w[:learned_w.shape[0], :learned_w.shape[1]] = learned_w
                ckpt_state[model_key] = new_w
        
        predictor.network.load_state_dict(ckpt_state, strict=False)

    predictor = estimator.train(training_data=network_ds)
    torch.save(predictor.network.state_dict(), "real_world_network_llama.pt")
    print("Success. real_world_network_llama.pt generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_name", type=str, required=True)
    parser.add_argument("-d", "--dataset_path", type=str, default=".")
    parser.add_argument("-r", "--results_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_batches_per_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--context_length", type=int, default=32)
    parser.add_argument("--prediction_length", type=int, default=1)
    parser.add_argument("--n_layer", type=int, default=8)
    parser.add_argument("--n_head", type=int, default=9)
    parser.add_argument("--n_embd_per_head", type=int, default=16)
    parser.add_argument("--data_normalization", default="mean")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0) # ADDED THIS ARGUMENT
    
    args = parser.parse_args()
    train(args)
