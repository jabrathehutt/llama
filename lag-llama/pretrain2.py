import os
import sys
import torch
import pandas as pd
import numpy as np
import random
from types import ModuleType
from pathlib import Path
import lightning.pytorch as lp

# --- 0. DETERMINISTIC SEEDING ---
def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Forces PyTorch to use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # For newer PyTorch versions
    torch.use_deterministic_algorithms(True, warn_only=True)
    print(f"Deterministic mode enabled. Seed: {seed}")

set_seed(42)

# --- 1. THE LEGACY BRIDGE (ESSENTIAL) ---
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

import gluonts.torch.distributions as gtd
torch.serialization.add_safe_globals([gtd.studentT.StudentTOutput, gtd.StudentTOutput])

create_dummy_module('gluonts.torch.modules.loss')
class MockL:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, *args, **kwargs): return 0.0
    def __getattr__(self, name): return lambda *args, **kwargs: None
sys.modules['gluonts.torch.modules.loss'].DistributionLoss = MockL
sys.modules['gluonts.torch.modules.loss'].NegativeLogLikelihood = MockL
print("Legacy Compatibility Bridge: ONLINE")

from lag_llama.gluon.estimator import LagLlamaEstimator
from lag_llama.gluon.lightning_module import LagLlamaLightningModule
import gluonts.time_feature.lag as lag_mod

# --- 2. THE CUSTOM SLICING MODULE ---
class NetworkAnomalySpecialist(LagLlamaLightningModule):
    def training_step(self, batch, batch_idx):
        past_target = batch["past_target"] # [64, 832]

        distr_args, loc, scale = self.model(
            past_target=past_target,
            past_observed_values=batch["past_observed_values"]
        )

        target = past_target[:, -256:]
        loss = self.model.distr_output.loss(
            target,
            [arg[:, -256:] for arg in distr_args],
            loc[:, -256:],
            scale[:, -256:]
        ).mean()

        self.log("train_loss", loss, prog_bar=True)
        return loss

# --- 3. ARCHITECTURE DEFINITION ---
lags_seq_92 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
    47, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216, 228, 240, 252,
    264, 276, 288, 300, 312, 324, 336, 348, 360, 372, 384, 396, 408, 420, 432, 444, 456, 468,
    480, 492, 504, 516, 528, 540, 552, 564, 576
]

def run_finetuning():
    print("Loading telemetry data...")
    df = pd.read_csv("/root/lag-llama/clean_training_traffic_data.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    g_mean, g_std = df['traffic_volume_Tbits'].mean(), df['traffic_volume_Tbits'].std() + 1e-8
    df['scaled'] = (df['traffic_volume_Tbits'] - g_mean) / g_std

    from gluonts.dataset.pandas import PandasDataset
    from gluonts.itertools import Map

    base_dataset = PandasDataset.from_long_dataframe(
        df, target="scaled", item_id="flow_key_id", timestamp="timestamp", freq="5min"
    )

    def prepare_entry(entry):
        target = np.array(entry["target"], dtype=np.float32)
        if target.ndim > 1: target = target.squeeze()
        entry["target"] = target
        entry["observed_values"] = np.ones_like(target).astype(np.float32)
        return entry

    dataset = list(Map(prepare_entry, base_dataset))

    # --- 4. MODEL INITIALIZATION ---
    model = NetworkAnomalySpecialist(
        prediction_length=1,
        context_length=256,
        model_kwargs={
            "n_layer": 8, "n_head": 6, "n_embd_per_head": 24,
            "lags_seq": lags_seq_92, "context_length": 256,
            "max_context_length": 1024, "scaling": "mean",
            "input_size": 1, "distr_output": gtd.StudentTOutput(),
            "time_feat": False,
        }
    )

    # --- 5. WEIGHT GRAFTING ---
    print("Grafting foundation weights...")
    ckpt = torch.load("/root/lag-llama/lag-llama.ckpt", map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    model_wte_shape = model.model.transformer.wte.weight.shape
    ckpt_wte_shape = state_dict["model.transformer.wte.weight"].shape
    if model_wte_shape != ckpt_wte_shape:
        new_wte_weight = torch.zeros(model_wte_shape)
        new_wte_weight[:, :ckpt_wte_shape[1]] = state_dict["model.transformer.wte.weight"]
        state_dict["model.transformer.wte.weight"] = new_wte_weight

    model.load_state_dict(state_dict, strict=False)
    model.float()

    # --- 6. DATA LOADING ---
    lag_mod.to_offset = lambda x: pd.tseries.frequencies.to_offset('5min')

    est = LagLlamaEstimator(
        prediction_length=1, context_length=256, lags_seq=lags_seq_92,
        batch_size=64, n_layer=8, n_head=6, n_embd_per_head=24
    )

    transformation = est.create_transformation()
    training_loader = est.create_training_data_loader(dataset, transformation)

    # ADDED DETERMINISTIC FLAG
    trainer = lp.Trainer(
        max_epochs=50, 
        accelerator="gpu", 
        devices=1,
        deterministic=True, 
        default_root_dir="network_finetuning_v57"
    )

    print("--- STARTING SPECIALIZATION LOOP (Deterministic & Slicing) ---")
    trainer.fit(model, train_dataloaders=training_loader)

    torch.save(model.state_dict(), "specialized_network_llama2.pt")
    print("Success. Specialized model saved.")

if __name__ == "__main__":
    run_finetuning()
