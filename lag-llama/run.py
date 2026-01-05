# --- 1. PYTHON 3.13 & GLUONTS COMPATIBILITY SHIM ---
import sys, types, pickle
d = types.ModuleType("distutils")
sys.modules["distutils"] = d
u = types.ModuleType("distutils.util")
sys.modules["distutils.util"] = u
setattr(d, "util", u)
u.strtobool = lambda v: 1 if str(v).lower() in ("y", "yes", "t", "true", "1") else 0

m = types.ModuleType("gluonts.torch.modules.loss")
sys.modules["gluonts.torch.modules.loss"] = m
class ML: pass
m.NegativeLogLikelihood = ML
m.DistributionLoss = ML

# --- 2. STANDARD IMPORTS ---
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

import argparse
import os
from hashlib import sha1
from pathlib import Path

import lightning
import torch
from lightning.pytorch.loggers import WandbLogger

from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.transform import AddObservedValuesIndicator, Chain

try:
    from data.data_utils import create_train_and_val_datasets_with_dates
    from utils.utils import set_seed
except ImportError:
    print("Error: Ensure you are running from the root of the Lag-Llama repo.")
    sys.exit(1)

from lag_llama.gluon.estimator import LagLlamaEstimator

# --- 3. THE TRAINING FUNCTION ---
def train(args):
    set_seed(args.seed)
    lightning.seed_everything(args.seed)

    fulldir_experiments = os.path.join(args.results_dir, args.experiment_name, str(args.seed))
    os.makedirs(fulldir_experiments, exist_ok=True)
    checkpoint_dir = os.path.join(fulldir_experiments, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger = WandbLogger(
        name=args.experiment_name + "-seed-" + str(args.seed),
        save_dir=fulldir_experiments,
        project=args.wandb_project,
        mode=args.wandb_mode
    )

    estimator = LagLlamaEstimator(
        prediction_length=args.prediction_length,
        context_length=args.context_length,
        input_size=1,
        batch_size=args.batch_size,
        n_layer=args.n_layer,
        n_embd_per_head=args.n_embd_per_head,
        n_head=args.n_head,
        scaling=args.data_normalization,
        lr=args.lr,
        lags_seq=args.lags_seq,
        num_batches_per_epoch=args.num_batches_per_epoch,
        trainer_kwargs=dict(
            max_epochs=args.max_epochs,
            accelerator="gpu",
            devices=[args.gpu],
            logger=logger,
            default_root_dir=fulldir_experiments,
        ),
    )

    # --- 4. DATA PREPARATION (PICKLE -> LISTDATASET) ---
    dataset_pkl = Path(args.dataset_path) / args.single_dataset / "processed_data.pkl"
    
    if dataset_pkl.exists():
        print(f"Loading processed dataset from: {dataset_pkl}")
        with open(dataset_pkl, "rb") as f:
            loaded_list = pickle.load(f)
        train_data = ListDataset(loaded_list, freq="5min")
        val_data = train_data 
    else:
        print("Processed data not found. Falling back to default loader.")
        history_length = estimator.context_length + max(estimator.lags_seq)
        (train_data, val_data, _, _, _, _, _, _) = create_train_and_val_datasets_with_dates(
            args.single_dataset, args.dataset_path, 0, history_length, args.prediction_length,
        )

    # --- 5. WEIGHT GRAFTING ---
    lightning_module = estimator.create_lightning_module()
    if args.ckpt_path and os.path.exists(args.ckpt_path):
        print(f"Grafting weights from {args.ckpt_path}...")
        checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model_state = lightning_module.state_dict()
        
        wte_key = "model.transformer.wte.weight"
        if wte_key in state_dict:
            ckpt_wte, curr_wte = state_dict[wte_key], model_state[wte_key]
            new_wte = torch.zeros_like(curr_wte)
            r, c = min(ckpt_wte.shape[0], curr_wte.shape[0]), min(ckpt_wte.shape[1], curr_wte.shape[1])
            new_wte[:r, :c] = ckpt_wte[:r, :c]
            state_dict[wte_key] = new_wte
            print(f"✅ Aligned WTE table: {ckpt_wte.shape} -> {curr_wte.shape}")

        lightning_module.load_state_dict(state_dict, strict=False)
        estimator.ckpt_path = None 

    # --- 6. START TRAINING ---
    transformation = Chain([
        AddObservedValuesIndicator(target_field=FieldName.TARGET, output_field=FieldName.OBSERVED_VALUES),
        estimator.create_transformation()
    ])

    training_loader = estimator.create_training_data_loader(
        train_data,
        transformation,
        **estimator.trainer_kwargs
    )

    trainer = lightning.Trainer(**estimator.trainer_kwargs)
    print("Starting specialized training loop...")
    trainer.fit(model=lightning_module, train_dataloaders=training_loader)

    final_save = os.path.join(checkpoint_dir, "network_specialized_final.pt")
    torch.save(lightning_module.state_dict(), final_save)
    print(f"DONE. Model saved at {final_save}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_name", type=str, required=True)
    parser.add_argument("-r", "--results_dir", type=str, required=True)
    parser.add_argument("-d", "--dataset_path", type=str, default="datasets")
    parser.add_argument("--single_dataset", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-m", "--max_epochs", type=int, default=50)
    parser.add_argument("-n", "--num_batches_per_epoch", type=int, default=100)
    parser.add_argument("--context_length", type=int, default=32)
    parser.add_argument("--prediction_length", type=int, default=1)
    parser.add_argument("--n_layer", type=int, default=8)
    parser.add_argument("--n_head", type=int, default=9)
    parser.add_argument("--n_embd_per_head", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data_normalization", default="mean")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--wandb_mode", type=str, default="offline")
    parser.add_argument("--wandb_project", type=str, default="network-telemetry")
    parser.add_argument("--lags_seq", type=int, nargs="+")
    args = parser.parse_args()
    train(args)
