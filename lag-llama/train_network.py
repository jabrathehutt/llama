import sys
import subprocess
import os

def run_official_training():
    # 1. Prepare the environment variables
    env = os.environ.copy()
    # Add the current directory to PYTHONPATH so it finds 'data', 'utils', etc.
    env["PYTHONPATH"] = f"{os.getcwd()}:{env.get('PYTHONPATH', '')}"

    # 2. Define the arguments exactly as your pretrain.sh would
    # Note: We are using the 'python -c' trick to inject the distutils patch 
    # and then immediately import the official run script.
    
    cmd = [
        sys.executable, "-c",
        "import sys; from types import ModuleType; "
        "d=ModuleType('distutils'); sys.modules['distutils']=d; "
        "u=ModuleType('distutils.util'); sys.modules['distutils.util']=u; "
        "setattr(d,'util',u); u.strtobool=lambda v: 1 if v.lower() in ('y','yes','t','true','1') else 0; "
        "import run; run.train(run.parser.parse_args())",
        "-e", "network_specialization_v1",
        "-r", "experiments/results",
        "--single_dataset", "network_telemetry",
        "--dataset_path", "datasets",
        "--ckpt_path", "lag-llama.ckpt",
        "--batch_size", "64",
        "--max_epochs", "50",
        "--num_batches_per_epoch", "100",
        "--context_length", "32",
        "--lr", "0.0001",
        "--data_normalization", "mean",
        "--gpu", "0",
        "--wandb_mode", "disabled"
    ]

    print("--- Starting Patched Training Environment ---")
    print(f"Executing: {' '.join(cmd)}")
    
    # 3. Execute and stream output to the terminal
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user.")

if __name__ == "__main__":
    run_official_training()
