import torch
import os

MODEL_PATH = "real_world_network_llama.pt"

def probe_raw_checkpoint():
    if not os.path.exists(MODEL_PATH):
        print(f"File {MODEL_PATH} not found.")
        return

    print(f"Probing {MODEL_PATH} ({os.path.getsize(MODEL_PATH) / 1024 / 1024:.2f} MB)...")
    
    try:
        # Load the raw dictionary
        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        
        # Check if it's a Lightning checkpoint or a raw state_dict
        state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        
        print("\n--- Structural Analysis ---")
        print(f"Total keys found: {len(state_dict.keys())}")
        
        # Pick specific transformer layers to check for "life"
        # Using common Lag-Llama naming conventions
        test_keys = [
            "model.transformer.h.0.attn.c_attn.weight",
            "model.transformer.h.4.mlp.c_fc.weight",
            "model.transformer.wte.weight"
        ]
        
        print("\n--- Numerical Vital Signs ---")
        found_any = False
        for key in state_dict.keys():
            # Check a few layers to see if weights are updated or at zero
            if any(tk in key for tk in test_keys):
                found_any = True
                weights = state_dict[key]
                w_sum = weights.abs().sum().item()
                w_max = weights.max().item()
                w_min = weights.min().item()
                w_mean = weights.mean().item()
                
                print(f"Layer: {key}")
                print(f"  - Sum of Absolutes: {w_sum:.8f}")
                print(f"  - Range: [{w_min:.4f}, {w_max:.4f}]")
                print(f"  - Mean:  {w_mean:.8f}")
                
                if w_sum < 1e-6:
                    print("  [!] ALERT: This layer is DEAD (Zero weights).")
                elif abs(w_mean) < 1e-9 and w_sum > 0:
                    print("  [+] INFO: Weights are healthy (Centered around zero).")

        if not found_any:
            print("Could not find specific transformer keys. Listing first 5 keys found:")
            for k in list(state_dict.keys())[:5]:
                print(f"  - {k}")

    except Exception as e:
        print(f"Failed to probe: {e}")

if __name__ == "__main__":
    probe_raw_checkpoint()
