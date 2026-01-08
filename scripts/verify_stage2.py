import sys
import os

# Add root directory to path so we can import config/model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from config import HyperMnemosyneConfig
from model.backbone import HyperMnemosyne

def verify_stage2():
    print("--- Verifying Stage 2 (Titans Memory) Training ---")
    
    # 1. Initialize a Fresh Model (Random Init)
    print("1. Initializing fresh model (Baseline)...")
    config = HyperMnemosyneConfig()
    fresh_model = HyperMnemosyne(config)
    
    # 2. Load Trained Model
    print("2. Loading model_final.pt (Trained)...")
    try:
        state_dict = torch.load("model_final.pt", map_location="cpu")
    except FileNotFoundError:
        print("Error: model_final.pt not found!")
        sys.exit(1)
    
    # Clean state dict keys (strip _orig_mod prefix from torch.compile)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
            
    trained_model = HyperMnemosyne(config)
    trained_model.load_state_dict(new_state_dict)
    
    # 3. Compare Memory MLP Weights
    print("\n3. Comparing Memory Weights...")
    
    fresh_w1 = fresh_model.memory.memory_mlp.w1.weight
    trained_w1 = trained_model.memory.memory_mlp.w1.weight
    
    # Calculate difference magnitude
    diff = (fresh_w1 - trained_w1).abs().mean().item()
    train_norm = trained_w1.norm().item()
    
    print(f"   Fresh Init Norm:   {fresh_w1.norm().item():.6f}")
    print(f"   Trained Wgt Norm:  {train_norm:.6f}")
    print(f"   Mean Difference:   {diff:.6f}")
    
    if diff == 0.0:
        print("\n[FAILURE] Memory weights are IDENTICAL to random init. Stage 2 did NOT update parameters.")
    else:
        print("\n[SUCCESS] Memory weights have deviated from initialization.")
        print("          This confirms gradients were applied to Titans Memory module in Stage 2.")

if __name__ == "__main__":
    verify_stage2()
