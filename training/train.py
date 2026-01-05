import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse

from config import HyperMnemosyneConfig
from model.backbone import HyperMnemosyne
from training.muon import Muon
from training.data_utils import create_dataloader

try:
    import bitsandbytes as bnb
    AdamW = bnb.optim.AdamW8bit
except ImportError:
    print("BitsAndBytes not found, falling back to torch.optim.AdamW")
    AdamW = torch.optim.AdamW

def get_param_groups(model):
    """
    Separate parameters into Matrix (2D) and Vector (1D) groups.
    Matrix -> Muon
    Vector -> 8-bit AdamW
    """
    matrix_params = []
    vector_params = []
    
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.ndim == 2:
                matrix_params.append(p)
            else:
                vector_params.append(p)
                
    return [
        {'params': matrix_params, 'optimizer': 'muon', 'lr': 0.02},
        {'params': vector_params, 'optimizer': 'adamw', 'lr': 0.001}
    ]

def save_checkpoint(model, step, keep=2):
    """
    Saves a checkpoint and rotates old ones to save disk space.
    """
    import glob
    import os
    
    filename = f"model_step_{step}.pt"
    print(f"Saving checkpoint to {filename}...")
    torch.save(model.state_dict(), filename)
    
    # Rotate
    checkpoints = sorted(glob.glob("model_step_*.pt"), key=os.path.getmtime)
    if len(checkpoints) > keep:
        for old_ckpt in checkpoints[:-keep]:
            print(f"Removing old checkpoint {old_ckpt}...")
            os.remove(old_ckpt)

def train(args):
    config = HyperMnemosyneConfig()
    if args.training_stage:
        config.training_stage = args.training_stage
    
    # 1. Initialize Model
    print("Initializing Hyper-Mnemosyne...")
    model = HyperMnemosyne(config)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Pretrained Weights
    if args.pretrained_path:
        print(f"Loading pretrained backbone from {args.pretrained_path}...")
        state_dict = torch.load(args.pretrained_path, map_location=model.embeddings.weight.device)
        model.load_state_dict(state_dict)
        
    # Freezing Logic for Stage 2
    if config.training_stage == "memory":
        print("Stage 2: Freezing backbone, training Titans Memory only.")
        for name, param in model.named_parameters():
            if "memory" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
    # 2. Variable Grouping
    param_groups = get_param_groups(model)
    
    # 3. Optimizers
    # Since we have different optimizers for different groups, we might need manual handling
    # or a wrapper. Pytorch doesn't natively support mixed optimizer classes easily in one step().
    # Standard practice: create two optimizers.
    
    matrix_params = [p for p in model.parameters() if p.ndim == 2 and p.requires_grad]
    vector_params = [p for p in model.parameters() if p.ndim != 2 and p.requires_grad]
    
    opt_muon = Muon(matrix_params, lr=0.02)
    opt_adam = AdamW(vector_params, lr=0.001)
    
    optimizers = [opt_muon, opt_adam]
    
    # Compile
    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # 4. Data
    dataloader = create_dataloader(args.data_dir, args.batch_size, config.max_seq_len)
    
    # 5. Loop
    model.train()
    print("Starting training...")
    
    for step, (input_ids, target_ids) in enumerate(dataloader):
        if step > args.max_steps:
            break
            
        input_ids = input_ids.to(model.embeddings.weight.device)
        target_ids = target_ids.to(model.embeddings.weight.device)
        
        # Zero grad only at start of accumulation cycle
        if step % args.grad_accum_steps == 0:
            for opt in optimizers:
                opt.zero_grad()
            
        # JEPA Hybrid Training Step
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # 1. Forward pass on Context
            logits_ctx, hidden_ctx, mem_loss_ctx = model(input_ids)
            
            # 2. Forward pass on Target
            logits_tgt, hidden_tgt, _ = model(target_ids)
            
            total_loss = 0
            
            if config.training_stage == "backbone":
                # Optimize Generation + JEPA
                # Titans memory loss is ignored or acts as auxiliary (optional)
                
                # A. Generative Loss
                loss_gen = nn.functional.cross_entropy(logits_tgt.view(-1, config.vocab_size), target_ids.view(-1))
                
                # B. JEPA Latent Loss
                pred_hidden = model.jepa_predictor(hidden_ctx)
                loss_jepa = nn.functional.mse_loss(pred_hidden, hidden_tgt.detach())
                
                total_loss = loss_gen + (config.jepa_weight * loss_jepa)
                
                log_str = f"Step {step}: Loss {total_loss.item():.4f} (Gen: {loss_gen.item():.4f}, JEPA: {loss_jepa.item():.4f})"
                
            elif config.training_stage == "memory":
                # Optimize Titans Memory Meta-Parameters
                # Backbone is frozen (handled in optimizer setup or via requires_grad)
                
                # Meta-Learning Protocol:
                # 1. Split sequence/batch into "Support" (A) and "Query" (B)
                # Since input_ids is [B, S], let's split along Sequence dimension for causality
                # S = 4096 -> S_support = 2048, S_query = 2048
                
                cutoff = input_ids.shape[1] // 2
                input_A = input_ids[:, :cutoff]
                input_B = input_ids[:, cutoff:]
                
                # --- Inner Loop Step (on Batch A) ---
                # Forward pass to get surprise on A
                _, _, loss_A = model(input_A)
                
                # Compute new weights for memory based on loss_A
                # This uses the current meta-params (step_size, decay)
                # We need access to the memory layer directly
                updated_weights = model.memory.get_updated_weights(loss_A)
                
                # --- Outer Loop Step (on Batch B) ---
                # Forward pass on B using the *updated* weights
                # This checks "Did the update using these meta-params actually help prediction/memory?"
                # We need to pass the functional weights to the model
                
                # Note: We need to modify model.forward to accept 'memory_params'
                # (which I handled in the previous tool call)
                _, _, loss_B = model(input_B, memory_params=updated_weights)
                
                total_loss = loss_B
                log_str = f"Step {step}: Meta-Loss {total_loss.item():.4f} (Inner Loss: {loss_A.item():.4f})"
                
            # Scale loss for gradient accumulation
            loss_scaled = total_loss / args.grad_accum_steps
            
        # Backward
        loss_scaled.backward()

        # Step
        if (step + 1) % args.grad_accum_steps == 0:
            for opt in optimizers:
                opt.step()
        
        if step % 10 == 0:
            print(log_str)
            
        # Check Titans Memory update logic?
        # If in Stage 2, we would also run memory updates.
        # But this is the base loop.
        if step % 100 == 0:
             save_checkpoint(model, step)

    print("Saving final model...")
    torch.save(model.state_dict(), "model_final.pt")
    print("Model saved to model_final.pt")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--pretrained_path", type=str, default=None, help="Path to pretrained backbone checkpoint")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for speedup")
    parser.add_argument("--training_stage", type=str, default=None, choices=["backbone", "memory"], help="Override config training stage")
    args = parser.parse_args()
    
    # Override config if argument is provided
    if args.training_stage:
        # We need to pass this to train() or handle it inside train()
        # Since train() creates a new config instance: config = HyperMnemosyneConfig()
        # We should modify train() to accept overrides or args.
        pass # Handling inside train() now.
    
    train(args)




