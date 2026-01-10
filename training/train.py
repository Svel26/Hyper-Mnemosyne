import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse

from config import HyperMnemosyneConfig
# from model.backbone import HyperMnemosyne # Use the updated one
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
    Excludes Embeddings and Head from Muon as per paper.
    """
    matrix_params = []
    vector_params = []
    
    # Identify excluded modules
    # We want to exclude the embedding layer weights and the lm_head weights from Muon
    # Even if they are 2D.
    excluded_param_ids = set()
    excluded_param_ids.add(id(model.embeddings.weight))
    excluded_param_ids.add(id(model.lm_head.weight))
    
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.ndim == 2 and id(p) not in excluded_param_ids:
                matrix_params.append(p)
            else:
                vector_params.append(p)
                
    return [
        {'params': matrix_params, 'optimizer': 'muon', 'lr': 0.02},
        {'params': vector_params, 'optimizer': 'adamw', 'lr': 0.001}
    ]

def save_checkpoint(model, optimizers, step, keep=2):
    """
    Saves a checkpoint (model + optimizer states) and rotates old ones.
    """
    import glob
    import os
    
    filename = f"model_step_{step}.pt"
    print(f"Saving checkpoint to {filename}...")
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'step': step,
        'optimizer_states': [opt.state_dict() for opt in optimizers]
    }
    
    torch.save(checkpoint, filename)
    
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Auto-Resume Logic
    import glob
    import os
    import re
    
    start_step = 0
    checkpoints = glob.glob("model_step_*.pt")
    
    # Priority 1: Explicit Pretrained Path (e.g. for Stage 2)
    if args.pretrained_path:
        print(f"Loading pretrained backbone from {args.pretrained_path}...")
        state_dict = torch.load(args.pretrained_path, map_location=device)
        model.load_state_dict(state_dict, strict=False) # strict=False for flexibility
        
    # Priority 2: Auto-Resume from latest checkpoint (Higher priority if exists and matches stage)
    elif checkpoints:
        # Extract steps
        steps = []
        for ckpt in checkpoints:
            match = re.search(r"model_step_(\d+).pt", ckpt)
            if match:
                steps.append(int(match.group(1)))
        
        if steps:
            max_step = max(steps)
            latest_ckpt = f"model_step_{max_step}.pt"
            print(f"Resuming from latest checkpoint: {latest_ckpt} (Step {max_step})")
            
            try:
                checkpoint = torch.load(latest_ckpt, map_location=device)
                
                # Handle legacy checkpoints (just model weights)
                if 'model_state_dict' in checkpoint:
                     model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                     # Optimizer state loading happens after optimizer creation
                else:
                     # Old format
                     model.load_state_dict(checkpoint, strict=False)
                
                start_step = max_step
            except Exception as e:
                print(f"Failed to load checkpoint {latest_ckpt}: {e}")
                start_step = 0
        

        
    # Freezing Logic for Stage 2
    if config.training_stage == "memory":
        print("Stage 2: Freezing backbone, training Titans Memory only.")
        for name, param in model.named_parameters():
            if "memory" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
    # 2. Variable Grouping
    # get_param_groups returns dicts, need to unpack for separate optimizers manually if we want different classes
    # Or just use the lists created inside
    
    matrix_params = []
    vector_params = []
    excluded_param_ids = set()
    excluded_param_ids.add(id(model.embeddings.weight))
    excluded_param_ids.add(id(model.lm_head.weight))
    
    for name, p in model.named_parameters():
         if p.requires_grad:
            if p.ndim == 2 and id(p) not in excluded_param_ids:
                matrix_params.append(p)
            else:
                vector_params.append(p)

    optimizers = []
    if matrix_params:
        opt_muon = Muon(matrix_params, lr=0.02)
        optimizers.append(opt_muon)
    
    if vector_params:
        try:
            opt_adam = AdamW(vector_params, lr=0.001)
            optimizers.append(opt_adam)
        except ValueError:
            # Still possible if vector_params is empty, though the check above handles mostly
            # but let's be safe
            print("Warning: No vector parameters found for AdamW.")
            pass
    
    # Compile
    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # 3b. Load Optimizer State if Resuming
    if start_step > 0 and checkpoints:
        latest_ckpt = f"model_step_{start_step}.pt"
        try:
            checkpoint = torch.load(latest_ckpt, map_location=device)
            if 'optimizer_states' in checkpoint:
                 print("Loading optimizer states...")
                 opt_states = checkpoint['optimizer_states']
                 if len(opt_states) == len(optimizers):
                     for opt, state in zip(optimizers, opt_states):
                         opt.load_state_dict(state)
                 else:
                     print("Warning: Optimizer count mismatch, skipping state load.")
        except Exception as e:
            print(f"Error loading optimizer state: {e}")

    # 4. Data
    dataloader = create_dataloader(args.data_dir, args.batch_size, config.max_seq_len)
    
    # 5. Loop
    model.train()
    print("Starting training...")
    
    use_amp = torch.cuda.is_available()
    
    for step, (input_ids, target_ids) in enumerate(dataloader, start=start_step):
        if step > args.max_steps:
            break
            
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        # Zero grad only at start of accumulation cycle
        if step % args.grad_accum_steps == 0:
            for opt in optimizers:
                opt.zero_grad()
            
        # JEPA Hybrid Training Step
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            # 1. Forward pass on Context
            logits_ctx, hidden_ctx, mem_loss_ctx, _ = model(input_ids)
            
            # 2. Forward pass on Target
            logits_tgt, hidden_tgt, _, _ = model(target_ids)
            
            total_loss = 0
            
            if config.training_stage == "backbone":
                # A. Generative Loss (Next Token Prediction)
                # Shift logits and labels
                # logits: [B, S, V]
                # targets: [B, S]
                
                shift_logits = logits_tgt[..., :-1, :].contiguous()
                shift_labels = target_ids[..., 1:].contiguous()
                
                loss_gen = nn.functional.cross_entropy(shift_logits.view(-1, config.vocab_size), shift_labels.view(-1), ignore_index=50256)
                
                # B. JEPA Latent Loss
                # We predict hidden_tgt from hidden_ctx using jepa_predictor. 
                # Crucially: STOP GRADIENT on hidden_tgt to prevent collapse.
                pred_hidden = model.jepa_predictor(hidden_ctx)
                loss_jepa = nn.functional.mse_loss(pred_hidden, hidden_tgt.detach())
                
                total_loss = loss_gen + (config.jepa_weight * loss_jepa)
                
                log_str = f"Step {step}: Loss {total_loss.item():.4f} (Gen: {loss_gen.item():.4f}, JEPA: {loss_jepa.item():.4f})"
                
            elif config.training_stage == "memory":
                # Optimize Titans Memory Meta-Parameters
                # Simplified protocol: Training memory to minimize surprise on future tokens?
                # Or just standard causal loss with memory enabled?
                # The original "meta-learning" loop was broken.
                # Here we just train the memory params on the standard loss.
                
                # We reuse the generative loss from above but only backprop into memory params
                shift_logits = logits_ctx[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                 
                loss_gen = nn.functional.cross_entropy(shift_logits.view(-1, config.vocab_size), shift_labels.view(-1), ignore_index=50256)
                
                # We can also use the explicit memory_loss (reconstruction) if returned
                # memory_loss from forward is "surprise"
                
                total_loss = loss_gen + mem_loss_ctx
                log_str = f"Step {step}: Mem-Loss {total_loss.item():.4f} (Gen: {loss_gen.item():.4f}, Surprise: {mem_loss_ctx.item():.4f})"
                
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
            
        if step % 100 == 0:
             save_checkpoint(model, optimizers, step)

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
    
    train(args)




