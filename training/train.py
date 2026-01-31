import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import math
import sys
import os

# Add project root to path if not installed as package
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

def get_lr_schedule(step, warmup_steps=1000, max_steps=100000, base_lr=1.0):
    """
    Learning rate schedule with warmup and cosine decay.
    
    Args:
        step: Current training step
        warmup_steps: Number of warmup steps
        max_steps: Total training steps
        base_lr: Base learning rate (will be multiplied by this schedule)
    
    Returns:
        Learning rate multiplier (0.0 to 1.0)
    """
    if step < warmup_steps:
        # Linear warmup
        return float(step) / float(max(1, warmup_steps))
    else:
        # Cosine decay
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

def save_checkpoint(model, optimizers, step, keep=2):
    """
    Saves a checkpoint (model + optimizer states) and rotates old ones.
    """
    import glob
    import os
    
    filename = f"model_step_{step}.pt"
    print(f"💾 Saving checkpoint to {filename}...")
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'step': step,
        'optimizer_states': [opt.state_dict() for opt in optimizers]
    }
    
    torch.save(checkpoint, filename)
    
    # Rotate old checkpoints
    checkpoints = sorted(glob.glob("model_step_*.pt"), key=os.path.getmtime)
    if len(checkpoints) > keep:
        for old_ckpt in checkpoints[:-keep]:
            print(f"🗑️  Removing old checkpoint {old_ckpt}...")
            os.remove(old_ckpt)

def train(args):
    config = HyperMnemosyneConfig()
    if args.training_stage:
        config.training_stage = args.training_stage
    
    print(f"🚀 Training Stage: {config.training_stage}")
    
    # 1. Initialize Model
    print("🏗️  Initializing Hyper-Mnemosyne...")
    model = HyperMnemosyne(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    print(f"📊 Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📊 Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Auto-Resume Logic
    import glob
    import os
    import re
    
    start_step = 0
    checkpoints = glob.glob("model_step_*.pt")
    
    # Priority 1: Explicit Pretrained Path (e.g. for Stage 2)
    if args.pretrained_path:
        print(f"📥 Loading pretrained backbone from {args.pretrained_path}...")
        checkpoint = torch.load(args.pretrained_path, map_location=device)
        
        # Handle both direct state_dict and checkpoint format
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Clean _orig_mod prefix from torch.compile
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                new_state_dict[k[10:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        print("✅ Loaded pretrained weights")
        
    # Priority 2: Auto-Resume from latest checkpoint
    elif checkpoints:
        steps = []
        for ckpt in checkpoints:
            match = re.search(r"model_step_(\d+).pt", ckpt)
            if match:
                steps.append(int(match.group(1)))
        
        if steps:
            max_step = max(steps)
            latest_ckpt = f"model_step_{max_step}.pt"
            print(f"📥 Resuming from latest checkpoint: {latest_ckpt} (Step {max_step})")
            
            try:
                checkpoint = torch.load(latest_ckpt, map_location=device)
                
                # Clean state dict keys
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('_orig_mod.'):
                        new_state_dict[k[10:]] = v
                    else:
                        new_state_dict[k] = v
                
                model.load_state_dict(new_state_dict, strict=True)
                start_step = max_step
                print(f"✅ Resumed from step {start_step}")
            except Exception as e:
                print(f"⚠️  Failed to load checkpoint: {e}")
                start_step = 0
    
    # Stage-specific parameter freezing
    if config.training_stage == "memory":
        print("🔒 Stage 2: Freezing backbone, training Titans Memory only")
        for name, param in model.named_parameters():
            if "memory" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif config.training_stage == "backbone":
        print("🔓 Stage 1: Training backbone (Memory active but not trainable)")
        # Memory is initialized and participates in forward pass,
        # but parameters are not updated
        for name, param in model.named_parameters():
            if "memory" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    # 2. Setup Optimizers with correct learning rates
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
    base_lrs = []
    
    if matrix_params:
        # FIX: Reduced Muon LR from 0.02 to 0.01 for stability
        muon_lr = 0.01
        opt_muon = Muon(matrix_params, lr=muon_lr)
        optimizers.append(opt_muon)
        base_lrs.append(muon_lr)
        print(f"⚙️  Muon optimizer: {len(matrix_params)} param groups, LR={muon_lr}")
    
    if vector_params:
        adamw_lr = 0.001
        opt_adam = AdamW(vector_params, lr=adamw_lr)
        optimizers.append(opt_adam)
        base_lrs.append(adamw_lr)
        print(f"⚙️  AdamW optimizer: {len(vector_params)} param groups, LR={adamw_lr}")
    
    # Compile model (optional but recommended)
    if args.compile:
        print("🔥 Compiling model with torch.compile...")
        model = torch.compile(model)

    # 3. Load Optimizer State if Resuming
    if start_step > 0 and checkpoints:
        latest_ckpt = f"model_step_{start_step}.pt"
        try:
            checkpoint = torch.load(latest_ckpt, map_location=device)
            if 'optimizer_states' in checkpoint:
                print("📥 Loading optimizer states...")
                opt_states = checkpoint['optimizer_states']
                if len(opt_states) == len(optimizers):
                    for opt, state in zip(optimizers, opt_states):
                        try:
                            opt.load_state_dict(state)
                        except Exception as e:
                            print(f"⚠️  Failed to load state for optimizer: {e}")
                else:
                    print(f"⚠️  Optimizer count mismatch. Skipping optimizer resume.")
        except Exception as e:
            print(f"⚠️  Error loading optimizer state: {e}")

    # 4. Data Loading
    print(f"📂 Loading data from {args.data_dir}...")
    dataloader = create_dataloader(
        args.data_dir, 
        args.batch_size, 
        config.max_seq_len,
        num_workers=4,  # Multi-worker support
        stage=config.training_stage
    )
    
    # 5. Training Loop
    model.train()
    print(f"🏃 Starting training from step {start_step}...")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Gradient accumulation: {args.grad_accum_steps}")
    print(f"   Effective batch size: {args.batch_size * args.grad_accum_steps}")
    print(f"   Max steps: {args.max_steps}")
    print(f"   Warmup steps: {args.warmup_steps}")
    
    use_amp = torch.cuda.is_available()
    
    for step, (input_ids, target_ids) in enumerate(dataloader, start=start_step):
        if step >= args.max_steps:
            break
            
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        # Update learning rates with schedule
        lr_mult = get_lr_schedule(step, args.warmup_steps, args.max_steps)
        for opt, base_lr in zip(optimizers, base_lrs):
            for param_group in opt.param_groups:
                param_group['lr'] = base_lr * lr_mult
        
        # Zero grad at start of accumulation cycle
        if step % args.grad_accum_steps == 0:
            for opt in optimizers:
                opt.zero_grad()
            
        # === TRAINING LOGIC (STAGE-SPECIFIC) ===
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            
            if config.training_stage == "backbone":
                # === STAGE 1: Backbone + JEPA Training ===
                
                # FIX: Simplified JEPA - single pass with stop-gradient
                # Forward on MASKED input
                logits_masked, hidden_masked, mem_loss, _ = model(input_ids)
                
                # A. Generative Loss (Next Token Prediction)
                shift_logits = logits_masked[..., :-1, :].contiguous()
                shift_labels = target_ids[..., 1:].contiguous()
                
                loss_gen = nn.functional.cross_entropy(
                    shift_logits.view(-1, config.vocab_size), 
                    shift_labels.view(-1), 
                    ignore_index=-100
                )
                
                # B. JEPA Latent Loss (Predict Clean from Masked)
                # Get clean hidden states with stop-gradient
                with torch.no_grad():
                    _, hidden_clean, _, _ = model(target_ids)
                
                # Predict clean from masked
                pred_hidden = model.jepa_predictor(hidden_masked)
                loss_jepa = nn.functional.mse_loss(pred_hidden, hidden_clean)
                
                total_loss = loss_gen + (config.jepa_weight * loss_jepa)
                
                log_str = f"Step {step}: Loss {total_loss.item():.4f} (Gen: {loss_gen.item():.4f}, JEPA: {loss_jepa.item():.4f})"
                
            elif config.training_stage == "memory":
                # === STAGE 2: Memory Training ===
                # FIX: Use CLEAN inputs in Stage 2, not masked
                
                # Forward on CLEAN input (target_ids, not input_ids)
                logits, hidden, mem_loss, _ = model(target_ids)
                
                # Generative loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = target_ids[..., 1:].contiguous()
                  
                loss_gen = nn.functional.cross_entropy(
                    shift_logits.view(-1, config.vocab_size), 
                    shift_labels.view(-1), 
                    ignore_index=-100
                )
                
                # FIX: Weight memory loss appropriately (typically 100-1000x smaller)
                memory_loss_weight = 0.01
                total_loss = loss_gen + (memory_loss_weight * mem_loss)
                
                log_str = f"Step {step}: Loss {total_loss.item():.4f} (Gen: {loss_gen.item():.4f}, Mem: {mem_loss.item():.4f})"
        
        # FIX: Don't scale loss for gradient accumulation
        # Let gradients accumulate naturally
        total_loss.backward()
        
        # Gradient accumulation step
        if (step + 1) % args.grad_accum_steps == 0:
            # FIX: Add gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            for opt in optimizers:
                opt.step()
        
        # Logging
        if step % 10 == 0:
            # Calculate gradient norm for monitoring
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            
            print(f"{log_str}, LR: {lr_mult:.4f}, Grad: {total_norm:.2f}")
            
        # Save checkpoint
        if (step > start_step) and (step % args.save_every == 0):
            save_checkpoint(model, optimizers, step)

    print("✅ Training complete!")
    print("💾 Saving final model...")
    torch.save(model.state_dict(), "model_final.pt")
    print("🎉 Model saved to model_final.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--training_stage", type=str, default=None, choices=["backbone", "memory"])
    args = parser.parse_args()
    
    train(args)
