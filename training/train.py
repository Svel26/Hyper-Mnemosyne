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

def train(args):
    config = HyperMnemosyneConfig()
    
    # 1. Initialize Model
    print("Initializing Hyper-Mnemosyne...")
    model = HyperMnemosyne(config)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    
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
        
        # Zero grad
        for opt in optimizers:
            opt.zero_grad()
            
        # JEPA Hybrid Training Step
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
            # Optimize ONLY the Titans Memory 'Surprise'
            # Backbone is frozen (handled in optimizer setup)
            total_loss = mem_loss_ctx
            
            log_str = f"Step {step}: Memory Loss {total_loss.item():.4f}"
            
        # Backward
        total_loss.backward()

        # Step
        for opt in optimizers:
            opt.step()
        
        if step % 10 == 0:
            print(log_str)
            
        # Check Titans Memory update logic?
        # If in Stage 2, we would also run memory updates.
        # But this is the base loop.
        if step % 100 == 0:
             torch.save(model.state_dict(), f"model_step_{step}.pt")

    print("Saving final model...")
    torch.save(model.state_dict(), "model_final.pt")
    print("Model saved to model_final.pt")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=100)
    args = parser.parse_args()
    
    train(args)
