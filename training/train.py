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
            
        # Forward
        logits = model(input_ids)
        
        # Loss
        loss = nn.functional.cross_entropy(logits.view(-1, config.vocab_size), target_ids.view(-1))
        
        # Backward
        loss.backward()
        
        # Step
        for opt in optimizers:
            opt.step()
            
        if step % 10 == 0:
            print(f"Step {step}: Loss {loss.item():.4f}")
            
        # Check Titans Memory update logic?
        # If in Stage 2, we would also run memory updates.
        # But this is the base loop.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=100)
    args = parser.parse_args()
    
    train(args)
