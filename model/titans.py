import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.memory_width, bias=False)
        self.w2 = nn.Linear(config.memory_width, config.d_model, bias=False)
        self.activation = nn.SiLU()
        
    def forward(self, x):
        return self.w2(self.activation(self.w1(x)))

class TitansMemoryLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.memory_mlp = MemoryMLP(config)
        
        # Note: Meta-learning parameters (step_size, decay) removed
        # as we are using a simplified Gated Residual / Auxiliary Loss approach.
        
    def forward(self, x):
        """
        x: [B, S, D]
        """
        # Forward pass through the memory MLP
        mem_out = self.memory_mlp(x)
        
        # Surprise Loss (Reconstruction)
        # This acts as an auxiliary objective: the memory should predict the current state
        surprise_loss = F.mse_loss(mem_out, x.detach())
        
        return mem_out, surprise_loss
