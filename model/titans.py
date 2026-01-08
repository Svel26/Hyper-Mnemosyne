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
        
        # Learnable or fixed decay? Let's use a fixed decay for stability first, or learnable sigmoid.
        # Simplified: Fixed decay 0.9
        self.decay = 0.9

    def forward(self, x, memory_state=None):
        """
        x: [B, S, D]
        memory_state: [B, S, D] (or [B, D] if handling recurrence differently, but here 
                       we assume simple token-wise temporal bias or sequence-level state)
                       
        For a true recurrence, it should be [B, D] and updated step-by-step. 
        But given we trained efficiently with parallelism, we likely want a "Global Context" state.
        
        Simplified Variant: Time-Mixing is handled by Mamba. 
        Titans Memory here acts as a "Long-Term Buffer".
        We will treat memory_state as a running average of the sequence.
        """
        if memory_state is None:
            memory_state = torch.zeros_like(x)
            
        # EMA Update: New State = 0.9 * Old + 0.1 * New
        # Note: This is a very simplified view. Genuine Titans uses attention.
        # This implementation matches the "Gated Residual" description in Blueprint.
        new_memory_state = self.decay * memory_state + (1 - self.decay) * x
        
        # Forward pass through the memory MLP using the STATE, not the raw input
        mem_out = self.memory_mlp(new_memory_state)
        
        # Surprise Loss (Reconstruction)
        # The memory should predict the current input from its state
        surprise_loss = F.mse_loss(mem_out, x.detach())
        
        return mem_out, surprise_loss, new_memory_state
