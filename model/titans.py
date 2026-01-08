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
        memory_state: [B, S, D]
        """
        # --- TRAINING MODE (Parallel-ish via Loop needed for correct time-mixing) ---
        if memory_state is None:
            B, S, D = x.shape
            
            # Use JIT-friendly helper for the loop to compute h_{t-1} for all t
            new_memory_state = self._scan(x, self.decay)
            
            # Predict x_t using h_{t-1}
            mem_out = self.memory_mlp(new_memory_state)
            
            # Surprise Loss (Reconstruction)
            surprise_loss = F.mse_loss(mem_out, x.detach())
            
            return mem_out, surprise_loss, None # Don't need state return for training
            
        # --- INFERENCE MODE (Step-by-Step) ---
        else:
             # memory_state passed in is [B, D] from previous step (h_{t-1})
             # We use h_{t-1} to predict x_t.
             # So we use memory_state AS IS for the MLP input.
             
             # Current input for MLP: memory_state (which is h_{t-1})
             state_for_mlp = memory_state.unsqueeze(1)
             
             # Update State: h_t = decay * h_{t-1} + (1-decay) * x_t
             new_memory_state = self.decay * memory_state + (1 - self.decay) * x.squeeze(1)
             
             # Prepare for return (restore dimensions)
             new_memory_state = new_memory_state.unsqueeze(1)
             
             # MLP Input is PRE-UPDATE state
             mem_out = self.memory_mlp(state_for_mlp)
             
             # Surprise Loss
             surprise_loss = F.mse_loss(mem_out, x.detach())
             
             return mem_out, surprise_loss, new_memory_state.squeeze(1)

    @torch.jit.export
    def _scan(self, x: torch.Tensor, decay: float) -> torch.Tensor:
        B, S, D = x.shape
        h = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        states = []
        
        # Sequential Scan
        for t in range(S):
            # 1. READ (Store previous state h_{t-1})
            # For t=0, h is 0. Correct.
            states.append(h)
            
            # 2. WRITE (Update to h_t)
            # h_t = decay * h_{t-1} + (1-decay) * x_t
            # Note: x[:, t, :] is [B, D]
            h = decay * h + (1 - decay) * x[:, t, :]
            
        # Stack -> [B, S, D]
        return torch.stack(states, dim=1)
