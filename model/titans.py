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
        # --- TRAINING MODE (Parallel-ish via Loop needed for correct time-mixing) ---
        if memory_state is None:
            B, S, D = x.shape
            # Initialize hidden state (t=-1)
            h = torch.zeros(B, D, device=x.device, dtype=x.dtype)
            
            states = []
            # Sequential Scan (loop over time)
            # h_t = decay * h_{t-1} + (1-decay) * x_t
            for t in range(S):
                h = self.decay * h + (1 - self.decay) * x[:, t, :]
                states.append(h)
                
            # Stack to get [B, S, D]
            new_memory_state = torch.stack(states, dim=1)
            
        # --- INFERENCE MODE (Step-by-Step) ---
        else:
             # memory_state passed in is [B, D] from previous step
             # But wait, in inference loop we pass 'titans_state' which is likely the LAST state.
             # However, backbone.py passes 'titans_state' which we return as new_memory_state.
             # Let's assume memory_state here is [B, D] (the "h" from previous step).
             
             # But if input is [B, 1, D], we do one update.
             # new_state = decay * old + (1-decay) * x
             new_memory_state = self.decay * memory_state + (1 - self.decay) * x.squeeze(1)
             
             # Restore dimension for output [B, 1, D]
             new_memory_state = new_memory_state.unsqueeze(1)
        
        # Forward pass through the memory MLP using the STATE, not the raw input
        mem_out = self.memory_mlp(new_memory_state)
        
        # Surprise Loss (Reconstruction)
        # The memory should predict the current input from its state
        surprise_loss = F.mse_loss(mem_out, x.detach())
        
        if memory_state is not None:
             # Return state as [B, D] for next inference step
             return_state = new_memory_state.squeeze(1)
        else:
             # Return full sequence [B, S, D] (though backbone might not use it, useful for debug)
             # Actually backbone expects 'new_titans_state' to be passed back in inference loop.
             # If we are in training, we don't care about return state.
             return_state = new_memory_state
             
        return mem_out, surprise_loss, return_state
