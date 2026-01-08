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
        self.decay = 0.9

    def forward(self, x, memory_state=None):
        """
        x: [B, S, D]
        memory_state: [B, D] (Previous step state for inference)
        """
        B, S, D = x.shape
        
        # --- PARALLEL MODE (Training / Prefill) ---
        if memory_state is None:
            # 1. Fast Parallel Decay (Vectorized)
            # Compute weights for cumulative sum: [decay^0, decay^1, ..., decay^S]
            # We want h_t = (1-decay) * sum_{i=0}^{t} decay^(t-i) * x_i
            
            indices = torch.arange(S, device=x.device)
            # diff[t, i] = t - i
            diff = indices[:, None] - indices[None, :]
            mask = diff >= 0
            
            # Decay Matrix: [S, S]
            decay_matrix = (self.decay ** diff.float()) * mask
            decay_matrix = decay_matrix * (1.0 - self.decay)
            
            # Apply: H = Matrix @ X
            # [S, S] @ [B, S, D] -> [B, S, D]
            # Cast to x dtype
            decay_matrix = decay_matrix.to(dtype=x.dtype)
            current_states = torch.einsum('ts, bsd -> btd', decay_matrix, x)
            
            # 2. Shift for "Read-Before-Write"
            # We need h_{t-1} to predict x_t.
            previous_states = torch.roll(current_states, shifts=1, dims=1)
            previous_states[:, 0, :] = 0.0 # t=0 has no history
            
            # 3. Output
            mem_out = self.memory_mlp(previous_states)
            
            # 4. FIX: Return the FINAL state for Prefill->Decode handoff
            # The state at the end of the block is the last element of current_states
            final_state = current_states[:, -1, :]
            
            surprise_loss = F.mse_loss(mem_out, x.detach())
            
            return mem_out, surprise_loss, final_state
            
        # --- STEP MODE (Inference Decoding) ---
        else:
            # memory_state is h_{t-1} [B, D]
            
            # 1. Read (Predict current x using old state)
            state_for_mlp = memory_state.unsqueeze(1) # [B, 1, D]
            mem_out = self.memory_mlp(state_for_mlp)
            
            # 2. Write (Update state with current x)
            # x is [B, 1, D]
            update_x = x.squeeze(1)
            new_memory_state = self.decay * memory_state + (1.0 - self.decay) * update_x
            
            surprise_loss = F.mse_loss(mem_out, x.detach())
            
            return mem_out, surprise_loss, new_memory_state
