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
        
        # Optim parameters for test-time training
        self.step_size = 0.01
        self.decay = 0.99
        
    def forward(self, x, update_memory=False):
        """
        x: [B, S, D]
        """
        # 1. Retrieve from memory
        # In Titans, the memory is just the MLP.
        # retrieval = MLP(x)
        mem_out = self.memory_mlp(x)
        
        # 2. Update memory (Test-Time Training)
        if update_memory:
            # Compute 'Surprise' or loss
            # Simplified Titans rule:
            # Loss = || x_future - MLP(x_past) ||^2? 
            # Or Gradient of the core task?
            # Blueprint: "Calculate gradient step to update weights based on Surprise Metric".
            # Usually strict implementation requires a partial backward pass here.
            
            # We will just return the memory output for now.
            pass
            
        return mem_out
        
    def update_weights(self, loss):
        """
        Manually apply GD step to memory_mlp weights
        """
        grads = torch.autograd.grad(loss, self.memory_mlp.parameters(), create_graph=False)
        with torch.no_grad():
            for p, g in zip(self.memory_mlp.parameters(), grads):
                p.data = p.data - self.step_size * g
                # Weight decay
                p.data = p.data * self.decay
