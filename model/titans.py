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
        
        # Meta-Parameters for test-time training
        # Initialize step_size to a small value, sigmoid will constrain it to (0, 1) or similar range
        # Using softplus or simple parameter for now.
        self.step_size = nn.Parameter(torch.tensor(0.01))
        self.decay = nn.Parameter(torch.tensor(0.99))
        
    def forward(self, x, memory_params=None):
        """
        x: [B, S, D]
        memory_params: Optional dict of weights for the memory MLP (if using updated weights)
        """
        # Functional call if params provided
        if memory_params is not None:
             # This requires torch > 2.0 ideally, or manual functional application.
             # MemoryMLP is simple: w2(act(w1(x)))
             # Let's unpack manually for clarity and stability across versions
             w1 = memory_params['w1']
             w2 = memory_params['w2']
             act = nn.functional.silu
             
             hidden = act(torch.matmul(x, w1.t()))
             mem_out = torch.matmul(hidden, w2.t())
        else:
             mem_out = self.memory_mlp(x)
        
        # Surprise Loss (Reconstruction)
        surprise_loss = F.mse_loss(mem_out, x.detach())
        
        return mem_out, surprise_loss
        
    def get_updated_weights(self, loss, current_params=None):
        """
        Differentiable weight update step.
        Returns new weights without modifying in-place (unless we want to).
        For meta-learning, we need the new weights to be a function of the old weights + meta-params.
        """
        if current_params is None:
            # Use current model weights
            ws = {'w1': self.memory_mlp.w1.weight, 'w2': self.memory_mlp.w2.weight}
        else:
            ws = current_params
            
        # Calculate gradients of the loss w.r.t weights
        # We need create_graph=True if we want to differentiate through this step later w.r.t step_size
        grads = torch.autograd.grad(loss, ws.values(), create_graph=True)
        
        new_weights = {}
        for (name, p), g in zip(ws.items(), grads):
            # SGD Step: p_new = p - step_size * g
            # Decay: p_new = p_new * decay
            
            # Note: step_size and decay are learnable parameters
            new_p = p - self.step_size * g
            new_p = new_p * self.decay
            new_weights[name] = new_p
            
        return new_weights

