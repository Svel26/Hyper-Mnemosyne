import torch
import torch.optim as optim

def zeroth_power_via_newton_schulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power of a matrix G.
    Returns UV^T where G = USV^T. 
    Effectively orthogonalizes the matrix.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    # Normalize spectral norm
    # Estimate spectral norm via Frobenius norm? Or iterations?
    # Simple strategy: divide by Fro norm (approx)
    X = X / (X.norm() + eps) 
    
    # In practice, for Muon, we might assume X is already conditioned or just run iterations
    # Paper suggests scaling such that singular values are close to 1.
    
    if G.size(0) > G.size(1):
        X = X.T
        
    # Newton-Schulz loop
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A 
        X = a * X + B @ X
        
    if G.size(0) > G.size(1):
        X = X.T
        
    return X

class Muon(optim.Optimizer):
    """
    Muon - Momentum Orthogonalized Optimizer.
    Designed for 2D parameters (matrices).
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Muon does not support sparse gradients')
                
                # Check for 2D
                if len(p.shape) != 2:
                    # Fallback or error? Blueprint says use Muon for matrices only.
                    # We assume user filters params correctly.
                    # Or we treat as 1D vector learning (standard SGD/Adam?)
                    # For now, just apply standard SGD-like update or skip.
                    continue
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    # Momentum buffer in BFloat16
                    state['momentum_buffer'] = torch.zeros_like(p, dtype=torch.bfloat16)
                    
                buf = state['momentum_buffer']
                
                # Update momentum
                buf.mul_(momentum).add_(grad, alpha=1.0)
                
                if nesterov:
                    update = grad + momentum * buf
                else:
                    update = buf
                    
                # Orthogonalize update using Newton-Schulz
                # This is the "Muon" magic
                update_ortho = zeroth_power_via_newton_schulz5(update, steps=ns_steps)
                
                # Apply update
                # p -= lr * update_ortho * scaling_factor? 
                # Theoretically Muon scales updates by spectral radius.
                # Simplified:
                
                # Scale ref (RMS of params) to match RMS of update?
                # Usually Muon replaces the update matrix with its orthogonal projection * learning_rate
                
                p.data.add_(update_ortho, alpha=-lr)
                
        return loss
