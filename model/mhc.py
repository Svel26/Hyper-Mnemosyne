import torch
import torch.nn as nn
from .triton_kernels import mhc_forward

class MHC(nn.Module):
    def __init__(self, d_model, n_branches=4):
        super().__init__()
        self.d_model = d_model
        self.n_branches = n_branches
        
        # Learnable mixing matrix (unconstrained logits)
        # We initialize it close to identity to start with stable flow? 
        # Or random? DeepSeek paper implies identity initialization is good for residuals.
        # Let's start with a scaled random initialization + identity bias.
        self.w = nn.Parameter(torch.randn(n_branches, n_branches) * 0.01)
        
        # Bias the diagonal to encourage identity flow initially
        with torch.no_grad():
            self.w.data.fill_diagonal_(1.0)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Seq, n_branches, d_model)
        Returns:
            Mixed tensor of shape (Batch, Seq, n_branches, d_model)
        """
        # x is expected to be [B, S, N, D]
        # We invoke the kernel (currently PyTorch ref)
        return mhc_forward(x, self.w)
