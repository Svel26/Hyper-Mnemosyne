import torch
import pytest
import math
from model.triton_kernels import mhc_forward, sinkhorn_kernel, fused_mhc_mixing_kernel

def torch_sinkhorn(w, n_iters=5):
    """Reference PyTorch implementation of Log-Space Sinkhorn"""
    # w: [N, N] in log space
    # Output: P = exp(Sinkhorn(w))
    
    # Clone to avoid modifying input
    w = w.clone()
    
    for _ in range(n_iters):
        # Row norm: w = w - logsumexp(w, dim=1)
        w = w - torch.logsumexp(w, dim=1, keepdim=True)
        # Col norm: w = w - logsumexp(w, dim=0)
        w = w - torch.logsumexp(w, dim=0, keepdim=True)
        
    return torch.exp(w)

def torch_mhc_mixing(x, w, n_iters=5):
    """Reference PyTorch implementation of MHC Mixing"""
    # x: [B, S, N, D]
    # w: [N, N]
    
    P = torch_sinkhorn(w, n_iters)
    # y = P @ x
    y = torch.einsum('oi,bsid->bsod', P, x)
    return y

@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("S", [8, 32])
@pytest.mark.parametrize("D", [64, 128])
def test_sinkhorn_kernel_match(B, S, D):
    """Test if Triton Sinkhorn kernel matches PyTorch reference"""
    device = "cuda"
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    N = 4 # Hardcoded for now
    torch.manual_seed(42)
    
    # Random weights
    w = torch.randn(N, N, device=device, dtype=torch.float32)
    
    # PyTorch Ref
    p_ref = torch_sinkhorn(w, n_iters=5)
    
    # Triton Kernel
    p_triton = torch.empty_like(w)
    # Note: Kernel modifies w in place potentially? No, I implemented it to load from W and store to P
    # But wait, my kernel implementation:
    # tl.store(W_ptr + ..., ...) -> updates W in global memory!
    # So I must pass a copy of W to the kernel if I want to preserve original W?
    # Yes, the kernel updates W in place for the iterations.
    
    w_copy = w.clone()
    sinkhorn_kernel[(1,)](w_copy, p_triton, N=N, n_iters=5)
    
    # Check numerical closeness
    # Sinkhorn is sensitive, but with float32 should be close
    assert torch.allclose(p_ref, p_triton, atol=1e-4), "Sinkhorn output mismatch"

@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("S", [8, 32])
@pytest.mark.parametrize("D", [64, 128])
def test_mhc_mixing_match(B, S, D):
    """Test if fused MHC mixing matches PyTorch reference"""
    device = "cuda"
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    N = 4
    torch.manual_seed(42)
    
    x = torch.randn(B, S, N, D, device=device)
    w = torch.randn(N, N, device=device)
    
    # PyTorch Ref
    y_ref = torch_mhc_mixing(x, w, n_iters=5)
    
    # Triton Wrapper (which calls kernel)
    y_triton = mhc_forward(x, w)
    
    assert torch.allclose(y_ref, y_triton, atol=1e-3), "MHC Mixing output mismatch"
