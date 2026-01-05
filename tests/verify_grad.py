import torch
from model.mhc import MHC

def test_mhc_grad():
    print("Testing MHC Gradient Flow...")
    
    # 1. Setup
    B, S, N, D = 2, 16, 4, 32
    model = MHC(d_model=D, n_branches=N).cuda()
    
    x = torch.randn(B, S, N, D, device='cuda', requires_grad=True)
    
    # 2. Forward
    print("Forward pass...")
    out = model(x)
    
    # 3. Loss
    loss = out.sum()
    
    # 4. Backward
    print("Backward pass...")
    loss.backward()
    
    # 5. Check Gradients
    print(f"Input X grad: {x.grad is not None}")
    if x.grad is not None:
        print(f"Input X grad norm: {x.grad.norm().item()}")
        
    print(f"Weight W grad: {model.w.grad is not None}")
    if model.w.grad is not None:
        print(f"Weight W grad norm: {model.w.grad.norm().item()}")
        
    if x.grad is not None and model.w.grad is not None and model.w.grad.norm() > 0:
        print("SUCCESS: Gradients are flowing!")
    else:
        print("FAILURE: Gradients missing or zero.")

if __name__ == "__main__":
    if torch.cuda.is_available():
        test_mhc_grad()
    else:
        print("Skipping Triton test (No CUDA)")
