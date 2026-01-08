import torch
import triton
import triton.language as tl
from torch.autograd import Function

# -----------------------------------------------------------------------------
# TRITON KERNELS
# -----------------------------------------------------------------------------

@triton.jit
def sinkhorn_kernel(
    w_ptr,
    out_ptr,
    N: tl.constexpr,
    n_iters: tl.constexpr
):
    """
    Computes Sinkhorn normalization for N=4 matrix.
    W is (N, N). Flatted.
    """
    # Load W [16]
    w_offsets = tl.arange(0, 16)
    w_flat = tl.load(w_ptr + w_offsets)
    
    # View as [4, 4] for efficient reduction matches N=4 constraint
    w = tl.reshape(w_flat, (4, 4))
    
    # Sinkhorn Iterations (Log Space)
    for _ in range(n_iters):
        # --- Row Norm ---
        # LogSumExp over rows (axis 1)
        # We compute max first for numerical stability (optional in log space pure subtraction but good practice)
        # For Sinkhorn in log-space: u <- u - lse(u, axis=1)
        
        # Triton's tl.max/sum reducers work on specific axis
        # current w is [4, 4]
        
        # Row LSE: [4, 1]
        max_row = tl.max(w, axis=1)
        sum_exp_row = tl.sum(tl.exp(w - max_row[:, None]), axis=1)
        lse_row = max_row + tl.log(sum_exp_row)
        
        # Broadcast subtraction: w [4, 4] - lse_row [4, 1]
        w = w - lse_row[:, None]
        
        # --- Col Norm ---
        # Col LSE: [1, 4]
        max_col = tl.max(w, axis=0)
        sum_exp_col = tl.sum(tl.exp(w - max_col[None, :]), axis=0)
        lse_col = max_col + tl.log(sum_exp_col)
        
        # Broadcast subtraction: w [4, 4] - lse_col [1, 4]
        w = w - lse_col[None, :]

    # Final Exp and Store
    w_out = tl.reshape(tl.exp(w), (16,))
    tl.store(out_ptr + w_offsets, w_out)


@triton.jit
def fused_mhc_forward_kernel(
    X_ptr,          # [B, S, N, D]
    P_ptr,          # [N, N] (Pre-normalized)
    Y_ptr,          # [B, S, N, D]
    stride_xb, stride_xs, stride_xn, stride_xd,
    stride_yb, stride_ys, stride_yn, stride_yd,
    B, S, N, D,
    BLOCK_SIZE_D: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    
    # Load P scalars manually
    # Row 0
    p00 = tl.load(P_ptr + 0); p01 = tl.load(P_ptr + 1); p02 = tl.load(P_ptr + 2); p03 = tl.load(P_ptr + 3)
    # Row 1
    p10 = tl.load(P_ptr + 4); p11 = tl.load(P_ptr + 5); p12 = tl.load(P_ptr + 6); p13 = tl.load(P_ptr + 7)
    # Row 2
    p20 = tl.load(P_ptr + 8); p21 = tl.load(P_ptr + 9); p22 = tl.load(P_ptr + 10); p23 = tl.load(P_ptr + 11)
    # Row 3
    p30 = tl.load(P_ptr + 12); p31 = tl.load(P_ptr + 13); p32 = tl.load(P_ptr + 14); p33 = tl.load(P_ptr + 15)

    # Pointers
    x_base = X_ptr + pid_b * stride_xb + pid_s * stride_xs
    y_base = Y_ptr + pid_b * stride_yb + pid_s * stride_ys
    
    d_offsets = tl.arange(0, BLOCK_SIZE_D)
    
    for d_start in range(0, D, BLOCK_SIZE_D):
        offs = d_start + d_offsets
        mask = offs < D
        
        # Load X columns for all 4 branches
        x0 = tl.load(x_base + 0 * stride_xn + offs * stride_xd, mask=mask, other=0.0)
        x1 = tl.load(x_base + 1 * stride_xn + offs * stride_xd, mask=mask, other=0.0)
        x2 = tl.load(x_base + 2 * stride_xn + offs * stride_xd, mask=mask, other=0.0)
        x3 = tl.load(x_base + 3 * stride_xn + offs * stride_xd, mask=mask, other=0.0)
        
        # Compute Y = P @ X
        
        # Y0 = P00*X0 + P01*X1 + ...
        y0 = p00*x0 + p01*x1 + p02*x2 + p03*x3
        
        # Y1
        y1 = p10*x0 + p11*x1 + p12*x2 + p13*x3
        
        # Y2
        y2 = p20*x0 + p21*x1 + p22*x2 + p23*x3
        
        # Y3
        y3 = p30*x0 + p31*x1 + p32*x2 + p33*x3
        
        # Store
        tl.store(y_base + 0 * stride_yn + offs * stride_yd, y0, mask=mask)
        tl.store(y_base + 1 * stride_yn + offs * stride_yd, y1, mask=mask)
        tl.store(y_base + 2 * stride_yn + offs * stride_yd, y2, mask=mask)
        tl.store(y_base + 3 * stride_yn + offs * stride_yd, y3, mask=mask)


# -----------------------------------------------------------------------------
# AUTOGRAD FUNCTION
# -----------------------------------------------------------------------------

class FusedMHCFunction(Function):
    @staticmethod
    def forward(ctx, x, w, n_iters=5):
        """
        x: [B, S, N, D]
        w: [N, N] (raw logits)
        """
        B, S, N, D = x.shape
        # Assertion added to prevent silent failures if config.mhc_branches != 4
        assert N == 4, f"MHC Triton kernel currently hardcoded for N=4, got N={N}"
        
        # Sinkhorn
        P = torch.empty_like(w)
        # Using sinkhorn_kernel (renamed from sinkhorn_knopp_kernel)
        sinkhorn_kernel[(1,)](w, P, N=4, n_iters=n_iters)
        
        # Mixing
        y = torch.empty_like(x)
        grid = (B, S)
        # Ensure BLOCK_SIZE_D is power of 2 and consistent
        BLOCK_SIZE_D = 128
        
        fused_mhc_forward_kernel[grid](
            x, P, y,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
            B, S, N, D,
            BLOCK_SIZE_D=BLOCK_SIZE_D
        )
        
        ctx.save_for_backward(x, P, w)
        ctx.n_iters = n_iters
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, P, w = ctx.saved_tensors
        # grad_output: [B, S, N, D]
        
        # 1. Gradient w.r.t X = P.T @ dL/dY
        grad_x = torch.matmul(P.t(), grad_output)
        
        # 2. Gradient w.r.t P = dL/dY @ X.T -> Sum over B, S
        grad_P = torch.matmul(grad_output, x.transpose(-1, -2)) # [B, S, N, N]
        grad_P = grad_P.sum(dim=(0, 1)) # [N, N]
        
        # 3. Gradient w.r.t W (Backprop through Sinkhorn)
        # Re-compute Sinkhorn in Torch to define gradient path
        with torch.enable_grad():
            w_temp = w.detach().clone().requires_grad_(True)
            w_curr = w_temp
            for _ in range(ctx.n_iters):
                # Row
                w_curr = w_curr - torch.logsumexp(w_curr, dim=1, keepdim=True)
                # Col
                w_curr = w_curr - torch.logsumexp(w_curr, dim=0, keepdim=True)
            
            P_recomputed = torch.exp(w_curr)
            P_recomputed.backward(grad_P)
            grad_w = w_temp.grad
            
        return grad_x, grad_w, None


def mhc_forward(x, w):
    return FusedMHCFunction.apply(x, w)
