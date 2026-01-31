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
    # Load W [N*N]
    w_offsets = tl.arange(0, N*N)
    w_flat = tl.load(w_ptr + w_offsets)
    
    # View as [N, N]
    w = tl.reshape(w_flat, (N, N))
    
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
    w_out = tl.reshape(tl.exp(w), (N*N,))
    tl.store(out_ptr + w_offsets, w_out)


@triton.jit
def fused_mhc_forward_kernel(
    X_ptr,          # [B, S, N, D]
    P_ptr,          # [N, N] (Pre-normalized)
    Y_ptr,          # [B, S, N, D]
    stride_xb, stride_xs, stride_xn, stride_xd,
    stride_yb, stride_ys, stride_yn, stride_yd,
    B, S, D,
    N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    
    # Load P scalars manually? No, load full P matrix N*N
    # P is small, we can load it into register
    # But for cleaner dynamic N, let's load it as a block?
    # P is [N, N]. N is typically 4, can be 8.
    
    # For fully dynamic N in triton loop, we might need loops.
    # But since N is small constexpr, we can unroll.
    
    # Load P: [N, N]
    p_offsets = tl.arange(0, N*N)
    P_flat = tl.load(P_ptr + p_offsets)
    P = tl.reshape(P_flat, (N, N))
    
    # Pointers
    x_base = X_ptr + pid_b * stride_xb + pid_s * stride_xs
    y_base = Y_ptr + pid_b * stride_yb + pid_s * stride_ys
    
    d_offsets = tl.arange(0, BLOCK_SIZE_D)
    
    for d_start in range(0, D, BLOCK_SIZE_D):
        offs = d_start + d_offsets
        mask = offs < D
        
        # Load X columns for all N branches
        # We need a way to load variable number of pointers?
        # Triton doesn't support list of tensors well.
        # We iterate over rows of P (output branches)
        
        # Pre-load X inputs
        # X: [N, BLOCK_SIZE_D]
        # We can construct a block pointer? 
        # Or simple load loop.
        
        # Let's use a temporary accumulator for result
        
        # Implementation constraint:
        # Triton dynamic indexing is tricky. 
        # But we can iterate over range(N) because N is constexpr.
        
        # 1. Load all input branches X_j
        # We can load X as [N, BLOCK_SIZE_D] using 2D load if strides allow? 
        # X is [B, S, N, D]. Stride_xn is the stride for N. Stride_xd is 1 usually.
        # We can load a block [N, BLOCK_SIZE_D]. 
        # Be careful with strides.
        
        # Advanced: Load [N, BLOCK_SIZE_D] block directly
        # x_ptr_base = x_base + offs * stride_xd
        # But stride_xn might be large? No, N is adjacent usually.
        # Actually X is [B, S, N, D].
        # So X[b,s] is [N, D].
        # We are at X[b,s].
        # It is contiguous in memory as N x D matrix (row major usually).
        # We want to load columns [N, BLOCK_SIZE_D].
        # We can use tl.load using 2D offsets.
        
        n_range = tl.arange(0, N)
        d_range = offs
        
        # Ptrs: [N, BLOCK]
        # X_ptrs = x_base + (n_range[:, None] * stride_xn) + (d_range[None, :] * stride_xd)
        X_vals = tl.load(x_base + (n_range[:, None] * stride_xn) + (d_range[None, :] * stride_xd), 
                         mask=mask[None, :], other=0.0)
                         
        # Computing Y = P @ X
        # P: [N, N]
        # X: [N, BLOCK]
        # Y: [N, BLOCK]
        # Y_ij = sum_k (P_ik * X_kj)
        
        # Expand P to [N, N, 1] (Broadcasting axis 2)
        # Expand X to [1, N, BLOCK] (Broadcasting axis 0)
        # Prod: [N, N, BLOCK]
        
        # Triton slicing for broadcast
        P_b = P[:, :, None]
        X_b = X_vals[None, :, :]
        
        Y_vals = tl.sum(P_b * X_b, axis=1)
        # But for small matrices, standard matrix mult:
        
        # If tl.dot doesn't work for non-block types efficiently, manual loop:
        # P [N, N], X [N, Block]
        # Y_i = sum_j (P_ij * X_j)
        
        # Actually P is small, X is small-ish (BLOCK=128).
        # tl.dot works on blocks.
        # We need P converted to proper float type.
        
        # Store Y
        tl.store(y_base + (n_range[:, None] * stride_yn) + (d_range[None, :] * stride_yd), 
                 Y_vals, mask=mask[None, :])


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
        # assert N == 4, f"MHC Triton kernel currently hardcoded for N=4, got N={N}"
        
        # Sinkhorn
        P = torch.empty_like(w)
        # Using sinkhorn_kernel (renamed from sinkhorn_knopp_kernel)
        sinkhorn_kernel[(1,)](w, P, N=N, n_iters=n_iters)
        
        # Mixing
        y = torch.empty_like(x)
        grid = (B, S)
        # Ensure BLOCK_SIZE_D is power of 2 and consistent
        BLOCK_SIZE_D = 128
        
        fused_mhc_forward_kernel[grid](
            x, P, y,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
            B, S, D,
            N=N,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
            num_warps=4 # Ensure sufficient warps for dot ops
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
        # Re-compute Sinkhorn in Torch to define gradient path.
        # Trade-off: Re-computation saves VRAM (don't store N intermediate matrices)
        # but costs compute. For small N=4, this is negligible and preferred.
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
