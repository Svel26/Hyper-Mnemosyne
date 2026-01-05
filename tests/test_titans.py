import torch
import torch.nn as nn
import pytest
from model.titans import TitansMemoryLayer
from config import HyperMnemosyneConfig

def test_titans_meta_learning_gradients():
    """
    Verify that the meta-parameters (step_size, decay) receive gradients
    after the inner-loop/outer-loop process.
    """
    config = HyperMnemosyneConfig()
    titans = TitansMemoryLayer(config)
    
    # Enable grad for meta-params
    assert titans.step_size.requires_grad
    assert titans.decay.requires_grad
    
    # Data
    B, S, D = 2, 10, config.d_model
    x_support = torch.randn(B, S, D)
    x_query = torch.randn(B, S, D)
    
    # 1. Inner Loop (Support Set)
    _, loss_support = titans(x_support)
    
    # 2. Get Updated Weights (Functional)
    # This involves taking gradients of loss_support w.r.t memory_mlp weights
    # And computing new weights using step_size/decay
    updated_weights = titans.get_updated_weights(loss_support)
    
    # Check if updated weights are part of the graph connected to step_size
    # (By checking if we can backprop from them)
    # Pick one weight to test
    w1_new = updated_weights['w1']
    # w1_new = w1_old - step_size * grad
    # So w1_new should have grad_fn connecting to step_size
    
    # 3. Outer Loop (Query Set)
    # Forward pass using updated weights
    _, loss_query = titans(x_query, memory_params=updated_weights)
    
    # 4. Backward
    loss_query.backward()
    
    # 5. Verify Gradients on Meta-Params
    assert titans.step_size.grad is not None, "step_size did not receive gradient!"
    assert titans.decay.grad is not None, "decay did not receive gradient!"
    
    print(f"Step Size Grad: {titans.step_size.grad}")
    
    # Ensure MemoryMLP weights also have grads (from the standard path + optimization path?)
    # In MAML (Model-Agnostic Meta-Learning), we optimize the initial weights too.
    # Here, we do want to optimize the base memory weights too? 
    # Yes, usually.
    assert titans.memory_mlp.w1.weight.grad is not None

if __name__ == "__main__":
    test_titans_meta_learning_gradients()
