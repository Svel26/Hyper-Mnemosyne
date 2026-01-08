import torch
import pytest
from model.titans import TitansMemoryLayer, MemoryMLP

class MockConfig:
    d_model = 128
    memory_width = 256
    
def test_titans_memory_layer_structure():
    config = MockConfig()
    memory = TitansMemoryLayer(config)
    
    assert isinstance(memory.memory_mlp, MemoryMLP)
    
def test_titans_forward_pass():
    config = MockConfig()
    memory = TitansMemoryLayer(config)
    
    B, S, D = 2, 10, 128
    x = torch.randn(B, S, D)
    
    mem_out, surprise_loss, state = memory(x)
    
    # Check shapes
    assert mem_out.shape == (B, S, D), f"Expected shape {(B, S, D)}, got {mem_out.shape}"
    assert surprise_loss.ndim == 0, "Surprise loss should be scalar mean"
    assert state.shape == (B, S, D), "State should match input size for this simplified EMA"
    
    # Check value consistency (simple pass)
    # MLP is w2(act(w1(x)))
    # If initialized, output shouldn't be zero/nan
    assert not torch.isnan(mem_out).any()
    assert not torch.isnan(surprise_loss).any()
    assert not torch.isnan(state).any()

def test_titans_time_mixing():
    """
    Verify that the memory state actually mixes time.
    If we input [1, 0, 0, 0], the state at t=3 should be non-zero (decayed 1).
    In the broken version, it was 0.
    """
    config = MockConfig()
    config.d_model = 1 # Override for simple scalar test
    config.memory_width = 4
    memory = TitansMemoryLayer(config)
    memory.decay = 0.5 # Simple decay
    
    # Input: [1, 0, 0, 0]
    B, S, D = 1, 4, 1
    x = torch.zeros(B, S, D)
    x[0, 0, 0] = 1.0
    
    _, _, state = memory(x) # Training mode
    
    # Expected State Evolution:
    # t=0: d*0 + (1-d)*1 = 0.5
    # t=1: d*0.5 + (1-d)*0 = 0.25
    # t=2: d*0.25 + (1-d)*0 = 0.125
    # t=3: d*0.125 + (1-d)*0 = 0.0625
    
    final_val = state[0, 3, 0].item()
    print(f"Final State Value: {final_val}")
    
    assert final_val > 0.0, "State should decay over time, not disappear!"
    assert abs(final_val - 0.0625) < 1e-4, f"Expected 0.0625, got {final_val}"

def test_titans_gradients():
    config = MockConfig()
    memory = TitansMemoryLayer(config)
    
    x = torch.randn(2, 5, 128, requires_grad=True)
    mem_out, loss, _ = memory(x)
    
    loss.backward()
    
    # Check that gradients flowed to weights
    assert memory.memory_mlp.w1.weight.grad is not None
    assert memory.memory_mlp.w2.weight.grad is not None
    assert x.grad is not None
