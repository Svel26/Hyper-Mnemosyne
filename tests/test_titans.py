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
    assert state is None, "Training mode should not return state sequence (optimization)"
    
    # Check value consistency (simple pass)
    # MLP is w2(act(w1(x)))
    # If initialized, output shouldn't be zero/nan
    assert not torch.isnan(mem_out).any()
    assert not torch.isnan(surprise_loss).any()
    # State is None in training, so strict check above (assert state is None) is sufficient.

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
    
    # To verify internal state mixing, we can't look at returned state (it's None).
    # We must trust the _scan method or check mem_out.
    # OR, we can access the protected method for this test.
    
    states = memory._scan(x, memory.decay)
    
    # Expected State Evolution (Logic Updated for Read-Before-Write):
    # t=0: READ h_{-1}=0. PREDICT based on 0. WRITE h_0 = d*0 + (1-d)*1 = 0.5.
    # t=1: READ h_0=0.5. PREDICT based on 0.5. WRITE h_1 = d*0.5 + 0 = 0.25.
    # t=2: READ h_1=0.25. PREDICT based on 0.25. WRITE h_2 = 0.125.
    # t=3: READ h_2=0.125. PREDICT based on 0.125. WRITE h_3 = 0.0625.
    
    # The 'states' tensor returned by _scan contains [h_{-1}, h_0, h_1, h_2]
    # states[0] = 0
    # states[1] = 0.5
    # states[2] = 0.25
    # states[3] = 0.125
    
    val_t3 = states[0, 3, 0].item() # Should be 0.125
    print(f"State used for prediction at t=3 (h_2): {val_t3}")
    
    assert val_t3 > 0.0, "State should decay over time, not disappear!"
    assert abs(val_t3 - 0.125) < 1e-4, f"Expected 0.125, got {val_t3}"

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
