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
    assert state.shape == (B, D), "Training mode should return final state [B, D]"
    
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
    
    # To verify internal state mixing, we check mem_out or the final returned state.
    # The Parallel Matrix implementation returns final state (h_T).
    # But h_T is result of processing x_{0..T}.
    # However, 'mem_out' is predictions based on h_{t-1}.
    
    # Let's check mem_out predictions.
    # t=0: READ h_{-1}=0. mem_out[0] predicts from 0.
    # t=1: READ h_0 = d*0 + (1-d)*1 = 0.5. mem_out[1] predicts from 0.5.
    # t=2: READ h_1 = d*0.5 + 0 = 0.25. mem_out[2] predicts from 0.25.
    # t=3: READ h_2 = 0.125. mem_out[3] predicts from 0.125.
    
    # The final returned state 'state' is h_3 = 0.0625.
    
    mem_out, _, final_state = memory(x)
    
    final_val = final_state[0, 0].item()
    print(f"Final State Value (h_3): {final_val}")
    
    assert final_val > 0.0, "State should decay over time, not disappear!"
    assert abs(final_val - 0.0625) < 1e-4, f"Expected 0.0625, got {final_val}"

def test_titans_gradients():
    config = MockConfig()
    memory = TitansMemoryLayer(config)
    
    x = torch.randn(2, 5, 128, requires_grad=True)
    mem_out, loss, _ = memory(x) # State is returned but ignored here
    
    loss.backward()
    
    # Check that gradients flowed to weights
    assert memory.memory_mlp.w1.weight.grad is not None
    assert memory.memory_mlp.w2.weight.grad is not None
    assert x.grad is not None
