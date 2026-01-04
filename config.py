from dataclasses import dataclass

@dataclass
class HyperMnemosyneConfig:
    # Model Dimensions
    d_model: int = 2048
    n_layers: int = 32
    vocab_size: int = 50257  # Standard GPT-2/Llama tokenizer size approximation, adjust as needed
    
    # Manifold-Constrained Hyper-Connections
    mhc_branches: int = 4
    
    # Mamba-2 Settings
    d_state: int = 128
    d_conv: int = 4
    expand: int = 2
    
    # Titans Memory
    memory_depth: int = 2
    memory_width: int = 2048
    
    # Training / Optimization
    max_seq_len: int = 4096  # Fit within 24GB with Mamba-2
    vocab_size: int = 32000 # Example vocab size
