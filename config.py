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
    max_seq_len: int = 4096  # Blueprint requirement
    gradient_checkpointing: bool = True
    
    # JEPA
    jepa_weight: float = 0.5  # Weight of the latent prediction loss
    predictor_dim: int = 512 # Dimension of the latent predictor bottleneck
    
    # Training Stage
    # "backbone": Train mHC + Mamba-2 + JEPA (Titans frozen/disabled)
    # "backbone": Train mHC + Mamba-2 + JEPA (Titans frozen/disabled)
    # "memory": Freeze backbone, Train Titans Memory
    training_stage: str = "memory"
