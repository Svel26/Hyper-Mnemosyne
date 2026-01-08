from dataclasses import dataclass

@dataclass
class HyperMnemosyneConfig:
    # Model Dimensions (Scaled for RTX 3090 / 24GB VRAM)
    # Target: ~350M parameters
    d_model: int = 1024
    n_layers: int = 24
    vocab_size: int = 50257
    
    # Manifold-Constrained Hyper-Connections
    mhc_branches: int = 4
    
    # Mamba-2 Settings
    d_state: int = 128
    d_conv: int = 4
    expand: int = 2
    
    # Titans Memory
    memory_depth: int = 2
    memory_width: int = 4096 # Wide memory for production
    
    # Training / Optimization
    max_seq_len: int = 4096  # Blueprint requirement
    gradient_checkpointing: bool = True
    
    # JEPA
    jepa_weight: float = 0.5  # Weight of the latent prediction loss
    predictor_dim: int = 512 # Dimension of the latent predictor bottleneck
    
    # Training Stage
    # "backbone": Train mHC + Mamba-2 + JEPA (Titans frozen/disabled)
    # "memory": Freeze backbone, Train Titans Memory
    training_stage: str = "backbone"
