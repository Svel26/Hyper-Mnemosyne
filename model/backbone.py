import torch
import torch.nn as nn
try:
    from mamba_ssm.modules.mamba2 import Mamba2
except ImportError:
    try:
        from mamba_ssm import Mamba2
    except ImportError:
         print("Warning: Mamba2 not found, using placeholder or crashing.")
         raise

from transformers.models.llama.modeling_llama import LlamaAttention, LlamaConfig, LlamaRotaryEmbedding


from .mhc import MHC
from .titans import TitansMemoryLayer

class HybridBlock(nn.Module):
    def __init__(self, config, layer_idx, use_attention=False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.use_attention = use_attention
        
        self.norm = nn.LayerNorm(config.d_model)
        
        # mHC Mixer for the residual stream
        self.mhc = MHC(d_model=config.d_model, n_branches=config.mhc_branches)
        
        self.mixer_needs_pos_emb = False
        
        if use_attention:
            # Using Llama Attention as a robust baseline
            llama_config = LlamaConfig(
                hidden_size=config.d_model,
                num_attention_heads=config.mhc_branches * 4, # Just a heuristic
                num_key_value_heads=config.mhc_branches * 4,
                max_position_embeddings=config.max_seq_len,
                _attn_implementation="eager",
            )
            self.mixer = LlamaAttention(config=llama_config, layer_idx=layer_idx)
            
            # Robustness: Check signature to support different Transformers versions
            import inspect
            sig = inspect.signature(self.mixer.forward)
            if "position_embeddings" in sig.parameters:
                self.mixer_needs_pos_emb = True
                
        else:
            # Mamba-2
            self.mixer = Mamba2(
                d_model=config.d_model,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand
            )
            
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Linear(config.d_model * 4, config.d_model)
        )

    def forward(self, x, residual_state=None, position_ids=None, attention_mask=None, past_key_values=None, position_embeddings=None):
        """
        x: [B, S, D]
        residual_state: [B, S, N, D]
        """
        # 1. Initialize mHC state if None (Entry to the Multi-Lane Highway)
        if residual_state is None:
            B, S, _ = x.shape
            # Fix Symmetry Collapse: Initialize with random noise or projection
            noise = torch.randn(B, S, self.config.mhc_branches, self.config.d_model, device=x.device, dtype=x.dtype) * 0.02
            residual_state = x.unsqueeze(2) + noise
        
        # 2. mHC Mixing
        # mixed_state: [B, S, N, D]
        mixed_state = self.mhc(residual_state)
        
        # 3. Aggregation
        layer_input = mixed_state.mean(dim=2) # [B, S, D]
        normalized_input = self.norm(layer_input)
        
        # 4. Core Processing
        if self.use_attention:
            # LlamaAttention Compatibility Layer
            # Dynamically pass position_embeddings only if signature requires it
            kwargs = {
                "hidden_states": normalized_input,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values
            }
            if self.mixer_needs_pos_emb:
                 kwargs["position_embeddings"] = position_embeddings
            
            mixer_out = self.mixer(**kwargs)[0]
        else:
            mixer_out = self.mixer(normalized_input)
            
        # 5. FFN
        ffn_out = self.ffn(self.ffn_norm(mixer_out + layer_input))
        
        # 6. Distribute back to branches
        delta = ffn_out.unsqueeze(2) # [B, S, 1, D]
        new_residual_state = mixed_state + delta
        
        return new_residual_state


class HyperMnemosyne(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        
        # Titans Memory (Simplified)
        self.memory = TitansMemoryLayer(config)
        
        self.layers = nn.ModuleList()
        # Ensure gradients flow for checkpointing check
        self.dummy_param = nn.Parameter(torch.empty(0))

        for i in range(config.n_layers):
            is_attn = (i % 6 == 0) and (i > 0)
            self.layers.append(HybridBlock(config, i, use_attention=is_attn))
            
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # RoPE Setup
        rope_config = LlamaConfig(
            hidden_size=config.d_model,
            num_attention_heads=config.mhc_branches * 4,
            max_position_embeddings=config.max_seq_len,
            _attn_implementation="eager",
        )
        self.rotary_emb = LlamaRotaryEmbedding(rope_config)

        # JEPA Latent Predictor
        self.jepa_predictor = nn.Sequential(
            nn.Linear(config.d_model, config.predictor_dim),
            nn.GELU(),
            nn.Linear(config.predictor_dim, config.d_model)
        )

    def forward(self, input_ids, **kwargs):
        x = self.embeddings(input_ids)
        B, S, D = x.shape
        
        # Create Position IDs for RoPE
        position_ids = torch.arange(0, S, dtype=torch.long, device=x.device).unsqueeze(0)
        
        # RoPE Embedding
        # Required for this version of LlamaAttention
        position_embeddings = self.rotary_emb(x, position_ids)
        
        # Causal Mask
        attention_mask = torch.triu(torch.ones(S, S, device=x.device) * float('-inf'), diagonal=1)
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0) # [1, 1, S, S]
        
        # Memory Interaction (Simplified Titans)
        mem_out, memory_loss = self.memory(x)
        x = x + mem_out # Simple residual
        
        # Initialize mHC state
        residual_state = None 
        
        # Gradient Checkpointing Logic
        use_checkpointing = self.config.gradient_checkpointing and self.training
        
        if use_checkpointing:
             # Manual init to ensure it's a tensor with grad
             noise = torch.randn(B, S, self.config.mhc_branches, self.config.d_model, device=x.device, dtype=x.dtype) * 0.02
             residual_state = x.unsqueeze(2) + noise
             
        for i, layer in enumerate(self.layers):
            if use_checkpointing:
                residual_state = torch.utils.checkpoint.checkpoint(
                    layer,
                    x,
                    residual_state,
                    position_ids,
                    attention_mask,
                    None, # past_key_values
                    position_embeddings,
                    use_reentrant=False
                )
            else:
                residual_state = layer(
                    x, 
                    residual_state, 
                    position_ids, 
                    attention_mask,
                    None, # past_key_values
                    position_embeddings
                )
            
        # Final Aggregation
        final_out = residual_state.mean(dim=2)
        final_out = self.final_norm(final_out)
        logits = self.lm_head(final_out)
        
        return logits, final_out, memory_loss
