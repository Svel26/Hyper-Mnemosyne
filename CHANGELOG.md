# Hyper-Mnemosyne Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-01-31

### Added

- Initial release of Hyper-Mnemosyne architecture
- Mamba-2 backbone with efficient sequence modeling
- Manifold-Constrained Hyper-Connections (mHC) for multi-branch processing
- Titans Neural Memory for lightweight context tracking
- JEPA (Joint-Embedding Predictive Architecture) training
- Two-stage training protocol (backbone + memory)
- Optimized data preparation pipeline (100x speedup)
- Multi-worker DataLoader with prefetching
- Gradient clipping and LR scheduling
- Comprehensive logging and monitoring
- Package installation support via setup.py
- Automated training script (start_training.sh)

### Fixed

- Data preparation tokenization (now supports 4096 sequence length)
- JEPA masking strategy (single-pass with stop-gradient)
- Titans Memory initialization (small random init instead of zero-init)
- Gradient checkpointing determinism
- Stage 2 training data flow (uses clean inputs)
- Loss scaling for gradient accumulation
- Import path issues (automatic PYTHONPATH setup)

### Performance

- Data preparation: ~30 minutes for 100k samples (was days)
- Vectorized data loading: 10x faster tensor creation
- GPU utilization: 80-95% (was <50%)
- Training stability: consistent convergence with gradient clipping

## [Unreleased]

### Planned

- Distributed training support (DDP)
- Validation metrics and evaluation
- Inference server implementation
- Quantization support (int8/int4)
- LoRA fine-tuning capability
