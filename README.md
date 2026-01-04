# Hyper-Mnemosyne

**Hyper-Mnemosyne** is a high-performance 3B parameter Language Model architecture designed for consumer hardware (specifically NVIDIA RTX 3090). It synthesizes:
- **Manifold-Constrained Hyper-Connections (mHC)** for deep signal stability.
- **Mamba-2** (State Space Models) for efficient linear-time context processing.
- **Titans Neural Memory** for test-time infinite context learning.
- **Muon Optimizer** for memory-efficient 2D parameter training.

## Installation

```bash
pip install -r requirements.txt
```

*Note: Ensure you have the NVIDIA CUDA Toolkit installed for Triton kernel compilation.*

## Usage

### Training
To train the model (assuming you have prepared data):

```bash
python3 -m training.train --data_dir ./data --batch_size 4
```

### Configuration
Adjust model hyperparameters in `config.py`.

## Project Structure
- `model/`: Core architecture (Backbone, mHC, Titans, Triton Kernels).
- `training/`: Optimization (Muon) and Training loops.
- `tests/`: Unit tests.
