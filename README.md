# Hyper-Mnemosyne

Advanced neural architecture combining Mamba-2, Manifold-Constrained Hyper-Connections, and Titans Neural Memory with JEPA training.

## Quick Start

### 1. Installation

Clone and install:

```bash
git clone https://github.com/yourusername/Hyper-Mnemosyne.git
cd Hyper-Mnemosyne

# Option A: Install as editable package (recommended)
pip install -e .

# Option B: Use virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Training

The easiest way:

```bash
./start_training.sh
```

Or manually:

```bash
# Prepare data (100k samples, ~30 min)
python scripts/prepare_data.py --num_samples 100000 --seq_len 4096

# Stage 1: Train backbone
python training/train.py \
  --data_dir data/ \
  --training_stage backbone \
  --batch_size 4 \
  --max_steps 50000

# Stage 2: Train memory
python training/train.py \
  --data_dir data/ \
  --training_stage memory \
  --pretrained_path model_final.pt \
  --batch_size 4 \
  --max_steps 70000
```

## Architecture

- **Backbone:** Mamba-2 (efficient sequence modeling)
- **Hyper-Connections:** Manifold-constrained multi-branch paths  
- **Memory:** Titans Neural Memory (lightweight context tracking)
- **Training:** JEPA (Joint-Embedding Predictive Architecture)

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU)
- 24GB+ VRAM (RTX 3090 or better)

See `requirements.txt` for full dependencies.

## Project Structure

```
Hyper-Mnemosyne/
├── config.py              # Model configuration
├── model/                 # Model architecture
│   ├── backbone.py        # Main model
│   ├── mhc.py            # Hyper-connections
│   └── titans.py         # Memory layer
├── training/             # Training code
│   ├── train.py          # Main training loop
│   ├── data_utils.py     # Data loading
│   └── muon.py           # Muon optimizer
├── scripts/              # Utilities
│   └── prepare_data.py   # Data preparation
└── start_training.sh     # Automated training
```

## Citation

If you use this code, please cite:

```bibtex
@misc{hypermnemosyne2026,
  title={Hyper-Mnemosyne: Efficient Long-Context Learning with Titans Memory},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/Hyper-Mnemosyne}
}
```

## License

Apache 2.0 - See LICENSE file
