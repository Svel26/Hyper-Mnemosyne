![Hyper-Mnemosyne Header](Header_IMG.jpeg)

# Hyper-Mnemosyne 🧠 (Research Prototype)

> [!WARNING]
> **Status: Experimental / Work in Progress**
> This is a small-scale research prototype (~150M parameters) designed for architectural experimentation on consumer hardware. It is **not** a production-ready LLM or a "3B parameter SOTA killer".

**Hyper-Mnemosyne** is an experimental testbed for exploring hybrid state-space and memory-augmented architectures on a single NVIDIA RTX 3090/4090.

It integrates simplified adaptations of three research concepts:

1. **Mamba-2 Backbone**: Leveraging SSD (Structured State Space Duality) for linear-time context processing.
2. **Titans-Inspired Memory**: An experimental "Gated Residual Memory" module inserted into the residual stream (simplified from the original Titans proposal).
3. **JEPA-Inspired Auxiliary Loss**: A latent consistency objective inspired by Joint-Embedding Predictive Architectures to improve semantic density.

## 🧪 Architecture & Goals

This project aims to verify whether complex architectural components like mHC (Manifold-Constrained Hyper-Connections) and Neural Memory can be stabilized and trained at a small scale.

* **Scale**: ~150M Parameters (d_model=768, layers=12)
* **Design Philosophy**: Convergence of diverse architectural ideas (SSM + Memory + Latent Prediction) into a single differentiable stack.
* **Hardware Target**: Single Consumer GPU (24GB VRAM).

## 🚀 Quick Start

### Prerequisites

* Linux (Ubuntu 20.04/22.04 recommended)
* Python 3.10+
* NVIDIA GPU with 8GB+ VRAM (tested on RTX 3090)
* CUDA Toolkit 11.8+

### Installation

```bash
# 1. Clone & Setup Environment
git clone https://github.com/svel26/Hyper-Mnemosyne
cd Hyper-Mnemosyne
python3 -m venv venv
source venv/bin/activate

# 2. Install Dependencies
pip install -r requirements.txt
```

## 🛠️ Usage

### Training

The training protocol allows for experimentation with the backbone and memory modules separately.

#### Stage 1: Backbone Training

Trains the Mamba-2 core with the mHC-inspired mixing and JEPA auxiliary loss.

```bash
# Small-scale debug run
python3 -m training.train --batch_size 4 --max_steps 1000 --training_stage backbone --compile
```

#### Stage 2: Memory Training (Experimental)

Freezes the backbone and trains only the memory gating/residual connection.

```bash
python3 -m training.train --batch_size 4 --max_steps 500 --training_stage memory --pretrained_path model_final.pt
```

### Inference

```bash
python3 inference.py --prompt "The future of AI is"
```

## 📂 Project Structure

```text
├── config.py               # Hyperparameters (Scaled down for research)
├── model/
│   ├── backbone.py         # Main Model Class
│   ├── mhc.py              # Manifold-Constrained Hyper-Connections (Triton)
│   ├── titans.py           # Simplified Gated Memory Module
│   └── triton_kernels.py   # Custom CUDA/Triton Kernels
├── training/
│   ├── train.py            # Training Loop
│   ├── muon.py             # Muon Optimizer
│   └── data_utils.py       # Data Loading
└── requirements.txt
```

## License

Apache License. See [LICENSE](LICENSE) for details.
