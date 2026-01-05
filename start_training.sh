#!/bin/bash
# Hyper-Mnemosyne Training Launcher

# 1. Activate Environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# 2. Check Data
if [ ! -d "data" ] || [ -z "$(ls -A data)" ]; then
    echo "Generating FineWeb-Edu Data (10k samples, Sequence Length 4096)..."
    # Adjust --seq_len if you have less than 24GB VRAM (e.g. 1024 or 2048)
    python3 scripts/prepare_fineweb.py --num_samples 10000 --output_dir data/ --seq_len 4096
fi

# 3. Train
echo "Starting Stage 1 Training (Backbone)..."
# Adjust batch_size or set gradient_checkpointing in config.py if OOM
python3 -m training.train --batch_size 1 --max_steps 5000 --training_stage backbone
