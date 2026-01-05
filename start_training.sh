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
    echo "Generatiing FineWeb-Edu Data (10k samples)..."
    python3 scripts/prepare_fineweb.py --num_samples 10000 --output_dir data/
fi

# 3. Train
echo "Starting Training (Press Ctrl+C to stop)..."
python3 -m training.train --batch_size 1 --max_steps 5000
