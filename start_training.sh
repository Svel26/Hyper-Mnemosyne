#!/bin/bash
# Hyper-Mnemosyne Training Launcher
set -e

# 1. Activate Environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# 2. Check Data (Scaled Up)
if [ ! -d "data" ] || [ -z "$(ls -A data)" ]; then
    echo "Generating Large Scale FineWeb-Edu Data (1M samples)..."
    # 1M sample-10BT subset. ~10GB parquet on disk.
    python3 scripts/prepare_fineweb.py --num_samples 1000000 --output_dir data/ --seq_len 4096
fi

# 3. Backbone Training (Stage 1) - FULL SCALE
echo "Starting Stage 1: Backbone Training (50,000 Steps)..."
# Batch 1 * Acc 4 = Batch 4. 50k steps.
python3 -m training.train \
    --data_dir data/fineweb_prod \
    --batch_size 1 \
    --grad_accum_steps 4 \
    --max_steps 50000 \
    --compile \
    --training_stage backbone

echo "Stage 1 Complete. Model saved to model_final.pt"

# 4. Memory Training (Stage 2) - FULL SCALE
echo "Starting Stage 2: Titans Memory Training (5,000 Steps)..."
# 5k steps for memory to settle.
python3 -m training.train \
    --data_dir data/fineweb_prod \
    --batch_size 1 \
    --grad_accum_steps 4 \
    --max_steps 5000 \
    --pretrained_path model_final.pt \
    --compile \
    --training_stage memory

echo "Full Training Complete! Final model ready."
