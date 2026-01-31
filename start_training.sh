#!/bin/bash
# Hyper-Mnemosyne Training Launcher
set -e

# Set PYTHONPATH to project root (CRITICAL for imports to work)
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "🔧 PYTHONPATH set to: $PYTHONPATH"

# 1. Activate Environment (optional but recommended)
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# 2. Check Data
if [ ! -d "data" ] || [ -z "$(ls -A data 2>/dev/null)" ]; then
    echo "📥 Generating training data (100k samples, ~30 min)..."
    python3 scripts/prepare_data.py --output_dir data/ --num_samples 100000 --seq_len 4096
fi

# 3. Stage 1: Backbone Training
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║          Stage 1: Backbone + JEPA Training               ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

python3 training/train.py \
    --data_dir data/ \
    --batch_size 4 \
    --grad_accum_steps 1 \
    --max_steps 50000 \
    --warmup_steps 1000 \
    --save_every 1000 \
    --training_stage backbone

echo "✅ Stage 1 Complete!"

# 4. Stage 2: Memory Training
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║       Stage 2: Titans Memory Training                   ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

python3 training/train.py \
    --data_dir data/ \
    --batch_size 4 \
    --grad_accum_steps 1 \
    --max_steps 70000 \
    --warmup_steps 1000 \
    --save_every 1000 \
    --pretrained_path model_final.pt \
    --training_stage memory

echo ""
echo "🎉 Full Training Complete! Model saved to model_final.pt"
