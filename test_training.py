#!/usr/bin/env python3
"""
Quick Test Script for Hyper-Mnemosyne Training
Creates minimal test dataset and runs 100 steps to verify fixes
"""

import os
import sys
import subprocess

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    print(f"$ {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n❌ Failed: {description}")
        return False
    print(f"\n✅ Success: {description}")
    return True

def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║      Hyper-Mnemosyne Training Test Suite                ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Create test dataset
    if not run_command(
        "python scripts/prepare_data.py --num_samples 1000 --seq_len 512 --output_dir data/test/",
        "Creating test dataset (1000 samples, 512 seq len)"
    ):
        return 1
    
    # Step 2: Run training for 100 steps
    if not run_command(
        "python training/train.py --data_dir data/test/ --max_steps 100 --batch_size 2 --training_stage backbone --save_every 50",
        "Running 100-step training test"
    ):
        return 1
    
    print("""
    
╔══════════════════════════════════════════════════════════╗
║                   🎉 ALL TESTS PASSED! 🎉                ║
╚══════════════════════════════════════════════════════════╝

Next steps:
1. Check training.log for decreasing loss
2. If loss decreases, prepare full dataset:
   python scripts/prepare_data.py --num_samples 100000

3. Start full training:
   ./start_training.sh

4. Monitor GPU utilization (should be >80%):
   nvidia-smi -l 1
    """)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
