import pandas as pd
import numpy as np
import os
import argparse

def create_dummy_data(output_dir, num_samples=1000, seq_len=128, vocab_size=50257):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Generating {num_samples} dummy samples...")
    
    data = []
    for _ in range(num_samples):
        # Generate random target_ids
        target_ids = np.random.randint(0, vocab_size, size=(seq_len,)).tolist()
        
        # Context is slightly noisy version
        context_ids = list(target_ids)
        # simplistic noise
        if len(context_ids) > 10:
             context_ids[5:10] = [vocab_size-1]*5 # mask
             
        data.append({
            "context_ids": context_ids,
            "target_ids": target_ids
        })
        
    df = pd.DataFrame(data)
    output_file = os.path.join(output_dir, "train_data_dummy.parquet")
    df.to_parquet(output_file)
    print(f"Saved {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/")
    args = parser.parse_args()
    create_dummy_data(args.output_dir)
