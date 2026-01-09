
import pandas as pd
import glob
import os
import numpy as np

def blend_data(data_dir="data"):
    print(f"Searching for parquet files in {data_dir}...")
    files = glob.glob(os.path.join(data_dir, "train_data_*.parquet"))
    
    if not files:
        print("No files found!")
        return

    print(f"Found {len(files)} shards. Loading into memory (The Blender)... 🌪️")
    
    # Load all
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not dfs:
        print("Failed to load any dataframes.")
        return

    full_df = pd.concat(dfs, ignore_index=True)
    total_samples = len(full_df)
    print(f"Total Samples Loaded: {total_samples}")
    
    print("Shuffling (Global Mix)... 🔀")
    # Global shuffle
    shuffled_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Verify Mix (Optional print)
    # print(shuffled_df.head())

    print(" rewriting shards...")
    # Clean old files first? Or overwrite? 
    # Safer to overwrite or delete old ones explicitly.
    # Let's delete old ones to ensure no dupes if we change shard count.
    for f in files:
        os.remove(f)
        
    # Chunk and Save
    samples_per_shard = 5000 
    num_shards = int(np.ceil(total_samples / samples_per_shard))
    
    for i in range(num_shards):
        start_idx = i * samples_per_shard
        end_idx = min((i + 1) * samples_per_shard, total_samples)
        
        chunk = shuffled_df.iloc[start_idx:end_idx]
        output_file = os.path.join(data_dir, f"blend_data_{i}.parquet") # New name scheme
        
        chunk.to_parquet(output_file)
        print(f"Saved {output_file} ({len(chunk)} samples)")
        
    print("Blender Complete. Data is now homogeneous meat-and-potatoes stew. 🍲")

if __name__ == "__main__":
    blend_data()
