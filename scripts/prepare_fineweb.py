import os
import argparse
import random
from datasets import load_dataset
from transformers import GPT2TokenizerFast
import multiprocessing

def jepa_masking_fn(batch):
    """
    Batched processing function for dataset.map
    """
    # Tokenizer is not pickleable directly in map if passed as partial often, 
    # but here we initialize it inside or use global if fork.
    # Better: initialize inside to be safe with 'spawn' context.
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    # Silence warnings about >1024 length (we handle 4k+ manually)
    tokenizer.model_max_length = 100000
    
    seq_len = 4096 # Hardcoded for this map function or passed via fn_kwargs
    outputs = {"context_ids": [], "target_ids": []}
    
    for text in batch['text']:
        if len(text) < 500: continue
        
        tokens = tokenizer.encode(text)
        
        # Sliding window with stride? Or just non-overlapping chunks?
        stride = seq_len
        for i in range(0, len(tokens) - seq_len + 1, stride):
            clean_seq = tokens[i : i + seq_len]
            if len(clean_seq) < seq_len: continue
            
            # Apply Masking
            noisy_seq = list(clean_seq)
            
            # 1. Span Deletion (10% chance)
            if random.random() < 0.1:
                span_len = random.randint(10, seq_len // 4)
                start = random.randint(0, seq_len - span_len)
                del noisy_seq[start:start+span_len]
                
            # 2. Token Masking (15% prob)
            # Vectorized-like loop
            for j in range(len(noisy_seq)):
                if random.random() < 0.15:
                    r = random.random()
                    if r < 0.8:
                        noisy_seq[j] = random.randint(0, len(tokenizer)-1)
                    elif r < 0.9:
                        noisy_seq[j] = tokenizer.eos_token_id
                        
            # Pad back to seq_len
            if len(noisy_seq) < seq_len:
                noisy_seq += [tokenizer.eos_token_id] * (seq_len - len(noisy_seq))
            else:
                noisy_seq = noisy_seq[:seq_len]
                
            outputs["context_ids"].append(noisy_seq)
            outputs["target_ids"].append(clean_seq)
            
    return outputs

def prepare_fineweb(output_dir, num_samples=100_000, seq_len=4096, num_proc=4):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Loading FineWeb-Edu (sample-10BT) with {num_proc} workers...")
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")
    
    # SHARDING STRATEGY
    # 10BT tokens ~ 2.5M sequences of 4k.
    # To improve UX, we split the dataset into N shards and process/save them sequentially.
    # This ensures files appear in `data/` incrementally.
    
    num_shards = 50 # 50 files
    print(f"Dataset Loaded. Splitting into {num_shards} shards for incremental processing...")
    
    for shard_idx in range(num_shards):
        print(f"--- Processing Shard {shard_idx+1}/{num_shards} ---")
        shard = dataset.shard(num_shards=num_shards, index=shard_idx)
        
        # Map just this shard
        processed_shard = shard.map(
            jepa_masking_fn,
            batched=True,
            batch_size=args.batch_size,
            num_proc=num_proc, 
            remove_columns=dataset.column_names,
            desc=f"Shard {shard_idx}"
        )
        
        if len(processed_shard) > 0:
            output_file = os.path.join(output_dir, f"train_data_{shard_idx}.parquet")
            processed_shard.to_parquet(output_file)
            print(f"Saved {output_file} ({len(processed_shard)} sequences)")
        
        # Clear memory explicitly (though python usually handles it)
        del processed_shard
        del shard

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/")
    parser.add_argument("--num_samples", type=int, default=1000000) 
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=200) 
    args = parser.parse_args()
    
    cpu_count = os.cpu_count() or 4
    n_proc = 6 if cpu_count >= 6 else cpu_count
    
    prepare_fineweb(args.output_dir, num_samples=args.num_samples, seq_len=args.seq_len, num_proc=n_proc)
