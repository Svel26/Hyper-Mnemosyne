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

def prepare_fineweb(output_dir, num_samples=100_000, seq_len=4096, num_proc=8):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Loading FineWeb-Edu (sample-10BT) with {num_proc} workers...")
    # Load FULL dataset (not streaming) to enable highly efficient map
    # It will download parquet files (~10GB)
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")
    
    # Select subset if needed (but 10BT is ~10M samples)
    # If user wants 1M samples, we can take first 1M *documents*?
    # No, num_samples usually refers to *sequences*.
    # Let's take a slice of documents that likely yields enough sequences.
    # 10BT tokens / 4096 ~ 2.5M sequences.
    # So we probably need ~40% of the dataset for 1M sequences.
    # Let's just process the whole "sample-10BT" and shuffle/select later or save all.
    # The user asked for "1M samples" (sequences).
    
    print("dataset loaded. applying map...")
    
    # Process
    processed = dataset.map(
        jepa_masking_fn,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Tokenizing & Masking"
    )
    
    print(f"Generated {len(processed)} sequences.")
    
    # Save to disk
    print("Saving to parquet chunks...")
    # Shard it
    # 1M rows per file roughly?
    shard_size = 100_000
    num_shards = (len(processed) // shard_size) + 1
    
    for i in range(num_shards):
        shard = processed.shard(num_shards=num_shards, index=i)
        output_file = os.path.join(output_dir, f"train_data_{i}.parquet")
        shard.to_parquet(output_file)
        print(f"Saved {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/")
    parser.add_argument("--num_samples", type=int, default=1000000) # Ignored, we consume full 10BT sample
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=1000) # For map
    args = parser.parse_args()
    
    # Get CPU count
    n_proc = os.cpu_count() or 4
    
    prepare_fineweb(args.output_dir, num_samples=args.num_samples, seq_len=args.seq_len, num_proc=n_proc)
