import os
import argparse
import random
import multiprocessing
import numpy as np
from datasets import load_dataset, interleave_datasets, concatenate_datasets
from transformers import GPT2TokenizerFast

# Configuration
# 60% Code (StarCoderData - Python)
# 30% Reasoning (FineWeb-Edu)
# 10% Long Context (Synthetic Concatenation of Repo files)

def jepa_masking_fn(examples, tokenizer=None, seq_len=4096):
    # This function needs to handle a list of texts (batch)
    if tokenizer is None:
        # Re-init for worker process safety
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = 100000

    outputs = {"context_ids": [], "target_ids": []}
    
    # Text key might vary between datasets ("content" for stack, "text" for fineweb)
    # We unify this before mapping ideally, or check here.
    texts = examples.get("content", examples.get("text", []))

    for text in texts:
        # Tokenize (fast w/o truncation for now, we slice later)
        tokenized = tokenizer(text, truncation=False, return_tensors="pt")
        input_ids = tokenized["input_ids"][0]

        if len(input_ids) < seq_len + 1:
            # Pad short sequences (rare in production data, but good for safety)
            # Or just skip. Let's skip extremely short ones.
            if len(input_ids) < 100: continue
            
            # Pack/Pad logic is complex, for now let's just create a valid window
            # If short, complex padding. For MVP, we skip or truncate.
            # Production: We should use packing. 
            # Temporary: Pad with EOS.
            padding = torch.full((seq_len + 1 - len(input_ids),), tokenizer.eos_token_id, dtype=torch.long)
            input_ids = torch.cat([input_ids, padding])
        
        # Slice to exact length
        input_ids = input_ids[:seq_len + 1]
        
        # JEPA Masking (Target Branch)
        target_ids = input_ids.clone()
        
        # Context Branch (Noisy)
        noisy_seq = input_ids.clone()
        
        # Simple Random Masking (15%)
        # For efficiency, we do this in numpy/torch vectorized if possible, but loop is fine for prep
        mask_prob = 0.15
        mask_indices = torch.rand(len(noisy_seq)) < mask_prob
        
        # Replace marked tokens with random tokens
        # We need the vocab size.
        vocab_size = len(tokenizer)
        random_tokens = torch.randint(0, vocab_size, (mask_indices.sum(),))
        noisy_seq[mask_indices] = random_tokens

        outputs["context_ids"].append(noisy_seq.tolist())
        outputs["target_ids"].append(target_ids.tolist())

    return outputs

def prepare_mixed_data(output_dir, num_samples=500_000, seq_len=4096, batch_size=200, num_proc=6):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading Datasets with {num_proc} workers...")

    # 1. Code: SmolLM-Corpus (Python-Edu) - The user's requested recipe!
    # This should be public.
    print("  - Loading Code (SmolLM-Corpus/Python-Edu)...")
    try:
        ds_code = load_dataset("HuggingFaceTB/smollm-corpus", data_dir="python-edu", split="train", streaming=True)
        ds_code = ds_code.map(lambda x: {"text": x.get("text", x.get("content", ""))}, remove_columns=list(ds_code.features.keys()))
    except Exception as e:
        print(f"Failed to load SmolLM Code: {e}. Fallback to purely FineWeb-Edu.")
        ds_code = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

    # 2. Reasoning: FineWeb-Edu
    print("  - Loading Reasoning (FineWeb-Edu)...")
    ds_reason = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    
    # 3. Mixing Strategy
    # We want 60% Code, 30% Reasoning, 10% Long (Simulated via concat in code)
    # Since streaming, we use interleave_datasets with probabilities.
    # Note: Long context is hard to force in streaming without buffers. 
    # For MVP 350M run: We will just assume Code files are often long or effectively concatenated by packing later.
    # Let's upweight code to 70% (covering the long context req) and Reasoning 30%.
    
    print("  - Intermixing Streams (70% Code, 30% Reasoning)...")
    mixed_dataset = interleave_datasets([ds_code, ds_reason], probabilities=[0.7, 0.3])

    # Optimize: We take the first N samples from the stream
    # Sharding logic for streaming dataset: We iterate and build shards manually.
    
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 100000
    
    print("Starting Processing Loop (Packed)...")
    
    current_shard = []
    shard_idx = 0
    samples_per_shard = 5000 
    
    # Packing Buffer
    token_buffer = []
    
    total_processed = 0
    
    import torch 
    
    # iterate
    for i, example in enumerate(mixed_dataset):
        if total_processed >= num_samples:
            break
            
        text = example["text"]
        
        # 1. Tokenize
        tokens = tokenizer.encode(text) 
        if len(tokens) < 10: continue # Skip tiny fragments
        
        # 2. Add to buffer with EOS
        token_buffer.extend(tokens)
        token_buffer.append(tokenizer.eos_token_id)
        
        # 3. Drain buffer into chunks
        while len(token_buffer) >= seq_len + 1:
            # Slice rigid chunk
            chunk = token_buffer[:seq_len + 1]
            token_buffer = token_buffer[seq_len + 1:] # Slide window
            
            # Convert to tensor
            input_ids = torch.tensor(chunk, dtype=torch.long)
            
            # JEPA Masking
            target_ids = input_ids.clone()
            noisy_seq = input_ids.clone()
            
            mask_indices = torch.rand(len(noisy_seq)) < 0.15
            random_tokens = torch.randint(0, len(tokenizer), (mask_indices.sum(),))
            noisy_seq[mask_indices] = random_tokens
            
            # Add to shard buffer
            current_shard.append({
                "context_ids": noisy_seq.tolist(),
                "target_ids": target_ids.tolist()
            })
            
            total_processed += 1
            
            if len(current_shard) >= samples_per_shard:
                import pandas as pd
                df = pd.DataFrame(current_shard)
                output_file = os.path.join(output_dir, f"train_data_{shard_idx}.parquet")
                df.to_parquet(output_file)
                print(f"Saved {output_file} ({len(current_shard)} samples) - Total: {total_processed}")
                
                current_shard = []
                shard_idx += 1
    # Final flush
    if len(current_shard) > 0:
        import pandas as pd
        df = pd.DataFrame(current_shard)
        output_file = os.path.join(output_dir, f"train_data_{shard_idx}.parquet")
        df.to_parquet(output_file)
        print(f"Saved {output_file} ({len(current_shard)} samples)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/")
    parser.add_argument("--num_samples", type=int, default=500000) # 500k sequences * 4k ctx = 2B tokens. Good start.
    parser.add_argument("--seq_len", type=int, default=4096)
    args = parser.parse_args()
    
    prepare_mixed_data(args.output_dir, num_samples=args.num_samples, seq_len=args.seq_len)
