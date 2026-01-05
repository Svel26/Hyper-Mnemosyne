import os
import argparse
import pandas as pd
import torch
import random
import numpy as np
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm

def apply_jepa_masking(tokens, tokenizer, mask_prob=0.15, span_prob=0.1):
    """
    Apply noise for the Context view.
    1. Random Token Masking (replace with <mask> or random token)
    2. Span Deletion (drop a chunk of text)
    """
    tokens = list(tokens) # Copy
    length = len(tokens)
    
    # 1. Span Deletion (Simulate missing information)
    if random.random() < span_prob:
        span_len = random.randint(10, length // 4)
        start = random.randint(0, length - span_len)
        # We just remove them for the context view, making it shorter/broken
        # Or better: replace with a single sentinel if we had one.
        # For this architecture (Mamba), just removing them forces it to predict the 'jump'.
        del tokens[start:start+span_len]
        
    # 2. Random Token Masking
    # Since we don't have a special Mask token in GPT2 usually, we can replace with 
    # a random token or a specific specialized token if we added one.
    # Let's replace with the EOS token as a proxy for "unknown/noise" or random tokens.
    vocab_size = len(tokenizer)
    for i in range(len(tokens)):
        if random.random() < mask_prob:
            rand = random.random()
            if rand < 0.8:
                # Replace with something random
                tokens[i] = random.randint(0, vocab_size-1)
            elif rand < 0.9:
                # Replace with generic "UNK" proxy (EOS)
                tokens[i] = tokenizer.eos_token_id
            # else: keep original (10%)
            
    return tokens

def prepare_fineweb(output_dir, seq_len=1024, num_samples=100_000, chunk_size=10_000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("Streaming FineWeb-Edu dataset (sample)...")
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    
    print("Initializing Tokenizer (GPT2)...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    data = []
    total_generated = 0
    file_idx = 0
    
    iterator = iter(dataset)
    
    print(f"Processing {num_samples} samples with JEPA masking...")
    
    with tqdm(total=num_samples) as pbar:
        while total_generated < num_samples:
            try:
                sample = next(iterator)
                text = sample['text']
            except StopIteration:
                break
                
            if len(text) < 500: 
                continue
                
            tokens = tokenizer.encode(text)
            
            for i in range(0, len(tokens) - seq_len, seq_len):
                clean_seq = tokens[i : i + seq_len]
                if len(clean_seq) < seq_len:
                    continue
                    
                noisy_seq = apply_jepa_masking(clean_seq, tokenizer)
                
                if len(noisy_seq) < seq_len:
                    noisy_seq += [tokenizer.eos_token_id] * (seq_len - len(noisy_seq))
                else:
                    noisy_seq = noisy_seq[:seq_len]
                
                data.append({
                    "context_ids": noisy_seq,
                    "target_ids": clean_seq
                })
                total_generated += 1
                pbar.update(1)
                
                if len(data) >= chunk_size:
                    # Save chunk
                    df = pd.DataFrame(data)
                    output_file = os.path.join(output_dir, f"train_data_{file_idx}.parquet")
                    df.to_parquet(output_file)
                    print(f"Saved {output_file}")
                    data = [] # Clear memory
                    file_idx += 1
                
                if total_generated >= num_samples:
                    break
            
    # Save remaining
    if data:
        df = pd.DataFrame(data)
        output_file = os.path.join(output_dir, f"train_data_{file_idx}.parquet")
        df.to_parquet(output_file)
        print(f"Saved {output_file}")
        
    print(f"Total generated: {total_generated}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/")
    parser.add_argument("--num_samples", type=int, default=20000)
    args = parser.parse_args()
    
    prepare_fineweb(args.output_dir, num_samples=args.num_samples)
