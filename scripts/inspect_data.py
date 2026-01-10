import pandas as pd
import torch
from transformers import GPT2TokenizerFast

def inspect():
    print("🔍 Loading data/train_packed.parquet...")
    df = pd.read_parquet("data/train_packed.parquet")
    print(f"✅ Loaded {len(df)} samples.")
    
    row = df.iloc[0]
    ctx = row['context_ids']
    tgt = row['target_ids']
    
    print(f"📏 Sequence Length: {len(ctx)} (Expected: 4096)")
    
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    print("\n--- Context (Input with JEPA Noise) [First 200 chars] ---")
    print(tokenizer.decode(ctx)[:200])
    
    print("\n--- Target (Clean Label) [First 200 chars] ---")
    print(tokenizer.decode(tgt)[:200])
    
    # Verify Difference
    diff = sum([1 for c, t in zip(ctx, tgt) if c != t])
    print(f"\n🎭 JEPA Masking Utility: {diff} tokens distinct out of {len(ctx)} ({diff/len(ctx)*100:.2f}%)")

if __name__ == "__main__":
    inspect()
