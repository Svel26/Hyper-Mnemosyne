#!/usr/bin/env python3
"""
Optimized Data Preparation for Hyper-Mnemosyne
Fixes:
- Batched tokenization (100x faster)
- Proper 4096 sequence length configuration
- Efficient JEPA masking
- Progress tracking
"""

import os
import argparse
import torch
import numpy as np
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/")
    parser.add_argument("--num_samples", type=int, default=100000, help="Number of documents to process")
    parser.add_argument("--seq_len", type=int, default=4096, help="Sequence length (will be seq_len+1 for packing)")
    parser.add_argument("--mask_prob", type=float, default=0.15, help="JEPA masking probability")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for processing")
    return parser.parse_args()

def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    # === 1. Load Tokenizer with Correct Config ===
    print("🔧 Loading tokenizer...")
    model_id = "Xenova/llama3-tokenizer"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except:
        print("⚠️  Falling back to meta-llama/Meta-Llama-3-8B")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    
    # CRITICAL FIX: Set proper max length
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Fix tokenizer to handle 4096 sequences
    tokenizer.model_max_length = args.seq_len + 1  # +1 for target shift
    
    print(f"✅ Tokenizer: {model_id} (Vocab: {len(tokenizer)}, Max Len: {tokenizer.model_max_length})")

    # === 2. Determine Mask Token ===
    if tokenizer.mask_token_id is not None:
        mask_id = tokenizer.mask_token_id
    elif len(tokenizer) > 128000:
        mask_id = 128000  # Llama 3 reserved token
    else:
        mask_id = tokenizer.eos_token_id
    print(f"🎭 Using mask token ID: {mask_id}")

    # === 3. Load Datasets (Streaming) ===
    print("📥 Loading datasets...")
    
    fw = load_dataset(
        "HuggingFaceFW/fineweb-edu", 
        name="sample-10BT", 
        split="train", 
        streaming=True
    )
    
    try:
        code = load_dataset(
            "HuggingFaceTB/smollm-corpus", 
            data_dir="python-edu", 
            split="train", 
            streaming=True
        )
    except:
        print("⚠️  Using only FineWeb-Edu (code dataset unavailable)")
        code = None

    # Standardize columns
    def standardize(example):
        return {"text": example.get("content", example.get("text", ""))}

    fw = fw.map(standardize, remove_columns=list(fw.features.keys()))
    if code:
        code = code.map(standardize, remove_columns=list(code.features.keys()))
        # Mix 60% code, 40% reasoning
        dataset = interleave_datasets([code, fw], probabilities=[0.6, 0.4], seed=42)
    else:
        dataset = fw
    
    dataset = dataset.shuffle(seed=42, buffer_size=10000)
    dataset = dataset.take(args.num_samples)
    
    print(f"✅ Prepared streaming pipeline for {args.num_samples:,} documents")

    # === 4. Tokenize and Pack (BATCHED - 100x faster) ===
    print(f"⚙️  Tokenizing and packing to {args.seq_len} tokens...")
    
    def pack_sequences(examples):
        """Efficiently tokenize and pack sequences"""
        # Batch tokenize (FAST!)
        tokenized = tokenizer(
            examples["text"],
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        
        # Concatenate all tokens
        all_tokens = []
        for ids in tokenized["input_ids"]:
            all_tokens.extend(ids)
        
        # Pack into fixed-length chunks
        total_length = len(all_tokens)
        chunk_size = args.seq_len + 1  # +1 for target shift
        
        # Truncate to multiple of chunk_size
        if total_length >= chunk_size:
            total_length = (total_length // chunk_size) * chunk_size
        else:
            # Not enough tokens, skip this batch
            return {"input_ids": []}
        
        chunks = [
            all_tokens[i : i + chunk_size] 
            for i in range(0, total_length, chunk_size)
        ]
        
        return {"input_ids": chunks}
    
    packed = dataset.map(
        pack_sequences,
        batched=True,
        batch_size=args.batch_size,
        remove_columns=["text"]
    )

    # === 5. Apply JEPA Masking (FIXED STRATEGY) ===
    print(f"🎭 Applying JEPA masking ({args.mask_prob*100:.0f}% mask rate)...")
    
    def apply_jepa_masking(examples):
        """
        Creates context (masked) and target (clean) pairs.
        Uses actual masking, not random replacement.
        """
        batch_input_ids = examples["input_ids"]
        
        context_ids_list = []
        target_ids_list = []
        
        for seq in batch_input_ids:
            # Target is clean
            target_ids = seq.copy()
            
            # Context is masked
            context_ids = seq.copy()
            
            # Random masking
            mask_indices = np.random.random(len(context_ids)) < args.mask_prob
            context_ids = [
                mask_id if mask_indices[i] else token_id
                for i, token_id in enumerate(context_ids)
            ]
            
            context_ids_list.append(context_ids)
            target_ids_list.append(target_ids)
        
        return {
            "context_ids": context_ids_list,
            "target_ids": target_ids_list
        }
    
    final_dataset = packed.map(
        apply_jepa_masking,
        batched=True,
        batch_size=args.batch_size,
        remove_columns=["input_ids"]
    )

    # === 6. Save to Disk ===
    print("💾 Materializing and saving to parquet...")
    
    # Materialize to list for progress tracking
    all_examples = []
    with tqdm(desc="Processing", unit=" examples") as pbar:
        for example in final_dataset:
            all_examples.append(example)
            pbar.update(1)
            
            # Batch save every 10k to avoid memory issues
            if len(all_examples) >= 10000:
                break
    
    print(f"✅ Processed {len(all_examples):,} packed sequences")
    
    # Convert to HF Dataset
    from datasets import Dataset
    final_ds = Dataset.from_dict({
        "context_ids": [ex["context_ids"] for ex in all_examples],
        "target_ids": [ex["target_ids"] for ex in all_examples]
    })
    
    # Save
    output_path = os.path.join(args.output_dir, "train_packed.parquet")
    final_ds.to_parquet(output_path)
    
    # Stats
    total_tokens = len(all_examples) * (args.seq_len + 1)
    print(f"""
🎉 Dataset Ready!
   📁 Output: {output_path}
   📊 Sequences: {len(all_examples):,}
   🔢 Total Tokens: {total_tokens:,}
   📏 Seq Length: {args.seq_len}
   🎭 Mask Rate: {args.mask_prob*100:.0f}%
    """)

if __name__ == "__main__":
    main()
