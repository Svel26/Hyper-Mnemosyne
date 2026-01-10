
import os
import multiprocessing
import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

# CONFIG
MODEL_ID = "gpt2"
SEQ_LEN = 4096
NUM_PROC = 4 
OUTPUT_DIR = "data/"

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.model_max_length = 100000 # Silence warnings

    print(f"🚀 Starting Multi-Threaded Prep on {multiprocessing.cpu_count()} cores...")

    # 1. Load Data
    print("📥 Loading datasets...")
    # Load FineWeb-Edu (Reasoning)
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True).take(200000)
    
    # Load Code
    # Trying python-edu first
    try:
        code = load_dataset("HuggingFaceTB/smollm-corpus", data_dir="python-edu", split="train", streaming=True).take(300000)
    except:
        print("Fallback to cosmos_corpus or similar not implemented, assuming smollm works.")
        code = load_dataset("HuggingFaceTB/smollm-corpus", data_dir="python-edu", split="train", streaming=True).take(300000)

    print("💾 Materializing data to RAM (The Blender)...")
    fw_texts = [x['text'] for x in fw]
    # Smollm has 'content' usually, let's check field names or just use get
    code_texts = []
    for x in code:
        code_texts.append(x.get("content", x.get("text", "")))
    
    # MIX THEM RAW
    # 300k code + 200k reasoning = 500k docs
    raw_data = fw_texts + code_texts
    
    # Create HF Dataset
    dataset = Dataset.from_dict({"text": raw_data})
    
    # Shuffle
    dataset = dataset.shuffle(seed=42)
    print(f"✅ Loaded {len(dataset)} raw documents.")

    # 2. Packing Logic
    def group_texts(examples):
        # Concatenate
        concatenated = tokenizer(examples["text"], add_special_tokens=True, truncation=False)["input_ids"]
        all_tokens = [token for sublist in concatenated for token in sublist]
        
        total_length = len(all_tokens)
        if total_length >= SEQ_LEN + 1:
            total_length = (total_length // (SEQ_LEN + 1)) * (SEQ_LEN + 1)
            
        chunks = [all_tokens[i : i + SEQ_LEN + 1] for i in range(0, total_length, SEQ_LEN + 1)]
        
        return {
            "input_ids": chunks
        }

    print("⚙️  Tokenizing and Packing (16 Threads)...")
    packed_dataset = dataset.map(
        group_texts,
        batched=True,
        batch_size=1000, 
        num_proc=NUM_PROC,
        remove_columns=["text"]
    )
    
    print(f"✅ Packed into {len(packed_dataset)} sequences.")

    # 3. JEPA Masking Logic
    def apply_jepa_masking(examples):
        # inputs are lists of lists
        batch_input_ids = examples["input_ids"]
        
        context_ids_list = []
        target_ids_list = []
        
        for seq in batch_input_ids:
            input_tensor = torch.tensor(seq, dtype=torch.long)
            
            # Target is just clean input (clone)
            target_ids = input_tensor.clone()
            
            # Context is noisy
            noisy_seq = input_tensor.clone()
            
            mask_prob = 0.15
            mask_indices = torch.rand(len(noisy_seq)) < mask_prob
            random_tokens = torch.randint(0, len(tokenizer), (mask_indices.sum(),))
            noisy_seq[mask_indices] = random_tokens
            
            context_ids_list.append(noisy_seq.tolist())
            target_ids_list.append(target_ids.tolist())
            
        return {
            "context_ids": context_ids_list,
            "target_ids": target_ids_list
        }

    print("🎭 Applying JEPA Masking...")
    final_dataset = packed_dataset.map(
        apply_jepa_masking,
        batched=True,
        batch_size=1000,
        num_proc=NUM_PROC,
        remove_columns=["input_ids"] # We don't need 'input_ids' anymore, just ctx/tgt
    )

    # 4. Save to Disk
    print("💾 Saving to parquet...")
    # Saving as multiple files (shards) automatically if large
    output_path = os.path.join(OUTPUT_DIR, "train_packed.parquet")
    final_dataset.to_parquet(output_path)
    print("🎉 Done! Ready to train.")

if __name__ == "__main__":
    main()
