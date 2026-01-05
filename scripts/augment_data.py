import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
import argparse
import os
from tqdm import tqdm

def load_rewriter_model(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """
    Load a small/quantized model for rewriting.
    Defaulting to TinyLlama for speed/compatibility, but user can override.
    """
    print(f"Loading {model_name}...")
    
    # Check for quantization support
    try:
        import bitsandbytes
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        print("Using 4-bit quantization.")
    except ImportError:
        print("bitsandbytes not found, loading in fp16.")
        quantization_config = None
        
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16 if quantization_config is None else None
    )
    
    return model, tokenizer

def rewrite_text(text, model, tokenizer):
    """
    Generate a rewrite of the text.
    """
    prompt = (
        "<|system|>\n"
        "You are a helpful assistant. Rewrite the following text to be more formal and academic, keeping the meaning exactly the same.\n"
        "</s>\n"
        "<|user|>\n"
        f"{text}\n"
        "</s>\n"
        "<|assistant|>\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=len(inputs.input_ids[0]) + 100, # Heuristic
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract the assistant response (naive parsing depending on model chat format)
    # TinyLlama format: ... <|assistant|>\n RESPONSE
    
    # Simple split heuristic for now
    if "<|assistant|>" in output_text:
        return output_text.split("<|assistant|>")[-1].strip()
    
    # Fallback: remove prompt
    return output_text.replace(prompt.replace("<|system|>", "").replace("<|user|>", "").replace("</s>", ""), "").strip()

def augment_file(input_file, output_file, model, tokenizer):
    df = pd.read_parquet(input_file)
    
    new_rows = []
    print(f"Augmenting {len(df)} rows from {input_file}...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Assume 'target_ids' were original text? 
        # Actually our parquet stores IDs. We need text.
        # This script assumes we have a source text file or generic dataset first.
        # But prepare_fineweb saves parquet with 'context_ids' and 'target_ids'.
        # We can't easily reverse IDs to Text perfectly without the exact tokenizer used there.
        # Ideally, this augmentation step happens BEFORE tokenization.
        
        # Let's assume input_file is a raw parquet with a 'text' column, like FineWeb chunk.
        text = row.get('text', "")
        if not text:
            continue
            
        rewritten = rewrite_text(text, model, tokenizer)
        
        new_rows.append({
            'original_text': text,
            'rewritten_text': rewritten
        })
        
        if idx > 100: # Limit for prototype
             break
             
    # Save
    pd.DataFrame(new_rows).to_parquet(output_file)
    print(f"Saved {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Parquet file with 'text' column")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    args = parser.parse_args()
    
    model, tokenizer = load_rewriter_model(args.model)
    augment_file(args.input_file, args.output_file, model, tokenizer)
