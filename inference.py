import torch
import argparse
import sys
import os
from transformers import AutoTokenizer

# Add project root to path if not installed as package
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import HyperMnemosyneConfig
from model.backbone import HyperMnemosyne

def generate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Config
    config = HyperMnemosyneConfig()
    
    # Load Model
    print("Loading model architecture...")
    model = HyperMnemosyne(config)
    
    print(f"Loading weights from {args.model_path}...")
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
            
        # Handle torch.compile prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace("_orig_mod.", "")
            new_state_dict[new_k] = v
            
        model.load_state_dict(new_state_dict)
    except FileNotFoundError:
        print(f"Error: Could not find {args.model_path}. Did you run training?")
        return

    model.to(device)
    model.eval()
    
    # Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("Xenova/llama3-tokenizer")
    except:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    
    print(f"🔍 Inference Tokenizer Vocab Size: {len(tokenizer)}")
    
    # Try importing InferenceParams (Keeping for future reference, but disabled logic below)
    try:
        from mamba_ssm.utils.generation import InferenceParams
    except ImportError:
        print("Warning: mamba_ssm not found or InferenceParams missing. Caching disabled.")
        InferenceParams = None

    # Encode Prompt
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    
    print(f"Generating from prompt: '{args.prompt}'")
    
    generated = input_ids
    
    # State Management
    titans_state = None
    inference_params = None
    
    # Initialize InferenceParams for Mamba if available
    if InferenceParams is not None:
        inference_params = InferenceParams(max_seqlen=config.max_seq_len, max_batch_size=1)

    print("-" * 40)
    print(f"Generating from prompt: '{args.prompt}'")
    print("-" * 40)
    # Print prompt first
    print(args.prompt, end="", flush=True)
    
    with torch.no_grad():
        # 1. Prefill Stage (Process whole prompt)
        # Pass inference_params (if Mamba) and titans_state=None (init)
        
        # NOTE: Mamba2 requires seqlen_offset to be set manually if not using their generation driver
        # But InferenceParams handles it if we update it? 
        # Actually standard usage: 
        # prefill: inference_params.seqlen_offset = 0
        
        logits, _, _, titans_state = model(input_ids, inference_params=inference_params, titans_state=None)
        next_token_logits = logits[:, -1, :]
        
        # Sampling (Prefill)
        next_token_logits = next_token_logits / args.temperature 
        top_k = 50
        v, i = torch.topk(next_token_logits, top_k)
        probs = torch.nn.functional.softmax(v, dim=-1)
        next_token_idx = torch.multinomial(probs, 1)
        next_token = i.gather(-1, next_token_idx)
        
        generated = torch.cat([generated, next_token], dim=1)
        
        # Determine offset for decoding
        seqlen_offset = input_ids.shape[1]
        if inference_params is not None:
             inference_params.seqlen_offset = seqlen_offset
        
        # Stream first token
        word = tokenizer.decode(next_token[0])
        print(word, end="", flush=True)
        
        # 2. Decoding Stage (Token by Token)
        curr_token = next_token
        
        for _ in range(args.max_new_tokens - 1):
             # O(1) Inference: Only pass the LAST token
             # Mamba uses inference_params cache.
             # Titans uses titans_state.
             
             # Step Mamba Cache
             if inference_params is not None:
                 inference_params.seqlen_offset += 1
                 
             logits, _, _, titans_state = model(curr_token, inference_params=inference_params, titans_state=titans_state)
             
             next_token_logits = logits[:, -1, :]
             
             # Sampling
             next_token_logits = next_token_logits / args.temperature 
             v, i = torch.topk(next_token_logits, top_k)
             probs = torch.nn.functional.softmax(v, dim=-1)
             next_token_idx = torch.multinomial(probs, 1)
             next_token = i.gather(-1, next_token_idx)
             
             generated = torch.cat([generated, next_token], dim=1)
             curr_token = next_token
             
             # Stream
             word = tokenizer.decode(curr_token[0])
             print(word, end="", flush=True)
             
             if curr_token.item() == tokenizer.eos_token_id:
                 break
                
    print("\n" + "-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="model_final.pt")
    parser.add_argument("--prompt", type=str, default="The quick brown fox")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    
    generate(args)
