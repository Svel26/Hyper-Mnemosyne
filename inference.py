import torch
import argparse
from transformers import GPT2TokenizerFast
from config import HyperMnemosyneConfig
from model.backbone import HyperMnemosyne

def generate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Config
    config = HyperMnemosyneConfig()
    # Ensure config matches training (safe to hardcode for this demo unless verified)
    # Ensure config matches training (safe to hardcode for this demo unless verified)
    # config.max_seq_len = 4096 # Config defaults are now correct from config.py 
    
    # Load Model
    print("Loading model architecture...")
    model = HyperMnemosyne(config)
    
    print(f"Loading weights from {args.model_path}...")
    try:
        state_dict = torch.load(args.model_path, map_location=device)
        
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
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    # Encode Prompt
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    
    print(f"Generating from prompt: '{args.prompt}'")
    
    # Simple Greedy Decoding loop
    # (For better quality, implement Top-K/Top-P samplers)
    generated = input_ids
    
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            # Forward
            # Truncate if too long
            if generated.size(1) > config.max_seq_len:
                generated = generated[:, -config.max_seq_len:]
                
            logits, _, _ = model(generated)
            # Get last token logits
            next_token_logits = logits[:, -1, :]
            
            # Apply Temperature (Higher = crazier, Lower = safer)
            next_token_logits = next_token_logits / 0.8 
            
            # Top-K Sampling
            top_k = 50
            v, i = torch.topk(next_token_logits, top_k)
            probs = torch.nn.functional.softmax(v, dim=-1)
            next_token_idx = torch.multinomial(probs, 1)
            next_token = i.gather(-1, next_token_idx)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if EOS (optional, depending on training)
            if next_token.item() == tokenizer.eos_token_id:
                break
                
    # Decode
    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print("-" * 40)
    print(output_text)
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="model_final.pt")
    parser.add_argument("--prompt", type=str, default="The quick brown fox")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    args = parser.parse_args()
    
    generate(args)
