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
    
    # Note: inference_params for Mamba disabled due to prefill assertion errors in current version.
    
    with torch.no_grad():
        # 1. Prefill Stage (Process whole prompt)
        print(f"Prefill input shape: {input_ids.shape}")
        # Pass inference_params=None, titans_state=None (init)
        logits, _, _, titans_state = model(input_ids, inference_params=None, titans_state=titans_state)
        next_token_logits = logits[:, -1, :]
        
        # Sampling (Prefill)
        next_token_logits = next_token_logits / 0.8 
        top_k = 50
        v, i = torch.topk(next_token_logits, top_k)
        probs = torch.nn.functional.softmax(v, dim=-1)
        next_token_idx = torch.multinomial(probs, 1)
        next_token = i.gather(-1, next_token_idx)
        
        generated = torch.cat([generated, next_token], dim=1)
        
        # 2. Decoding Stage (Token by Token)
        for _ in range(args.max_new_tokens - 1):
             # Logits calculation
             # Truncate if too long
             if generated.size(1) > config.max_seq_len:
                  run_input = generated[:, -config.max_seq_len:]
             else:
                  run_input = generated
                  
             # Naive loop: Recompute Mamba (Stateless) and Titans (Stateless) for stability
             # We pass titans_state=None because if we pass the *updated* state + the *full* sequence,
             # Titans will update the state again using the old history = double counting.
             # Since Mamba requires full sequence (no cache), we must feed full sequence.
             # Therefore, we must effectively treat Titans as stateless here too (re-scan from scratch).
             logits, _, _, _ = model(run_input, inference_params=None, titans_state=None)
             
             next_token_logits = logits[:, -1, :]
             
             # Sampling
             next_token_logits = next_token_logits / 0.8 
             v, i = torch.topk(next_token_logits, top_k)
             probs = torch.nn.functional.softmax(v, dim=-1)
             next_token_idx = torch.multinomial(probs, 1)
             next_token = i.gather(-1, next_token_idx)
             
             generated = torch.cat([generated, next_token], dim=1)
             
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
