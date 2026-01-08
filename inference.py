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
    
    # Try importing InferenceParams
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
    
    # Mamba Caching logic disabled due to 'AssertionError: Only support decoding with 1 token at a time'
    # during prefill. Using Titans State passing only.
    # if InferenceParams is not None:
    #    inference_params = InferenceParams(max_seqlen=config.max_seq_len, max_batch_size=1)
    #    inference_params.seqlen_offset = 0 # Force reset
    #    print(f"InferenceParams initialized. Offset: {inference_params.seqlen_offset}")
        
    with torch.no_grad():
        # 1. Prefill Stage (Process whole prompt)
        print(f"Prefill input shape: {input_ids.shape}")
        # Pass inference_params=None
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
            # Pass ONLY the last token for Titans (if we could), but Mamba needs full context if not cached.
            # So we pass the FULL sequence 'generated' to model, but we can pass 'titans_state' to update it?
            # PROBLEM: If we pass full sequence, Titans will process full sequence and double-update state.
            
            # Correction: If we can't cache Mamba, we must pass FULL sequence.
            # But Titans is now stateful. If we pass full sequence, Titans memory will see old tokens again.
            
            # Hybrid Solution:
            # We must stick to the "Baby Mode" loop (pass full sequence) to satisfy Mamba.
            # Consequently, we CANNOT use the stateful Titans update in this loop properly without resetting it every time.
            # OR, we disable Titans state passing for this loop and let it re-compute too.
            
            # Reverting: We pass titans_state=None and inference_params=None. 
            # It's inefficient but correct.
            
            # Wait, if Titans is stateful (EMA), re-computing full sequence changes the state result vs single pass?
            # Yes, EMA of (a, b, c) is different if you feed (a), then (a, b), then (a, b, c).
            # Actually, EMA is path dependent.
            # If TitansMemoryLayer treats memory_state=None as 0, then:
            # Step 1: Feed (a, b). State becomes f(a, b).
            # Step 2: Feed (a, b, c). State becomes f(a, b, c). 
            # Correct. It works fine statelessly (re-computed) as long as we don't double-feed.
            
            # Logits calculation
            # Truncate if too long
            if generated.size(1) > config.max_seq_len:
                 run_input = generated[:, -config.max_seq_len:]
            else:
                 run_input = generated
                 
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
