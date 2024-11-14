
import torch
import tiktoken
from modules import GPTModel, GPTDatasetV1



class SimpleTextGeneration:
    def __init__(self, model , idx, max_new_tokens, context_size):
        self.model = model
        self.idx = idx
        self.max_new_tokens = max_new_tokens
        self.context_size = context_size
        
    def generate(self):
            # idx is (B, T) array of indices in the current context
        for _ in range(self.max_new_tokens):

            # Crop current context if it exceeds the supported context size
            # E.g., if LLM supports only 5 tokens, and the context size is 10
            # then only the last 5 tokens are used as context
            idx_cond = self.idx[:, -self.context_size:]

            # Get the predictions
            with torch.no_grad():
                logits = self.model(idx_cond)

            # Focus only on the last time step
            # (batch, n_token, vocab_size) becomes (batch, vocab_size)
            logits = logits[:, -1, :]

            # Get the idx of the vocab entry with the highest logits value
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

            # Append sampled index to the running sequence
            self.idx = torch.cat((self.idx, idx_next), dim=1)  # (batch, n_tokens+1)

        return self.idx
    

def main():
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-Key-Value bias
    }
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()  # disable dropout

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)
   
    gts = SimpleTextGeneration( 
                            model=model,
                            idx = encoded_tensor,
                            max_new_tokens=10,
                            context_size=GPT_CONFIG_124M['context_length'])
    
    out = gts.generate()

    print(out)
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    


    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)
    
    
    
    # Model calculations:
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")
    
    print("Token embedding layer shape:", model.tok_emb.weight.shape)
    print("Output layer shape:", model.out_head.weight.shape)
        
    total_params_gpt2 = (
        total_params - sum(p.numel()
        for p in model.out_head.parameters())
        )
    print(f"Number of trainable parameters "
    f"considering weight tying: {total_params_gpt2:,}"
    )
    
    
    print("Model Layer Parameters:\n")
    for name, layer in model.named_children():
        print(f"{name}: {layer}")
        


if __name__ == "__main__":
    main()
