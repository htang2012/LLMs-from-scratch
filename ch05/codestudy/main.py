
import torch
import tiktoken
from modules import GPTModel, GPTDatasetV1, SimpleTextGeneration, text_to_token_ids, token_ids_to_text




def main():
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 256,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-Key-Value bias
    }
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()
    
    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")
   

    idx = text_to_token_ids(start_context, tokenizer)
    
    gts = SimpleTextGeneration( 
                            model=model,
                            idx = idx,
                            max_new_tokens=10,
                            context_size=GPT_CONFIG_124M['context_length'])
    
    token_ids = gts.generate()
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
    
    
    
if __name__ == "__main__":
    main()




