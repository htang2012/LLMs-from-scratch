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
    # model.eval()
    
    inputs = torch.tensor([[16833, 3626, 6100],
                           [40, 1107, 588]])
    targets = torch.tensor([[3626, 6100, 345 ],
                            [1107, 588, 11311]])
    tokenizer = tiktoken.get_encoding("gpt2")
    with torch.no_grad():
        logits = model(inputs)
    
    probas = torch.softmax(logits, dim = -1)
    print(probas.shape)
    token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    
    text_idx = 0
    example_probas = probas[text_idx]
    example_targets = targets[text_idx]
    positions = [0,1,2]
    selected_targets = example_targets[positions]
    target_probas_1 = example_probas[positions, selected_targets]
    
    
    text_idx = 0
    target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text 1:", target_probas_1)
    
    text_idx = 1
    target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text 2:", target_probas_2)
    
    combined = torch.cat((target_probas_1, target_probas_2))
    
    log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
    print(log_probas.mean())
    
    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()
    print("Flattened logits:", logits_flat.shape)
    print("Flattened targets:", targets_flat.shape)
    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    print(loss)
    
    
    
    '''
    print("Token IDs:\n", token_ids) 
    
    print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
    print(f"Outputs batch 1:"
          f" {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")
    
    print(f"Targets batch 2: {token_ids_to_text(targets[1], tokenizer)}")
    print(f"Outputs batch 2:"
          f" {token_ids_to_text(token_ids[1].flatten(), tokenizer)}")
    
    '''
    
    
if __name__ == "__main__":
    main()




