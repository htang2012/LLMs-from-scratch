import torch
from modules import SimpleSelfAttention, SelfAttention, CasualAttention, MultiHeadAttentionWrapper, MultiHeadAttention

def main():
    inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89], # Your
        [0.55, 0.87, 0.66], # journey
        [0.57, 0.85, 0.64], # starts
        [0.22, 0.58, 0.33], # with
        [0.77, 0.25, 0.10], # one
        [0.05, 0.80, 0.55]  # step
    ] )  
    
    simple_self_attention = SimpleSelfAttention()
    contextvectors = simple_self_attention(inputs)
    print(f"Simple Attention contextvectors: \n {contextvectors}, {contextvectors.shape}")
    
    torch.manual_seed(789)
    self_attention = SelfAttention(3, 2)
    contextvectors = self_attention(inputs)
    print(f"Attention contextvectors: \n {contextvectors}, {contextvectors.shape}")
    
    torch.manual_seed(123)
    input_batch = torch.stack((inputs, inputs), dim=0)
    print(input_batch.shape)
    context_length = input_batch.shape[1]
    causal_attention = CasualAttention(3, 2, context_length, 0.0)
    contextvectors = causal_attention(input_batch)
    print(f"Attention contextvectors: \n {contextvectors}, {contextvectors.shape}")
    
    torch.manual_seed(123)
    context_length = input_batch.shape[1]
    d_in, d_out = 3, 2 
    mha = MultiHeadAttentionWrapper(
          d_in, d_out, context_length, 0.0, num_heads=2
        )
    context_vecs = mha(input_batch)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)
    
    
    torch.manual_seed(123)
    batch_size, context_length, d_in = input_batch.shape
    d_out = 2
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vecs = mha(input_batch)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)
    

if __name__ == '__main__':
    main()