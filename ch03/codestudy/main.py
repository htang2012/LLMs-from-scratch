import torch
from modules import SimpleSelfAttention, SelfAttention, CasualAttention

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
    
    torch.manual_seed(789)
    causal_attention = CasualAttention(3, 2)
    inputs = inputs.unsqueeze(0)
    contextvectors = causal_attention(inputs)
    print(f"Attention contextvectors: \n {contextvectors}, {contextvectors.shape}")
    
    

if __name__ == '__main__':
    main()