import torch
import torch.nn as nn



class SimpleSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        atten_score = x @ x.T
        print(f"atten_score:\n {atten_score}")
        atten_weights = torch.softmax(atten_score, dim=1)
        #atten_weights1= torch.nn.functional.softmax(atten_score, dim=1)
        print(f"atten_weights:\n{atten_weights}")
        #print(f"atten_weights1:\n{atten_weights1}")
        
        context_vectors = atten_weights @ x
        return context_vectors
    
    
class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out,bias=qkv_bias) 
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias) 
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias) 
    
        
    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        print(f"attn_weights:\n{attn_weights}")
        context_vectors = attn_weights @ values
        return context_vectors


class CasualAttention(SelfAttention):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__(d_in, d_out, qkv_bias)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        keys = self.W_key(x)          # Shape: (batch_size, seq_len, d_out)
        queries = self.W_query(x)     # Shape: (batch_size, seq_len, d_out)
        values = self.W_value(x)      # Shape: (batch_size, seq_len, d_out)
        
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1))  # Shape: (batch_size, seq_len, seq_len)
        
        # Scale the attention scores
        scaling_factor = keys.size(-1) ** 0.5
        attn_scores = attn_scores / scaling_factor
        
        # Create a causal mask (lower triangular matrix)
        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).unsqueeze(0)  # Shape: (1, seq_len, seq_len)
        
        # Apply the mask: set positions where mask == 0 to -inf
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        print(f"masked attn_scores: \n {attn_scores}")
        
        # Compute attention weights with the masked scores
        attn_weights = torch.softmax(attn_scores, dim=-1)
        print(f"attn_weights with causal mask:\n{attn_weights}")
        
        # Compute context vectors
        context_vectors = torch.matmul(attn_weights, values)  # Shape: (batch_size, seq_len, d_out)
        return context_vectors
        
        