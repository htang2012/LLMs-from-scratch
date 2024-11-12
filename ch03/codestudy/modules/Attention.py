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
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out)) 
        self.W_key = nn.Parameter(torch.rand(d_in, d_out)) 
        self.W_value = nn.Parameter(torch.rand(d_in, d_out)) 
    
        
    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vectors = attn_weights @ values
        return context_vectors
        