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
        context_vectors = attn_weights @ values
        return context_vectors
