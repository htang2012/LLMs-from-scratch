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
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__(d_in, d_out, qkv_bias)
        self.dropout = nn.Dropout(dropout) 
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) # New
    
    def forward(self, x):
        batch_size, seq_len, d_in = x.size()      #Shape: (1, seq_len, d_in)
        keys = self.W_key(x)           # Shape: (batch_size, seq_len, d_out)
        queries = self.W_query(x)      # Shape: (batch_size, seq_len, d_out)
        values = self.W_value(x)       # Shape: (batch_size, seq_len, d_out)
        
        attn_scores = queries @ keys.transpose(1,2) # Shape: (batch_size, seq_len, seq_len)
        
        attn_scores.masked_fill_( 
                    self.mask.bool()[:seq_len, :seq_len], -torch.inf)
        
        # Scale the attention scores
        scaling_factor = keys.size(-1) ** 0.5
        attn_weights = torch.softmax(
                attn_scores / scaling_factor, dim= -1)
        
        attn_weights = self.dropout(attn_weights)
        
        # Compute context vectors
        context_vectors = attn_weights @ values  # Shape: (batch_size, seq_len, d_out)
        return context_vectors
        
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length,
                 dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CasualAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
        
        
        
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, 
                 context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
            diagonal=1)
        )
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(
            b, num_tokens, self.num_heads, self.head_dim
        )
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )
        context_vec = self.out_proj(context_vec)
        return context_vec