import torch

class EmbeddingLayer:
    
    def __init__(self, vocab_size, max_length, output_dim):
        self.token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
        self.pos_embedding_layer = torch.nn.Embedding(max_length, output_dim)
        self.max_length = max_length
    
    def Embedding(self, inputs):    
        token_embeddings = self.token_embedding_layer(inputs)
        pos_embeddings = self.pos_embedding_layer(torch.arange(self.max_length))
        return token_embeddings + pos_embeddings
        
