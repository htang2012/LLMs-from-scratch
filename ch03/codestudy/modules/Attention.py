import torch



class SimpleSelfAttention:
    def __init__(self):
        pass
    def contextvectors(self, inputs):
        atten_score = inputs @ inputs.T
        print(f"atten_score:\n {atten_score}")
        atten_weights = torch.softmax(atten_score, dim=1)
        print(f"atten_weights:\n{atten_weights}")
        context_vectors = atten_weights @inputs
        return context_vectors
    
    

   