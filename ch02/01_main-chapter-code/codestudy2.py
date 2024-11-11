import torch
'''
vocab_size = 6
output_dim = 3

torch.manual_seed(123)
device = torch.device('cuda')
embedding_layer = torch.nn.Embedding(vocab_size, output_dim).to(device)
print(embedding_layer.weight)
print(embedding_layer(torch.tensor([2,3,5,1]).to(device)))
'''


import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
tokens = tokenizer.encode("hello,world")
print(tokens)
text = tokenizer.decode(tokens)
print(text)

vocab_size = tokenizer.n_vocab
output_dim = 768


torch.manual_seed(123)
device = torch.device('cuda')
embedding_layer = torch.nn.Embedding(vocab_size, output_dim).to(device)
print(embedding_layer.weight)
print(embedding_layer(torch.tensor([2,3,5,1]).to(device)))



