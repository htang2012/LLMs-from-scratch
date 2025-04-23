import torch
import torch.nn.functional as F

input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5).softmax(dim=1)
loss = F.cross_entropy(input, target)
print(loss)
loss.backward()


inputsoftmax = torch.softmax(input, dim=-1)
loginsfm = -torch.log(inputsoftmax)
loss =  (target * loginsfm).sum(dim=1).mean()

print(loss)

