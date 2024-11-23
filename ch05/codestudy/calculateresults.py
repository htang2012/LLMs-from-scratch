import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Logits from the model (batch size = 3, 3 classes)
logits = torch.tensor([
    [2.0, 1.0, 0.1],  # Example 1
    [0.5, 2.0, 0.3],  # Example 2
    [0.3, 0.5, 1.7]   # Example 3
])

targets = torch.tensor([0, 1, 2])  # Ground truth for each example

criterion = nn.CrossEntropyLoss()

loss = criterion(logits, targets)
print(f"Cross Entropy Loss: {loss.item():.4f}")


output = F.nll_loss(F.log_softmax(logits, dim=1), targets)
print(output)
