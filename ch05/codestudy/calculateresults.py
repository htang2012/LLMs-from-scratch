import torch
import torch.nn.functional as F

# Example logits tensor of shape [batch_size, vocab_size]
logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])  # Shape [2, 3]
targets = torch.tensor([0, 1])  # Correct classes for each example in the batch

# Step 1: Convert logits to probabilities
probabilities = F.softmax(logits, dim=-1)

# Step 2: Calculate Cross-Entropy Loss
cross_entropy_loss = F.cross_entropy(logits, targets)

# Step 3: Convert logits to log probabilities
log_probabilities = F.log_softmax(logits, dim=-1)

# Step 4: Calculate Negative Log-Likelihood Loss
nll_loss = F.nll_loss(log_probabilities, targets)

# Step 5: Calculate Average Log Probability
# Extract log probabilities of the target classes
target_log_probs = log_probabilities[range(len(targets)), targets]
average_log_probability = target_log_probs.mean()

# Print results
print("Probabilities:\n", probabilities)
print("Cross-Entropy Loss:", cross_entropy_loss.item())
print("Log Probabilities:\n", log_probabilities)
print("Negative Log-Likelihood Loss:", nll_loss.item())
print("Average Log Probability:", average_log_probability.item())
