
import torch

class SimpleTextGeneration:
    def __init__(self, model , idx, max_new_tokens, context_size):
        self.model = model
        self.idx = idx
        self.max_new_tokens = max_new_tokens
        self.context_size = context_size
        
    def generate(self):
            # idx is (B, T) array of indices in the current context
        for _ in range(self.max_new_tokens):

            # Crop current context if it exceeds the supported context size
            # E.g., if LLM supports only 5 tokens, and the context size is 10
            # then only the last 5 tokens are used as context
            idx_cond = self.idx[:, -self.context_size:]

            # Get the predictions
            with torch.no_grad():
                logits = self.model(idx_cond)

            # Focus only on the last time step
            # (batch, n_token, vocab_size) becomes (batch, vocab_size)
            logits = logits[:, -1, :]

            # Get the idx of the vocab entry with the highest logits value
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

            # Append sampled index to the running sequence
            self.idx = torch.cat((self.idx, idx_next), dim=1)  # (batch, n_tokens+1)

        return self.idx
    