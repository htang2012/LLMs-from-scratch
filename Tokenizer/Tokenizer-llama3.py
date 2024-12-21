from transformers import AutoTokenizer

# Load the tokenizer
# This tokenizer is likely using Byte Pair Encoding (BPE) or a similar subword tokenization method,
# as it is common for models like LLaMA to use such techniques for efficient tokenization.
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")


# Sample text
text = "Hello, how are you"

# Add special tokens
bos_token = tokenizer.bos_token or "<s>"
eos_token = tokenizer.eos_token or "</s>"
text_with_special_tokens = f"{bos_token} {text} {eos_token}"

# Tokenize the text
# The encode method converts the input text into a list of token IDs.
encoded_text = tokenizer.encode(text)
print(f"Encoded text: {encoded_text}")

# Decode the text
# The decode method converts the list of token IDs back into a human-readable string.
decoded_text = tokenizer.decode(encoded_text)
print(f"Decoded text: {decoded_text}")