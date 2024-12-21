from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.utils.data import DataLoader

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Function to encode and decode text
def encode_decode_text(text):
    encoded_text = tokenizer.encode(text)
    decoded_text = tokenizer.decode(encoded_text)
    return encoded_text, decoded_text
'''
# Sample text
sample_text = "Hello, how are you?"

# Encode and decode the sample text
encoded_sample_text, decoded_sample_text = encode_decode_text(sample_text)
print(f"Original Sample Text: {sample_text}")
print(f"Encoded Sample Text: {encoded_sample_text}")
print(f"Decoded Sample Text: {decoded_sample_text}")

# Open and read the content of the-verdict.txt
with open("/workspace/Tokenizer/the-verdict.txt", "r") as file:
    verdict_text = file.read()

# Encode and decode the verdict text
encoded_verdict_text, decoded_verdict_text = encode_decode_text(verdict_text)

# Display a portion of the content
print("Original Text (First 500 characters):")
print(verdict_text[:500])
print("\nEncoded Text (First 50 tokens):")
print(encoded_verdict_text[:50])
print("\nDecoded Text (First 500 characters):")
print(decoded_verdict_text[:500])
'''

# Load a dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
train_dataset = dataset['train']
validation_dataset = dataset['validation']
test_dataset = dataset['test']

print(f"Number of training examples: {len(train_dataset)}")
print(f"Number of validation examples: {len(validation_dataset)}")
print(f"Number of test examples: {len(test_dataset)}")
# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)


tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_validation_dataset = validation_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Create a data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


batch_size = 8
# Create DataLoaders
train_dataloader = DataLoader(tokenized_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
validation_dataloader = DataLoader(tokenized_validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
test_dataloader = DataLoader(tokenized_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

# Example of iterating through the train data loader
for i, batch in enumerate(train_dataloader):
    print(f"Batch {i+1}: {batch}")
    if i == 1:  # Stop after printing the first two batches
        break