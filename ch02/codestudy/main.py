from modules import EmbeddingLayer, GPTDataLoader


vocab_size = 50257
max_length = 4 
output_dim = 2 #16 #256
batch_size = 8



def main():
    with open("/workspaces/ch02/01_main-chapter-code/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
        gptdataloader = GPTDataLoader(raw_text, batch_size=batch_size, max_length=max_length, stride=max_length)
    dataloader = gptdataloader.getDataLoader()
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Token IDs:\n", inputs)
    print("\nInputs shape:\n", inputs.shape) 
    
    embeddinglayer = EmbeddingLayer(vocab_size=vocab_size, 
                                    max_length = max_length,
                                    output_dim= output_dim)
    
    embedding = embeddinglayer.Embedding(inputs)
    print(f"embedding: {embedding}" )


if __name__ == '__main__':
    main()
      
