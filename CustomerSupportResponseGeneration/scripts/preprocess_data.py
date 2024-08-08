import pandas as pd
from transformers import GPT2Tokenizer
from utils.dataset_loader import load_and_preprocess_data, split_data

def main():
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Load and preprocess the dataset
    file_path = '../data/customer_support_responses.csv'
    df = load_and_preprocess_data(file_path, tokenizer)

    # Split the data
    train_data, test_data = split_data(df)

    # Save the preprocessed data
    train_data.to_csv('../data/train_data.csv', index=False)
    test_data.to_csv('../data/test_data.csv', index=False)

    print("Data preprocessing complete. Preprocessed data saved to '../data/'")

if __name__ == "__main__":
    main()
