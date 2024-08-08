import pandas as pd
from transformers import GPT2Tokenizer

def load_and_preprocess_data(file_path, tokenizer):
    """
    Load and preprocess the dataset.

    :param file_path: Path to the dataset CSV file.
    :param tokenizer: Pre-trained tokenizer.
    :return: Preprocessed DataFrame with input_ids and labels.
    """
    df = pd.read_csv(file_path)

    input_ids = []
    labels = []

    for _, row in df.iterrows():
        query = row['query']
        response = row['response']

        input_id = tokenizer.encode(query, return_tensors='pt').squeeze()
        label = tokenizer.encode(response, return_tensors='pt').squeeze()

        input_ids.append(input_id.tolist())
        labels.append(label.tolist())

    df['input_ids'] = input_ids
    df['labels'] = labels

    return df

def split_data(df, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.

    :param df: Preprocessed DataFrame.
    :param test_size: Proportion of the dataset to include in the test split.
    :param random_state: Random seed for reproducibility.
    :return: Training and testing DataFrames.
    """
    from sklearn.model_selection import train_test_split

    train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_data, test_data
