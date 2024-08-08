import pandas as pd
from models.seq2seq_model import Seq2SeqModel
from utils.evaluation_metrics import print_evaluation_metrics

def main():
    # Load the preprocessed test data
    test_data = pd.read_csv('../data/test_data.csv')

    # Initialize the model and load the trained weights
    model = Seq2SeqModel()
    model.load('../models/seq2seq_model')

    # Evaluate the model
    responses = model.evaluate(test_data)

    # Print evaluation metrics
    true_responses = [row['response'] for _, row in test_data.iterrows()]
    print_evaluation_metrics(true_responses, responses)

    print("Model evaluation complete.")

if __name__ == "__main__":
    main()
