import pandas as pd
from models.seq2seq_model import Seq2SeqModel

def main():
    # Load the preprocessed training data
    train_data = pd.read_csv('../data/train_data.csv')

    # Initialize and train the model
    model = Seq2SeqModel()
    model.train(train_data, epochs=3)

    # Save the trained model
    model.save('../models/seq2seq_model')

    print("Model training complete. Trained model saved to '../models/seq2seq_model'")

if __name__ == "__main__":
    main()
