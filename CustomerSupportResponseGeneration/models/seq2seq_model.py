import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class Seq2SeqModel:
    def __init__(self, model_name='gpt2'):
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    def generate_response(self, input_ids):
        """
        Generate a response given input_ids.
        
        :param input_ids: Tensor of input IDs.
        :return: Generated response as a string.
        """
        outputs = self.model.generate(input_ids, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def train(self, train_data, epochs=3):
        """
        Train the model on the provided training data.
        
        :param train_data: DataFrame containing 'input_ids' and 'labels'.
        :param epochs: Number of training epochs.
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for _, row in train_data.iterrows():
                input_ids = torch.tensor(row['input_ids']).unsqueeze(0)  # Convert list to tensor and add batch dimension
                labels = torch.tensor(row['labels']).unsqueeze(0)      # Same as above
                
                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_data)}')

    def evaluate(self, test_data):
        """
        Generate responses for test data and evaluate them.
        
        :param test_data: DataFrame containing 'input_ids'.
        :return: List of generated responses.
        """
        self.model.eval()
        responses = []

        for _, row in test_data.iterrows():
            input_ids = torch.tensor(row['input_ids']).unsqueeze(0)  # Convert list to tensor and add batch dimension
            generated_ids = self.model.generate(input_ids)
            generated_response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            responses.append(generated_response)
        
        return responses

    def save(self, path):
        """
        Save the model to the specified path.
        
        :param path: Path to save the model.
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path):
        """
        Load the model from the specified path.
        
        :param path: Path to load the model from.
        """
        self.model = GPT2LMHeadModel.from_pretrained(path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(path)
