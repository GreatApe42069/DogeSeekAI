import flower as flwr
import torch
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer
from datasets import load_dataset

class DogeSeekAIClient(flwr.client.NumPyClient):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = load_dataset("squad", split="train[:1000]")

    def get_parameters(self):
        return [val.cpu().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        for param, val in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(val)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(output_dir="./dogeseekai", num_train_epochs=1),
            train_dataset=self.dataset
        )
        trainer.train()
        return self.get_parameters(), len(self.dataset), {}

    def evaluate(self, parameters, config):
        return 0.0, 0, {}
