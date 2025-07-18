import flwr
import torch
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast, Trainer, TrainingArguments
from datasets import load_dataset

class DogeSeekAIClient(flwr.client.NumPyClient):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = load_dataset("squad", split="train[:1000]")
        # Preprocess dataset to match DistilBert input
        self.train_dataset = self._preprocess_dataset(self.dataset)

    def _preprocess_dataset(self, dataset):
        def preprocess_function(examples):
            questions = [q.strip() for q in examples["question"]]
            contexts = [c.strip() for c in examples["context"]]
            answers = examples["answers"]
            inputs = self.tokenizer(
                questions,
                contexts,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
                return_offsets_mapping=True
            )
            start_positions = []
            end_positions = []
            for i, answer in enumerate(answers):
                start_char = answer["answer_start"][0]
                end_char = start_char + len(answer["text"][0])
                tokens = inputs["input_ids"][i]
                offsets = inputs["offset_mapping"][i]
                start_token = None
                end_token = None
                for idx, (start, end) in enumerate(offsets):
                    if start <= start_char < end:
                        start_token = idx
                    if start < end_char <= end:
                        end_token = idx
                        break
                if start_token is None or end_token is None:
                    start_token = 0
                    end_token = 0
                start_positions.append(start_token)
                end_positions.append(end_token)
            inputs["start_positions"] = start_positions
            inputs["end_positions"] = end_positions
            del inputs["offset_mapping"]  # Remove offset_mapping to avoid serialization issues
            return inputs

        return dataset.map(preprocess_function, batched=True, remove_columns=["id", "title", "context", "question", "answers"])

    def get_parameters(self, config):
        return [val.detach().cpu().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters, config):
        for param, val in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(val)

    def fit(self, parameters, config):
        self.set_parameters(parameters, config)
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir="./dogeseekai",
                num_train_epochs=1,
                per_device_train_batch_size=4,
                logging_steps=10,
                save_strategy="no",
            ),
            train_dataset=self.train_dataset
        )
        trainer.train()
        return self.get_parameters(config), len(self.train_dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters, config)
        return 0.0, 0, {}