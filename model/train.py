import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import time

def load_data(file_path):
    """Loads data from a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)

class TranslationDataset(Dataset):
    """PyTorch dataset for translation tasks."""
    def __init__(self, tokenizer, data, max_length=512):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        start_time = time.time()
        item = self.data[idx]
        input_text = item["input_text"]
        target_text = item["target_text"]

        # Tokenize the input and target texts
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = input_encoding["input_ids"].squeeze()
        attention_mask = input_encoding["attention_mask"].squeeze()
        labels = target_encoding["input_ids"].squeeze()

        end_time = time.time()
        print(f"Time to process item {idx}: {end_time - start_time:.4f} seconds")

        # The T5 model expects the labels to be the input_ids of the target text.
        # The loss function will automatically shift the labels to the right.
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

if __name__ == "__main__":
    # Load the tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained("/Users/karanvirkhanna/BigCodeNet/model")
    model = T5ForConditionalGeneration.from_pretrained("/Users/karanvirkhanna/BigCodeNet/model")

    # Load the training and validation data
    training_data = load_data("/Users/karanvirkhanna/BigCodeNet/model/training_data.json")
    validation_data = load_data("/Users/karanvirkhanna/BigCodeNet/model/validation_data.json")

    # Create the datasets
    train_dataset = TranslationDataset(tokenizer, training_data)
    val_dataset = TranslationDataset(tokenizer, validation_data)

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir="/Users/karanvirkhanna/BigCodeNet/model/results",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="/Users/karanvirkhanna/BigCodeNet/model/logs",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        load_best_model_at_end=True,
    )

    # Create the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Start training
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model("/Users/karanvirkhanna/BigCodeNet/model/fine-tuned-model")