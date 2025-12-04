import json
import pandas as pd
import numpy as np
import torch
import argparse
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    RobertaTokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
import evaluate

# 1. Configuration & Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune CodeT5 for Python to Java translation")
    
    parser.add_argument("--model_name", type=str, default="Salesforce/codet5-small", help="Model checkpoint")
    parser.add_argument("--data_file", type=str, default="../../data/bigcodenet_pairs.json", help="Path to training data")
    parser.add_argument("--output_dir", type=str, default="./codet5_fine_tuned", help="Output directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples for testing")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check for device
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Apple Silicon) acceleration.")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA GPU.")
    else:
        device = "cpu"
        print("Using CPU.")

    # 2. Load Data
    print(f"Loading data from {args.data_file}...")
    try:
        with open(args.data_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {args.data_file} not found.")
        # Try relative path if absolute/relative failed
        fallback = os.path.join(os.path.dirname(__file__), args.data_file)
        print(f"Trying fallback path: {fallback}")
        try:
            with open(fallback, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print("Error: Data file not found.")
            exit(1)

    df = pd.DataFrame(data)
    
    if args.limit:
        print(f"Limiting to {args.limit} examples for testing.")
        df = df.head(args.limit)

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    print(f"Training examples: {len(train_df)}")
    print(f"Validation examples: {len(val_df)}")

    # 3. Tokenizer & Dataset
    print(f"Loading model: {args.model_name}")
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)

    class CodeTranslationDataset(Dataset):
        def __init__(self, df, tokenizer, max_length=128):
            self.df = df
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            source_code = "Translate Python to Java: " + row['python']
            target_code = row['java']

            model_inputs = self.tokenizer(
                source_code,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    target_code,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )

            return {
                "input_ids": model_inputs.input_ids.squeeze(),
                "attention_mask": model_inputs.attention_mask.squeeze(),
                "labels": labels.input_ids.squeeze(),
            }

    train_dataset = CodeTranslationDataset(train_df, tokenizer)
    val_dataset = CodeTranslationDataset(val_df, tokenizer)

    # 4. Model
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    model.to(device)

    # 5. Training Arguments
    metric = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        # Replace -100 in the labels
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        decoded_labels = [[l] for l in decoded_labels]
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": result["score"]}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        fp16=False,
        logging_dir='./logs',
        logging_steps=10,
        use_mps_device=(device == "mps"),
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )

    # 6. Train
    print("Starting training...")
    trainer.train()

    # 7. Save
    print(f"Saving model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done!")

if __name__ == "__main__":
    main()
