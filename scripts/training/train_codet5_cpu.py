import json
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    RobertaTokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
import evaluate
import os

# 1. Configuration
MODEL_NAME = "Salesforce/codet5-small"
OUTPUT_DIR = "./codet5_fine_tuned"
DATA_FILE = "bigcodenet_pairs.json"
FALLBACK_DATA_FILE = "../code_pairs.json"

# Check for device (MPS for Mac, else CPU)
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
print(f"Loading data from {DATA_FILE}...")
try:
    with open(DATA_FILE, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Warning: {DATA_FILE} not found. Trying fallback {FALLBACK_DATA_FILE}...")
    try:
        with open(FALLBACK_DATA_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Neither {DATA_FILE} nor {FALLBACK_DATA_FILE} found.")
        exit(1)

df = pd.DataFrame(data)
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
print(f"Training examples: {len(train_df)}")
print(f"Validation examples: {len(val_df)}")

# 3. Tokenizer & Dataset
print(f"Loading model: {MODEL_NAME}")
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

class CodeTranslationDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # CodeT5 is a T5 model, so we can add a prefix if we want, 
        # but for single task fine-tuning it's often fine without.
        # Let's add a simple prefix to be safe and explicit.
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
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
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
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4, # Small batch size for CPU/MPS
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=3,            # Few epochs for quick local test
    predict_with_generate=True,
    fp16=False,                    # MPS/CPU usually better with fp32 for stability or bf16
    logging_dir='./logs',
    logging_steps=10,
    use_mps_device=(device == "mps"),
    report_to="none"               # No wandb
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
)

# 6. Train
print("Starting training...")
trainer.train()

# 7. Save
print(f"Saving model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done!")
