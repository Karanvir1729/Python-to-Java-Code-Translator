import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU

import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import pandas as pd
import json
import sys
import io
from collections import Counter
import math
import re

# Set stdout to handle utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_MODEL = "Salesforce/codet5-small"
FINETUNED_MODEL = os.path.join(BASE_DIR, "..", "..", "models", "codet5_python_to_java", "codet5_fine_tuned")

# Custom sample sizes per dataset
SAMPLE_SIZES = {
    "Easy": 5,        # All of them
    "Advanced": 20,
    "BigCodeNet": 1050
}

DATA_FILES = {
    "Easy": os.path.join(BASE_DIR, "..", "..", "data", "easy_code_pairs.json"),
    "Advanced": os.path.join(BASE_DIR, "..", "..", "data", "advanced_code_pairs.json"),
    "BigCodeNet": os.path.join(BASE_DIR, "..", "..", "data", "bigcodenet_pairs.json")
}

# ======================== BLEU FUNCTIONS ========================
def tokenize(code):
    if not isinstance(code, str): return []
    tokens = re.findall(r'\w+|[^\w\s]', code)
    return [t.lower() for t in tokens]

def ngram_counts(tokens, n):
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    return Counter(ngrams)

def bleu_score(reference, candidate, max_n=4):
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)
    
    if len(cand_tokens) == 0: return 0.0
    
    bp = 1.0 if len(cand_tokens) >= len(ref_tokens) else math.exp(1 - len(ref_tokens) / max(len(cand_tokens), 1))
    
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = ngram_counts(ref_tokens, n)
        cand_ngrams = ngram_counts(cand_tokens, n)
        
        if sum(cand_ngrams.values()) == 0:
            precisions.append(0.0)
            continue
            
        clipped = sum(min(cand_ngrams[ng], ref_ngrams[ng]) for ng in cand_ngrams)
        total = sum(cand_ngrams.values())
        precisions.append(clipped / total if total > 0 else 0.0)
    
    if any(p == 0 for p in precisions):
        precisions = [p + 1e-10 for p in precisions]
    
    log_prec = sum(math.log(p) for p in precisions) / max_n
    return bp * math.exp(log_prec)

# ======================== LOGIC RECALL FUNCTIONS ========================
def get_logic_features(code, lang="python"):
    features = set()
    if not isinstance(code, str): return features
    
    if lang == "python":
        if re.search(r'\binput\(|\bsys\.stdin', code): features.add("input")
        if re.search(r'\bprint\(', code): features.add("print")
        if re.search(r'\breturn\b', code): features.add("return")
        if re.search(r'\bfor\b|\bwhile\b', code): features.add("loop")
        if re.search(r'\bif\b', code): features.add("conditional")
    else:
        if re.search(r'Scanner|BufferedReader|System\.in', code): features.add("input")
        if re.search(r'System\.out\.print', code): features.add("print")
        if re.search(r'\breturn\b', code): features.add("return")
        if re.search(r'\bfor\b|\bwhile\b', code): features.add("loop")
        if re.search(r'\bif\b|\bswitch\b', code): features.add("conditional")
    
    return features

def logic_recall(py_code, java_code):
    py_feats = get_logic_features(py_code, "python")
    java_feats = get_logic_features(java_code, "java")
    
    # Flexible matching: print OR return counts as output
    py_out = "print" in py_feats or "return" in py_feats
    java_out = "print" in java_feats or "return" in java_feats
    
    score = 0.0
    total = 0.0
    
    if py_out:
        total += 1.0
        if java_out: score += 1.0
    if "loop" in py_feats:
        total += 1.0
        if "loop" in java_feats: score += 1.0
    if "conditional" in py_feats:
        total += 1.0
        if "conditional" in java_feats: score += 1.0
    if "input" in py_feats:
        total += 1.0
        if "input" in java_feats: score += 1.0
        
    return score / total if total > 0 else 1.0

# ======================== MODEL FUNCTIONS ========================
def load_model(model_path):
    print(f"Loading model from {model_path}...", flush=True)
    tokenizer = RobertaTokenizer.from_pretrained(PRETRAINED_MODEL)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.to("cpu")
    model.eval()
    return tokenizer, model

def generate_prediction(tokenizer, model, python_code):
    inputs = tokenizer(
        f"translate Python to Java: {python_code}",
        return_tensors="pt",
        max_length=256,
        truncation=True,
        padding="max_length"
    )
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=256,
            num_beams=4,
            early_stopping=True
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ======================== MAIN ========================
def main():
    results = []
    
    # Load data
    all_data = {}
    for ds_name, path in DATA_FILES.items():
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                sample_size = SAMPLE_SIZES.get(ds_name, 25)
                all_data[ds_name] = data[:sample_size]
                print(f"  Loaded {len(all_data[ds_name])} samples from {ds_name}", flush=True)
    
    # Load Pre-trained model
    pt_tokenizer, pt_model = load_model(PRETRAINED_MODEL)
    
    print("\nGenerating Pre-trained predictions...", flush=True)
    for ds_name, data in all_data.items():
        print(f"  {ds_name}...", flush=True)
        for i, item in enumerate(data):
            pred = generate_prediction(pt_tokenizer, pt_model, item['python'])
            results.append({
                "Dataset": ds_name,
                "ID": i + 1,
                "Python": item['python'],
                "Reference Java": item['java'],
                "Pre-trained Pred": pred
            })
    
    # Cleanup
    del pt_model, pt_tokenizer
    torch.cuda.empty_cache()
    
    # Load Fine-tuned model
    ft_tokenizer, ft_model = load_model(FINETUNED_MODEL)
    
    print("\nGenerating Fine-tuned predictions...", flush=True)
    idx = 0
    for ds_name, data in all_data.items():
        print(f"  {ds_name}...", flush=True)
        for i, item in enumerate(data):
            pred = generate_prediction(ft_tokenizer, ft_model, item['python'])
            results[idx]["Fine-tuned Pred"] = pred
            idx += 1
    
    # Cleanup
    del ft_model, ft_tokenizer
    
    # Calculate metrics
    print("\nCalculating metrics...", flush=True)
    for r in results:
        r["Pre-trained BLEU"] = round(bleu_score(r["Reference Java"], r["Pre-trained Pred"]), 4)
        r["Fine-tuned BLEU"] = round(bleu_score(r["Reference Java"], r["Fine-tuned Pred"]), 4)
        r["Pre-trained Logic"] = round(logic_recall(r["Python"], r["Pre-trained Pred"]), 2)
        r["Fine-tuned Logic"] = round(logic_recall(r["Python"], r["Fine-tuned Pred"]), 2)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Print Summary
    print("\n### Summary (25 samples per dataset)")
    summary = df.groupby("Dataset")[["Pre-trained BLEU", "Fine-tuned BLEU", "Pre-trained Logic", "Fine-tuned Logic"]].mean()
    print(summary.to_markdown())
    
    # Save
    df.to_csv("results/evaluation_25_samples.csv", index=False)
    print("\nSaved to results/evaluation_25_samples.csv")

if __name__ == "__main__":
    main()
