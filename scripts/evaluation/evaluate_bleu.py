import os
import pandas as pd
import json
import sys
import io
from collections import Counter
import math

# Set stdout to handle utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def tokenize(code):
    """Simple tokenization: split by whitespace and punctuation."""
    if not isinstance(code, str):
        return []
    # Split by whitespace, then by common delimiters
    import re
    tokens = re.findall(r'\w+|[^\w\s]', code)
    return [t.lower() for t in tokens]

def ngram_counts(tokens, n):
    """Get n-gram counts from a list of tokens."""
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    return Counter(ngrams)

def bleu_score(reference, candidate, max_n=4):
    """Calculate BLEU score between reference and candidate."""
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)
    
    if len(cand_tokens) == 0:
        return 0.0
    
    # Brevity penalty
    bp = 1.0 if len(cand_tokens) >= len(ref_tokens) else math.exp(1 - len(ref_tokens) / len(cand_tokens))
    
    # N-gram precisions
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = ngram_counts(ref_tokens, n)
        cand_ngrams = ngram_counts(cand_tokens, n)
        
        if sum(cand_ngrams.values()) == 0:
            precisions.append(0.0)
            continue
            
        # Clipped counts
        clipped = sum(min(cand_ngrams[ng], ref_ngrams[ng]) for ng in cand_ngrams)
        total = sum(cand_ngrams.values())
        
        precisions.append(clipped / total if total > 0 else 0.0)
    
    # Geometric mean of precisions (with smoothing)
    if any(p == 0 for p in precisions):
        # Add smoothing for zero precisions
        precisions = [p + 1e-10 for p in precisions]
    
    log_prec = sum(math.log(p) for p in precisions) / max_n
    
    return bp * math.exp(log_prec)

def main():
    # Load comparison data
    df = pd.read_csv("results/comparison_10_samples.csv")
    
    # Load reference Java from original data files
    data_files = {
        "Easy": "data/easy_code_pairs.json",
        "Advanced": "data/advanced_code_pairs.json",
        "BigCodeNet": "data/bigcodenet_pairs.json"
    }
    
    references = {}
    for ds_name, path in data_files.items():
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)[:10]
                references[ds_name] = [item['java'] for item in data]
    
    # Calculate BLEU scores
    pt_bleu_scores = []
    ft_bleu_scores = []
    
    print("Calculating BLEU scores (Prediction vs Reference Java)...\n")
    
    for idx, row in df.iterrows():
        ds = row['Dataset']
        sample_id = int(row['ID']) - 1
        
        ref_java = references.get(ds, [])[sample_id] if sample_id < len(references.get(ds, [])) else ""
        pt_pred = row['Pre-trained Prediction']
        ft_pred = row['Fine-tuned Prediction']
        
        pt_bleu = bleu_score(ref_java, pt_pred)
        ft_bleu = bleu_score(ref_java, ft_pred)
        
        pt_bleu_scores.append(pt_bleu)
        ft_bleu_scores.append(ft_bleu)
    
    df['Pre-trained BLEU'] = pt_bleu_scores
    df['Fine-tuned BLEU'] = ft_bleu_scores
    
    # Print Averages
    print("### BLEU Score: Prediction vs Reference Java")
    avg_df = df.groupby("Dataset")[["Pre-trained BLEU", "Fine-tuned BLEU"]].mean()
    print(avg_df.to_markdown())
    
    # Save
    df.to_csv("results/bleu_score_results.csv", index=False)
    print("\nSaved to results/bleu_score_results.csv")

if __name__ == "__main__":
    main()
